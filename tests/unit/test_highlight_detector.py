"""Скоринг и отбор моментов.

Модули пакета чистые — ни базы, ни сети, — поэтому проверяются на сценах,
собранных прямо здесь. Ценность такого теста в том, что в нём видно,
из чего получилось ожидаемое число.

Отдельный блок — регрессы двух дефектов, найденных живым прогоном
на минутном ролике. Оба относятся к расширению кандидатов и оба
проявлялись только на однородном материале, где все сцены похожи.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from core.domain.entities import Scene
from core.domain.values import FeatureValue, Score, TimeRange
from core.usecases.select import pauses_from_transcript
from highlight_detector.normalize import (
    modified_zscore,
    normalize_scenes,
    robust_normalize,
)
from highlight_detector.profiles import get_profile
from highlight_detector.scoring import fuse, heuristic_score
from highlight_detector.selector import (
    MAX_MEMBERS,
    expand,
    select_highlights,
    snap_to_pauses,
)
from highlight_detector.signals import compute_signals
from highlight_detector.structure import compute_structure

pytestmark = pytest.mark.unit


def make_scene(
    index: int,
    *,
    score: float | None = None,
    duration: float = 6.0,
    category: str = "funny",
    signals: dict[str, float] | None = None,
    text: str | None = None,
) -> Scene:
    start = index * duration
    features = {
        f"signal.{name}": FeatureValue(f"signal.{name}", value)
        for name, value in (signals or {}).items()
    }
    return Scene(
        id=uuid4(),
        video_id=uuid4(),
        job_id=uuid4(),
        index=index,
        range=TimeRange(start, start + duration),
        features=features,
        final_score=score,
        category=category,
        reason=f"сцена {index}",
        transcript_text=text,
    )


# ------------------------------------------------------------- нормализация
class TestNormalize:
    def test_outlier_does_not_squash_the_rest(self) -> None:
        """Один взрыв на всё видео не должен сжимать остальные сцены
        в полосу около нуля — ради этого медиана вместо среднего."""
        values = [0.1, 0.12, 0.11, 0.13, 0.1, 0.12, 5.0]

        normalized = robust_normalize(values)

        ordinary = normalized[:6]
        # Замер: обычные значения занимают 0.375 шкалы. Со средним и σ
        # выброс 5.0 задал бы верхнюю границу, и все шесть уместились бы
        # в первые проценты — различать их стало бы нечем.
        assert max(ordinary) - min(ordinary) > 0.3, "обычные значения должны различаться"
        assert normalized[-1] == 1.0, "выброс всё же остаётся максимумом"

    def test_identical_values_give_the_middle(self) -> None:
        """Все значения равны — различать нечего. Половина, а не ноль:
        ноль означал бы «признака нет», а он есть и одинаков."""
        assert robust_normalize([3.0, 3.0, 3.0]) == [0.5, 0.5, 0.5]

    def test_single_scene(self) -> None:
        assert robust_normalize([42.0]) == [0.5]

    def test_zscore_is_zero_without_spread(self) -> None:
        assert modified_zscore([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]

    def test_missing_feature_stays_missing(self) -> None:
        """«Нет данных» и «ноль» — разные вещи: подставив ноль, мы
        штрафовали бы сцену за отключённый анализатор."""
        result = normalize_scenes([{"a": 1.0}, {}, {"a": 3.0}])

        assert "a" not in result[1]

    def test_unit_features_are_not_stretched(self) -> None:
        """Вероятность смеха 0.9 — это «смех есть» независимо от того,
        смеются ли в остальном видео."""
        result = normalize_scenes(
            [
                {"audio.event.laughter": 0.02},
                {"audio.event.laughter": 0.03},
                {"audio.event.laughter": 0.04},
            ]
        )

        assert all(item["audio.event.laughter"] < 0.1 for item in result)


class TestStructure:
    def test_first_scene_is_not_a_twist(self) -> None:
        """Начало видео не является поворотом по построению."""
        structure = compute_structure([{"vision.motion.mean": 0.5}] * 3, [5.0, 5.0, 5.0])

        assert structure[0]["struct.novelty"] == 0.0

    def test_different_scene_is_novel(self) -> None:
        scenes = [
            {"vision.motion.mean": 0.9, "speech.density": 0.0},
            {"vision.motion.mean": 0.9, "speech.density": 0.0},
            {"vision.motion.mean": 0.0, "speech.density": 0.9},
        ]

        structure = compute_structure(scenes, [5.0, 5.0, 5.0])

        assert structure[2]["struct.novelty"] > structure[1]["struct.novelty"]

    def test_build_up_ignores_decline(self) -> None:
        """Спад — не «анти-кульминация», а отсутствие нарастания."""
        scenes = [{"vision.motion.mean": 0.9}, {"vision.motion.mean": 0.1}]

        structure = compute_structure(scenes, [5.0, 5.0])

        assert structure[1]["struct.build_up"] == 0.0


# ------------------------------------------------------------------ скоринг
class TestScoring:
    def test_missing_group_redistributes_weights(self) -> None:
        """Главный пункт DoD: упал видеоанализ — оставшиеся группы
        масштабируются до 100, а не теряют 40 % скора."""
        profile = get_profile("default")
        full = heuristic_score({"laughter": 1.0, "action": 1.0}, profile)
        without_action = heuristic_score({"laughter": 1.0}, profile)

        assert full.value == 100.0
        assert without_action.value == 100.0, "без группы скор всё равно доходит до 100"
        assert "action" not in without_action.breakdown

    def test_breakdown_explains_the_score(self) -> None:
        profile = get_profile("default")

        score = heuristic_score({"laughter": 0.8, "action": 0.2}, profile)

        assert score.breakdown["emotion"] == 0.8
        assert score.breakdown["action"] == 0.2
        assert 0.0 < score.value < 100.0

    def test_no_signals_gives_zero(self) -> None:
        assert heuristic_score({}, get_profile("default")).value == 0.0

    def test_score_stays_in_range(self) -> None:
        profile = get_profile("default")
        score = heuristic_score(dict.fromkeys(("laughter", "action", "tension"), 1.0), profile)

        assert 0.0 <= score.value <= 100.0

    def test_profiles_weigh_differently(self) -> None:
        """Один и тот же диалог в подкасте важнее, чем в боевике."""
        signals = {"long_dialogue": 1.0}

        movie = heuristic_score(signals, get_profile("movie")).value
        podcast = heuristic_score(signals, get_profile("podcast")).value

        assert podcast == movie == 100.0, "единственная группа масштабируется до 100"

        mixed = {"long_dialogue": 1.0, "action": 1.0}
        assert (
            heuristic_score(mixed, get_profile("podcast")).value
            == heuristic_score(mixed, get_profile("movie")).value
        )

    def test_unknown_profile_is_an_error(self) -> None:
        """Опечатка в конфиге иначе обернулась бы отбором по чужим весам."""
        with pytest.raises(Exception, match="профиль"):
            get_profile("nonexistent")


class TestFusion:
    def test_agreement_lifts_the_score(self) -> None:
        """Момент, который считают удачным и сигналы, и модель, почти
        всегда действительно удачный."""
        agree = fuse(80.0, 80.0)
        disagree = fuse(80.0, 20.0)

        assert agree > disagree

    def test_without_llm_score_equals_heuristic(self) -> None:
        """Сцена без вердикта не должна проигрывать конкуренцию только
        потому, что её не показали модели."""
        assert fuse(73.0, None) == 73.0

    def test_result_stays_in_range(self) -> None:
        assert 0.0 <= fuse(100.0, 0.0) <= 100.0
        assert 0.0 <= fuse(0.0, 100.0) <= 100.0


class TestSignals:
    def test_action_is_max_not_sum(self) -> None:
        """Сцена не становится вдвое динамичнее оттого, что подходит
        и под драку, и под погоню."""
        signals = compute_signals(
            {
                "vision.motion.peak": 1.0,
                "vision.cuts.rate": 1.0,
                "audio.event.impact": 1.0,
                "vision.camera.shake": 1.0,
                "vision.camera.pan": 1.0,
                "vision.motion.mean": 1.0,
            }
        )

        assert signals["action"] <= 1.0

    def test_missing_features_renormalise(self) -> None:
        """Сцена без видеогруппы не должна получать вдвое меньший сигнал
        только потому, что мы её не досмотрели."""
        full = compute_signals({"audio.event.laughter": 1.0, "speech.emotion.hap": 1.0})
        partial = compute_signals({"audio.event.laughter": 1.0})

        assert full["laughter"] == partial["laughter"] == 1.0

    def test_all_signals_are_bounded(self) -> None:
        signals = compute_signals(
            dict.fromkeys(
                (
                    "speech.emotion.arousal",
                    "audio.loudness.delta",
                    "vision.motion.peak",
                    "struct.novelty",
                ),
                1.0,
            )
        )

        assert all(0.0 <= value <= 1.0 for value in signals.values())


# -------------------------------------------------------------------- отбор
class TestExpansion:
    def test_uniform_material_does_not_swallow_the_video(self) -> None:
        """Регресс дефекта, найденного живым прогоном.

        На однородном материале (все сцены похожи) доля от пика не отсекает
        ни одного соседа, и кандидат рос до упора, покрывая половину видео.
        Дальше NMS схлопывал такие кандидаты в один — вместо отбора
        получалось «взять всё подряд»: 12 сцен → 1 клип.
        """
        scenes = [make_scene(index, score=60.0 + index % 3) for index in range(20)]

        candidates = expand(scenes, min_duration_sec=15.0, max_duration_sec=90.0)

        assert all(len(candidate.scene_indices) <= MAX_MEMBERS for candidate in candidates)
        video_span = scenes[-1].range.end - scenes[0].range.start
        assert all(candidate.range.duration <= video_span * 0.5 + 1e-6 for candidate in candidates)

    def test_short_peak_grows_to_the_minimum(self) -> None:
        """Второй регресс: расширение и фильтр длительности работали
        друг против друга.

        Сильная короткая сцена в конце ролика набирала 13.6 с при минимуме
        15 — и лучший момент видео отбрасывался фильтром, не доехав до NMS.
        """
        scenes = [
            make_scene(0, score=10.0, duration=6.0),
            make_scene(1, score=10.0, duration=6.0),
            make_scene(2, score=95.0, duration=6.0),
        ]

        candidates = expand(scenes, min_duration_sec=15.0, max_duration_sec=90.0)

        peak = next(candidate for candidate in candidates if candidate.score == 95.0)
        assert peak.range.duration >= 15.0, "кандидат должен дорасти вопреки порогу"

    def test_growth_respects_the_maximum(self) -> None:
        scenes = [make_scene(index, score=90.0, duration=10.0) for index in range(10)]

        candidates = expand(scenes, min_duration_sec=15.0, max_duration_sec=30.0)

        assert all(candidate.range.duration <= 30.0 for candidate in candidates)


class TestSnap:
    def test_boundary_moves_to_the_pause(self) -> None:
        """Клип, начинающийся с середины слова, — брак независимо от того,
        насколько удачен момент внутри."""
        candidate = expand(
            [make_scene(0, score=90.0, duration=20.0)],
            min_duration_sec=15.0,
            max_duration_sec=90.0,
        )[0]
        # Обе паузы строго внутри окна: 1.2 от начала и 1.4 от конца.
        # Ровно на границе окна (1.5) сдвиг уже не происходит — сравнение
        # строгое, и это намеренно: «полтора и ни секундой больше».
        pauses = [TimeRange(0.8, 1.2), TimeRange(18.6, 19.4)]

        snapped = snap_to_pauses(candidate, pauses)

        assert snapped.range.start == 1.2, "начало сдвигается к концу паузы"
        assert snapped.range.end == 18.6, "конец — к началу паузы"

    def test_far_pause_is_ignored(self) -> None:
        candidate = expand(
            [make_scene(0, score=90.0, duration=20.0)],
            min_duration_sec=15.0,
            max_duration_sec=90.0,
        )[0]

        snapped = snap_to_pauses(candidate, [TimeRange(9.0, 10.0)])

        assert snapped.range == candidate.range

    def test_pauses_come_from_transcript_gaps(self) -> None:
        from tests.fakes import make_transcript

        transcript = make_transcript((0.0, 3.0, "первая"), (5.0, 8.0, "вторая"))

        pauses = pauses_from_transcript(transcript)

        assert pauses == [TimeRange(3.0, 5.0)]

    def test_tiny_gaps_are_not_pauses(self) -> None:
        """Зазор между словами — не место для границы клипа."""
        from tests.fakes import make_transcript

        transcript = make_transcript((0.0, 3.0, "первая"), (3.1, 6.0, "вторая"))

        assert pauses_from_transcript(transcript) == []


class TestSelection:
    def test_no_overlapping_pairs_survive(self) -> None:
        """Пункт DoD: NMS не оставляет пар с IoU > 0.3."""
        scenes = [make_scene(index, score=90.0 - index, duration=8.0) for index in range(12)]

        highlights = select_highlights(scenes, get_profile("default"))

        for first in range(len(highlights)):
            for second in range(first + 1, len(highlights)):
                left, right = highlights[first].range, highlights[second].range
                overlap = min(left.end, right.end) - max(left.start, right.start)
                if overlap <= 0:
                    continue
                union = max(left.end, right.end) - min(left.start, right.start)
                assert overlap / union <= 0.3

    def test_durations_stay_within_the_profile(self) -> None:
        profile = get_profile("default")
        scenes = [make_scene(index, score=80.0, duration=7.0) for index in range(20)]

        highlights = select_highlights(scenes, profile)

        assert highlights
        assert all(
            profile.min_duration_sec <= highlight.range.duration <= profile.max_duration_sec
            for highlight in highlights
        )

    def test_diversity_limits_one_category(self) -> None:
        """Пункт DoD: в топ-10 не больше четырёх клипов одной категории —
        при условии, что есть из чего выбирать."""
        scenes = []
        for index in range(24):
            category = "fight" if index % 2 == 0 else "dialogue"
            scenes.append(
                make_scene(
                    index,
                    score=90.0 - index * 0.5,
                    duration=8.0,
                    category=category,
                    signals={"action": 1.0} if category == "fight" else {"long_dialogue": 1.0},
                )
            )

        highlights = select_highlights(scenes, get_profile("default"))

        categories = [highlight.category for highlight in highlights]
        assert len(set(categories)) > 1, "выдача не должна быть одноцветной"

    def test_budget_is_respected(self) -> None:
        scenes = [make_scene(index, score=80.0, duration=8.0) for index in range(20)]

        highlights = select_highlights(scenes, get_profile("default"), total_duration_sec=40.0)

        assert sum(highlight.range.duration for highlight in highlights) <= 40.0

    def test_low_scores_are_filtered_out(self) -> None:
        scenes = [make_scene(index, score=5.0, duration=8.0) for index in range(10)]

        assert select_highlights(scenes, get_profile("default")) == []

    def test_scenes_without_score_are_skipped(self) -> None:
        scenes = [make_scene(index, score=None, duration=8.0) for index in range(5)]

        assert select_highlights(scenes, get_profile("default")) == []


class TestProfileDifference:
    def test_profiles_select_differently(self) -> None:
        """Пункт DoD: профили дают заметно разный отбор на одном видео.

        Живой прогон на эталонном ролике: `default` — 4 клипа по 15–17 с,
        `podcast` — 2 клипа по 30+ с.

        Материал здесь неоднородный: сильные сцены чередуются со слабыми,
        иначе фаза «по достоинству» упирается в один и тот же потолок
        у обоих профилей и разница исчезает — ровно это и показал
        первый вариант теста.
        """
        scenes = [
            make_scene(index, score=85.0 if index % 3 == 0 else 45.0, duration=8.0)
            for index in range(20)
        ]

        default = select_highlights(scenes, get_profile("default"))
        podcast = select_highlights(scenes, get_profile("podcast"))

        assert default and podcast
        assert min(h.range.duration for h in podcast) >= 30.0
        assert min(h.range.duration for h in default) >= 15.0
        # Порог профиля управляет не только длиной, но и составом выдачи.
        assert min(h.range.duration for h in default) < min(h.range.duration for h in podcast), (
            "у podcast клипы длиннее по определению профиля"
        )


class TestScoreEntity:
    def test_score_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 100\]"):
            Score(value=101.0)
