"""Анализаторы на моделях — без единой модели.

Три анализатора Этапа 5 (звуковые события, эмоция речи, лица) отличаются
от остальных тем, что содержат нейросеть. Проверять здесь надо не сеть —
её проверяли те, кто её обучал, — а **всё, что вокруг неё**: раскладку окон,
таблицу соответствий, агрегацию по кадрам, поведение на пустых данных.

Поэтому модели подменены фейками, отвечающими заранее заданными числами.
Тесты идут за миллисекунды, не требуют весов и ломаются ровно тогда,
когда ломается наш код, а не когда меняется чужой чекпойнт.

Настоящие веса проверяются отдельно — `tests/integration/test_ml_features_real.py`.
"""

from __future__ import annotations

import wave
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pytest
from tests.fakes import make_transcript

from audio.events import EVENT_LABELS, EventAnalyzer, events_from_scores, windows_for
from core.domain.analysis import AnalysisContext
from core.domain.media import AudioTrack, Transcript
from core.domain.values import MediaWindow, TimeRange
from core.infra.wav import read_window, track_duration
from speech.emotion import AROUSAL, EmotionAnalyzer, dimensions, speech_spans
from speech.sentiment import exclamation_density, text_sentiment
from vision.faces import CLOSEUP_RATIO, FaceAnalyzer, face_features, pick_frames
from vision.frames import FrameBatch

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------- фейки
class FakeScorer:
    """Классификатор, отвечающий одним и тем же на любой звук."""

    def __init__(self, scores: Mapping[str, float]) -> None:
        self._scores = dict(scores)
        self.calls: list[int] = []

    def probabilities(self, samples: Any, sample_rate: int) -> Mapping[str, float]:
        self.calls.append(len(samples))
        return self._scores


class FakeFaceDetector:
    """Детектор лиц, выдающий заранее заданные прямоугольники по кадрам."""

    def __init__(self, boxes_per_frame: Sequence[Sequence[tuple[float, float, float, float]]]):
        self._boxes = list(boxes_per_frame)
        self.calls = 0

    def detect(self, frame: Any) -> Sequence[tuple[float, float, float, float, float]]:
        boxes = self._boxes[min(self.calls, len(self._boxes) - 1)]
        self.calls += 1
        return [(x, y, w, h, 0.9) for x, y, w, h in boxes]


class FakePeopleDetector:
    def __init__(self, counts: Sequence[int]) -> None:
        self._counts = list(counts)
        self.calls = 0

    def count(self, frame: Any) -> int:
        value = self._counts[min(self.calls, len(self._counts) - 1)]
        self.calls += 1
        return value


def write_wav(path: Path, seconds: float, *, rate: int = 16000, freq: float = 220.0) -> Path:
    """Тон известной длительности — чтобы проверять чтение окон по времени."""
    times = np.arange(int(rate * seconds)) / rate
    samples = (np.sin(2 * np.pi * freq * times) * 0.5 * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    return path


def context_for(path: Path, transcript: Transcript | None = None) -> AnalysisContext:
    return AnalysisContext(
        job_id=uuid4(),
        video_id=uuid4(),
        window=MediaWindow(source_id="file://x", range=TimeRange(0.0, 30.0)),
        workdir=path.parent,
        audio=AudioTrack(path=path, sample_rate=16000, channels=1, duration_sec=30.0),
        transcript=transcript,
    )


def values(features: Mapping[int, Sequence[Any]], index: int) -> dict[str, float]:
    return {value.key: value.value for value in features.get(index, ())}


# ------------------------------------------------------------------ чтение
class TestReadWindow:
    def test_reads_exactly_the_requested_seconds(self, tmp_path: Path) -> None:
        path = write_wav(tmp_path / "tone.wav", 5.0)

        samples = read_window(path, TimeRange(1.0, 3.0))

        assert len(samples) == pytest.approx(2 * 16000, abs=2)

    def test_values_are_normalised(self, tmp_path: Path) -> None:
        """Модели ждут диапазон [-1, 1], а не сырые int16."""
        path = write_wav(tmp_path / "tone.wav", 1.0)

        samples = read_window(path, TimeRange(0.0, 1.0))

        assert float(np.max(np.abs(samples))) <= 1.0
        assert float(np.max(np.abs(samples))) > 0.4

    def test_window_beyond_the_end_is_clipped_not_an_error(self, tmp_path: Path) -> None:
        """Сцена кончается на последнем отсчёте, окно вокруг неё — за ним.
        Это норма, а не повод падать."""
        path = write_wav(tmp_path / "tone.wav", 2.0)

        samples = read_window(path, TimeRange(1.5, 12.0))

        assert 0 < len(samples) <= int(0.5 * 16000) + 1

    def test_window_entirely_outside_gives_nothing(self, tmp_path: Path) -> None:
        path = write_wav(tmp_path / "tone.wav", 2.0)

        assert len(read_window(path, TimeRange(5.0, 6.0))) == 0

    def test_duration_matches(self, tmp_path: Path) -> None:
        path = write_wav(tmp_path / "tone.wav", 3.5)

        assert track_duration(path) == pytest.approx(3.5, abs=0.01)


# ------------------------------------------------------------------ события
class TestEventWindows:
    def test_short_scene_gets_one_window(self) -> None:
        windows = windows_for(TimeRange(10.0, 14.0), window_sec=10.0, max_windows=3)

        assert len(windows) == 1

    def test_window_is_never_negative(self) -> None:
        """Сцена в самом начале файла: окно шире неё, и уехать в минус
        оно не имеет права — читать оттуда нечего."""
        windows = windows_for(TimeRange(0.0, 2.0), window_sec=10.0, max_windows=3)

        assert windows[0].start >= 0.0

    def test_long_scene_is_covered_by_several_windows(self) -> None:
        windows = windows_for(TimeRange(0.0, 60.0), window_sec=10.0, max_windows=3)

        assert len(windows) == 3
        assert windows[0].start == pytest.approx(0.0)
        assert windows[-1].end == pytest.approx(60.0)

    def test_limit_is_respected(self) -> None:
        """Двухчасовой фильм — это тысячи сцен, и сплошное покрытие
        окнами превратило бы этап в часы работы."""
        windows = windows_for(TimeRange(0.0, 600.0), window_sec=10.0, max_windows=3)

        assert len(windows) == 3


class TestEventMapping:
    def test_group_takes_the_strongest_label(self) -> None:
        """Модель размазывает уверенность между «Laughter» и «Giggle»;
        по одной метке смешная сцена набрала бы вдвое меньше."""
        found = events_from_scores({"Laughter": 0.3, "Giggle": 0.7}, threshold=0.1)

        assert {value.key: value.value for value in found}["audio.event.laughter"] == 0.7

    def test_quiet_labels_are_not_written_at_all(self) -> None:
        """527 вероятностей около нуля в базе — это мегабайты мусора."""
        found = events_from_scores({"Laughter": 0.02, "Music": 0.9}, threshold=0.1)

        assert [value.key for value in found] == ["audio.event.music"]

    def test_unknown_labels_are_ignored(self) -> None:
        assert events_from_scores({"Sneeze": 0.99}, threshold=0.1) == []

    def test_all_five_events_are_covered(self) -> None:
        """Реестр признаков док 05 обещает ровно пять событий."""
        assert set(EVENT_LABELS) == {"laughter", "scream", "applause", "music", "impact"}


class TestEventAnalyzer:
    async def test_features_land_on_the_right_scene(self, tmp_path: Path) -> None:
        path = write_wav(tmp_path / "a.wav", 30.0)
        scorer = FakeScorer({"Applause": 0.8})
        analyzer = EventAnalyzer(scorer=scorer, window_sec=5.0, max_windows=1, threshold=0.1)

        features = await analyzer.analyze(
            context_for(path), [TimeRange(0.0, 10.0), TimeRange(10.0, 20.0)]
        )

        assert values(features, 0)["audio.event.applause"] == 0.8
        assert values(features, 1)["audio.event.applause"] == 0.8

    async def test_maximum_over_windows_not_average(self, tmp_path: Path) -> None:
        """Секундный смех в тридцатисекундной сцене — это смех.
        Усреднение по окнам растворило бы его."""
        path = write_wav(tmp_path / "a.wav", 40.0)

        class Rising:
            def __init__(self) -> None:
                self.values = [0.1, 0.9, 0.2]
                self.index = 0

            def probabilities(self, samples: Any, sample_rate: int) -> Mapping[str, float]:
                value = self.values[min(self.index, len(self.values) - 1)]
                self.index += 1
                return {"Laughter": value}

        analyzer = EventAnalyzer(scorer=Rising(), window_sec=10.0, max_windows=3, threshold=0.05)

        features = await analyzer.analyze(context_for(path), [TimeRange(0.0, 40.0)])

        assert values(features, 0)["audio.event.laughter"] == 0.9

    async def test_silence_vetoes_the_window(self, tmp_path: Path) -> None:
        """Замер на реальных весах: на ровной тишине модель выдаёт
        ``Silence`` 0.61 и ``Music`` 0.27 — то есть выше порога. Без вето
        каждая тихая сцена получала бы признак «музыка»."""
        path = write_wav(tmp_path / "a.wav", 20.0)
        analyzer = EventAnalyzer(
            scorer=FakeScorer({"Silence": 0.61, "Music": 0.27}),
            window_sec=10.0,
            max_windows=1,
            threshold=0.1,
        )

        features = await analyzer.analyze(context_for(path), [TimeRange(0.0, 10.0)])

        assert features == {}

    async def test_music_without_silence_is_kept(self, tmp_path: Path) -> None:
        """Вето срабатывает по тишине, а не по музыке — иначе оно
        выбрасывало бы ровно то, ради чего анализатор написан."""
        path = write_wav(tmp_path / "a.wav", 20.0)
        analyzer = EventAnalyzer(
            scorer=FakeScorer({"Silence": 0.05, "Music": 0.27}),
            window_sec=10.0,
            max_windows=1,
            threshold=0.1,
        )

        features = await analyzer.analyze(context_for(path), [TimeRange(0.0, 10.0)])

        assert values(features, 0)["audio.event.music"] == 0.27

    async def test_without_audio_nothing_happens(self, tmp_path: Path) -> None:
        context = AnalysisContext(
            job_id=uuid4(),
            video_id=uuid4(),
            window=MediaWindow(source_id="file://x", range=TimeRange(0.0, 10.0)),
            workdir=tmp_path,
        )

        assert await EventAnalyzer(scorer=FakeScorer({})).analyze(context, [TimeRange(0, 5)]) == {}


# ------------------------------------------------------------------ эмоция
class TestSpeechSpans:
    def test_limited_by_total_speech_time(self) -> None:
        """Вся речь двухчасового фильма через wav2vec2 — это часы работы."""
        transcript = make_transcript(
            (0.0, 4.0, "первая реплика"),
            (4.0, 8.0, "вторая реплика"),
            (8.0, 12.0, "третья реплика"),
        )

        spans = speech_spans(transcript, TimeRange(0.0, 12.0), limit_sec=5.0)

        assert sum(span.duration for span in spans) == pytest.approx(5.0, abs=0.01)

    def test_only_the_part_inside_the_scene(self) -> None:
        transcript = make_transcript((8.0, 16.0, "реплика через границу"))

        spans = speech_spans(transcript, TimeRange(10.0, 20.0), limit_sec=10.0)

        assert spans[0].start == pytest.approx(10.0)

    def test_scene_without_speech(self) -> None:
        transcript = make_transcript((50.0, 55.0, "далеко"))

        assert speech_spans(transcript, TimeRange(0.0, 10.0)) == []


class TestDimensions:
    def test_confident_anger_is_high_arousal_and_negative(self) -> None:
        arousal, valence = dimensions({"ang": 0.95, "neu": 0.05, "hap": 0.0, "sad": 0.0})

        assert arousal > 0.8
        assert valence < -0.5

    def test_neutral_is_calm_and_neither_positive_nor_negative(self) -> None:
        arousal, valence = dimensions({"neu": 1.0, "hap": 0.0, "ang": 0.0, "sad": 0.0})

        assert arousal == pytest.approx(AROUSAL["neu"])
        assert valence == pytest.approx(0.0)

    def test_mixture_lands_between_not_at_a_corner(self) -> None:
        """Половина радости и половина грусти — это средний уровень
        возбуждения, а не прыжок в одну из точек таблицы."""
        arousal, valence = dimensions({"hap": 0.5, "sad": 0.5, "neu": 0.0, "ang": 0.0})

        assert 0.4 < arousal < 0.7
        assert -0.1 < valence < 0.3

    def test_unnormalised_probabilities_do_not_break_the_scale(self) -> None:
        arousal, _ = dimensions({"ang": 2.0, "neu": 2.0})

        assert 0.0 <= arousal <= 1.0


class TestSentiment:
    def test_positive_text(self) -> None:
        value = text_sentiment("Это отличный день, я счастлив и всех люблю")

        assert value is not None and value > 0.5

    def test_negative_text(self) -> None:
        value = text_sentiment("Какой ужас, ненавижу эту катастрофу")

        assert value is not None and value < -0.5

    def test_no_signal_means_no_feature(self) -> None:
        """Ноль сказал бы скорингу «нейтрально», а правда — «не знаю»."""
        assert text_sentiment("Он вышел из комнаты и закрыл дверь") is None

    def test_prefix_matching_handles_russian_forms(self) -> None:
        """Словарь короткий именно потому, что сравнение идёт по началу слова."""
        assert text_sentiment("люблю") == text_sentiment("любим")

    def test_exclamations_count_marks_and_caps(self) -> None:
        assert exclamation_density("ВСЁ пропало!") > 0.0

    def test_calm_text_has_no_exclamations(self) -> None:
        assert exclamation_density("он тихо сказал что уходит") == 0.0

    def test_density_is_bounded(self) -> None:
        assert exclamation_density("А?! Б?! В?!") <= 1.0

    def test_empty_text(self) -> None:
        assert exclamation_density("") == 0.0
        assert text_sentiment("") is None


class TestEmotionAnalyzer:
    async def test_writes_dimensions_and_class_probabilities(self, tmp_path: Path) -> None:
        path = write_wav(tmp_path / "a.wav", 20.0)
        transcript = make_transcript((0.0, 5.0, "какой отличный день"))
        analyzer = EmotionAnalyzer(
            scorer=FakeScorer({"neu": 0.1, "hap": 0.8, "ang": 0.05, "sad": 0.05}),
            max_speech_sec=5.0,
        )

        features = await analyzer.analyze(context_for(path, transcript), [TimeRange(0.0, 10.0)])

        scene = values(features, 0)
        assert scene["speech.emotion.hap"] == 0.8
        assert scene["speech.emotion.arousal"] > 0.6
        assert scene["speech.emotion.valence"] > 0.4

    async def test_text_features_are_computed_even_without_enough_speech(
        self, tmp_path: Path
    ) -> None:
        """Полсекунды речи для модели — шум, а буквы никуда не делись."""
        path = write_wav(tmp_path / "a.wav", 20.0)
        transcript = make_transcript((1.0, 1.3, "ужас какой!"))
        scorer = FakeScorer({"neu": 1.0})
        analyzer = EmotionAnalyzer(scorer=scorer, max_speech_sec=5.0)

        features = await analyzer.analyze(context_for(path, transcript), [TimeRange(0.0, 10.0)])

        scene = values(features, 0)
        assert "speech.text.sentiment" in scene
        assert "speech.emotion.arousal" not in scene
        assert scorer.calls == [], "модель не должна вызываться на обрывке"

    async def test_without_transcript_nothing_happens(self, tmp_path: Path) -> None:
        path = write_wav(tmp_path / "a.wav", 5.0)

        features = await EmotionAnalyzer(scorer=FakeScorer({"neu": 1.0})).analyze(
            context_for(path, None), [TimeRange(0.0, 5.0)]
        )

        assert features == {}


# -------------------------------------------------------------------- лица
class TestPickFrames:
    def make_batch(self, count: int) -> FrameBatch:
        frames = tuple(np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(count))
        return FrameBatch(
            scene=TimeRange(0.0, 5.0),
            times=tuple(float(i) for i in range(count)),
            frames=frames,
            color=frames,
        )

    def test_subsample_is_limited(self) -> None:
        """Лица меняются медленнее движения: разбирать все кадры — значит
        платить вчетверо за ту же информацию."""
        assert len(pick_frames(self.make_batch(30), 5)) == 5

    def test_short_scene_gives_everything_it_has(self) -> None:
        assert len(pick_frames(self.make_batch(3), 5)) == 3

    def test_frames_are_spread_over_the_scene(self) -> None:
        batch = self.make_batch(10)
        picked = pick_frames(batch, 2)

        assert picked[0] is batch.color[0]
        assert picked[1] is batch.color[5]

    def test_empty_batch(self) -> None:
        assert pick_frames(self.make_batch(0), 5) == []


class TestFaceFeatures:
    def test_max_and_mean_answer_different_questions(self) -> None:
        found = {value.key: value.value for value in face_features([0, 0, 3], [], [])}

        assert found["vision.faces.count_max"] == 3.0
        assert found["vision.faces.count_mean"] == 1.0

    def test_closeup_flag_follows_the_largest_face(self) -> None:
        big = {value.key: value.value for value in face_features([1], [CLOSEUP_RATIO + 0.05], [])}
        small = {value.key: value.value for value in face_features([1], [0.01], [])}

        assert big["vision.faces.closeup"] == 1.0
        assert small["vision.faces.closeup"] == 0.0

    def test_area_takes_the_maximum_not_the_average(self) -> None:
        """Крупный план в середине сцены — событие; усреднение с общими
        планами стёрло бы его."""
        found = {value.key: value.value for value in face_features([1, 1], [0.02, 0.30], [])}

        assert found["vision.faces.area_ratio"] == 0.30

    def test_nothing_found_means_no_area_features(self) -> None:
        keys = {value.key for value in face_features([0, 0], [], [])}

        assert "vision.faces.area_ratio" not in keys
        assert "vision.faces.count_max" in keys

    def test_people_are_separate_from_faces(self) -> None:
        """Человек со спины лицом не считается — для этого и нужен
        отдельный детектор."""
        found = {value.key: value.value for value in face_features([0], [], [4])}

        assert found["vision.people.count"] == 4.0


class TestFaceAnalyzer:
    class FakeSource:
        def __init__(self, batch: FrameBatch) -> None:
            self._batch = batch

        async def scene_frames(self, scene: TimeRange) -> FrameBatch:
            return self._batch

    def batch_of(self, count: int, size: int = 100) -> FrameBatch:
        frames = tuple(np.zeros((size, size, 3), dtype=np.uint8) for _ in range(count))
        return FrameBatch(
            scene=TimeRange(0.0, 5.0),
            times=tuple(float(i) for i in range(count)),
            frames=frames,
            color=frames,
        )

    async def test_area_ratio_is_a_share_of_the_frame(self, tmp_path: Path) -> None:
        # Кадр 100×100, лицо 30×40 — это 12 % площади.
        faces = FakeFaceDetector([[(0.0, 0.0, 30.0, 40.0)]])
        analyzer = FaceAnalyzer(
            source=self.FakeSource(self.batch_of(1)),  # type: ignore[arg-type]
            faces=faces,
            people=FakePeopleDetector([1]),
            frames_per_scene=1,
        )
        context = AnalysisContext(
            job_id=uuid4(),
            video_id=uuid4(),
            window=MediaWindow(source_id="file://x", range=TimeRange(0.0, 5.0)),
            workdir=tmp_path,
        )

        features = await analyzer.analyze(context, [TimeRange(0.0, 5.0)])

        assert values(features, 0)["vision.faces.area_ratio"] == pytest.approx(0.12, abs=0.001)

    async def test_detectors_are_called_once_per_picked_frame(self, tmp_path: Path) -> None:
        faces = FakeFaceDetector([[]])
        people = FakePeopleDetector([0])
        analyzer = FaceAnalyzer(
            source=self.FakeSource(self.batch_of(20)),  # type: ignore[arg-type]
            faces=faces,
            people=people,
            frames_per_scene=4,
        )
        context = AnalysisContext(
            job_id=uuid4(),
            video_id=uuid4(),
            window=MediaWindow(source_id="file://x", range=TimeRange(0.0, 5.0)),
            workdir=tmp_path,
        )

        await analyzer.analyze(context, [TimeRange(0.0, 5.0)])

        assert faces.calls == 4
        assert people.calls == 4

    async def test_without_video_nothing_happens(self, tmp_path: Path) -> None:
        context = AnalysisContext(
            job_id=uuid4(),
            video_id=uuid4(),
            window=MediaWindow(source_id="file://x", range=TimeRange(0.0, 5.0)),
            workdir=tmp_path,
        )

        assert (
            await FaceAnalyzer(faces=FakeFaceDetector([[]])).analyze(context, [TimeRange(0, 5)])
            == {}
        )
