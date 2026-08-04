"""LLM-слой без единого сетевого вызова.

Модель здесь подменена фейком, и это не компромисс, а единственный способ
проверить главное: **как система ведёт себя, когда модель ошибается**.
Настоящая модель ошибается редко и непредсказуемо, фейк — ровно так,
как нужно тесту: сбитая нумерация, чужие тайм-коды, отказ сервиса.

Живой вызов проверяется отдельно, по маркеру ``llm``.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID, uuid4

import pytest
from tests.fakes import FakeSceneRepository, FakeUnitOfWork, uow_factory

from core.domain.analysis import LLMVerdict, RankingOutcome, SceneSummary, VideoContext
from core.domain.entities import Job, Scene, Video
from core.domain.enums import JobStage, JobStatus, SourceKind, VideoStatus
from core.domain.errors import ExternalServiceError
from core.domain.media import MediaInfo
from core.domain.values import FeatureValue, TimeRange
from core.usecases.rank import RankScenes
from llm.batching import rank_in_batches, split_batches
from llm.schemas import CATEGORIES, RankingResponse, SceneVerdict
from llm.summarizer import (
    MAX_SPEECH_CHARS,
    format_video_context,
    is_worth_ranking,
    prepare,
    render,
    summarize,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------- фейки
class FakeProvider:
    """Провайдер, отвечающий заранее заданными вердиктами."""

    def __init__(
        self,
        verdicts: Sequence[SceneVerdict] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self._verdicts = list(verdicts or ())
        self._error = error
        self.calls: list[int] = []
        self.usage = _FakeUsage()

    async def rank_scenes(
        self,
        scenes: Sequence[SceneSummary],
        context: VideoContext,
    ) -> Sequence[LLMVerdict]:
        self.calls.append(len(scenes))
        if self._error is not None:
            raise self._error

        by_index = {summary.index: summary for summary in scenes}
        return [
            verdict.to_domain(by_index[verdict.scene_index].range)
            for verdict in self._verdicts
            if verdict.scene_index in by_index
        ]

    async def generate_metadata(self, highlight, transcript, language=None):  # type: ignore[no-untyped-def]
        raise NotImplementedError


class _FakeUsage:
    def as_dict(self) -> dict[str, float | int]:
        return {"requests": 1, "prompt_tokens": 100, "completion_tokens": 20, "cost_usd": 0.0}


class FakeRanker:
    """Ранжировщик, отдающий готовый результат или бросающий ошибку."""

    def __init__(
        self,
        outcome: RankingOutcome | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self._outcome = outcome or RankingOutcome()
        self._error = error
        self.calls = 0

    async def rank(self, scenes, context):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._outcome


def make_scene(
    index: int,
    *,
    job_id: UUID,
    video_id: UUID,
    text: str | None = None,
    features: dict[str, float] | None = None,
) -> Scene:
    return Scene(
        id=uuid4(),
        video_id=video_id,
        job_id=job_id,
        index=index,
        range=TimeRange(index * 10.0, index * 10.0 + 8.0),
        features={key: FeatureValue(key, value) for key, value in (features or {}).items()},
        transcript_text=text,
    )


def verdict_for(scene: Scene, *, score: int = 80, category: str = "funny") -> SceneVerdict:
    return SceneVerdict(
        scene_index=scene.index,
        start=_timecode(scene.range.start),
        end=_timecode(scene.range.end),
        score=score,
        category=category,  # type: ignore[arg-type]
        reason="потому что",
    )


def _timecode(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


# ------------------------------------------------------------------- схемы
class TestSchemas:
    def test_timecodes_from_the_answer_are_not_used_as_boundaries(self) -> None:
        """Границы клипа — наши. Ответ модели может их слегка сместить,
        и доверять ему как источнику правды нельзя."""
        scene_range = TimeRange(10.0, 18.0)
        verdict = SceneVerdict(
            scene_index=3,
            start="00:00:11",
            end="00:00:17",
            score=70,
            category="dialogue",
            reason="x",
        )

        domain = verdict.to_domain(scene_range)

        assert domain.range == scene_range

    def test_timecodes_close_to_the_scene_are_accepted(self) -> None:
        verdict = SceneVerdict(
            scene_index=0, start="00:00:09", end="00:00:19", score=50, category="other", reason="x"
        )

        assert verdict.timecodes_match(TimeRange(10.0, 18.0), tolerance=2.0)

    def test_timecodes_from_another_scene_are_rejected(self) -> None:
        """Не «неточность», а сбитая нумерация: вердикт относится к другой
        сцене, и засчитывать его нельзя."""
        verdict = SceneVerdict(
            scene_index=0, start="00:05:00", end="00:05:08", score=90, category="action", reason="x"
        )

        assert not verdict.timecodes_match(TimeRange(10.0, 18.0))

    def test_category_is_a_closed_set(self) -> None:
        """По категории работает MMR на Этапе 7: синоним сломал бы его молча."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SceneVerdict(
                scene_index=0,
                start="00:00:00",
                end="00:00:08",
                score=50,
                category="comedy",  # type: ignore[arg-type]
                reason="x",
            )

    def test_all_categories_are_declared(self) -> None:
        assert "funny" in CATEGORIES and "plot_twist" in CATEGORIES

    def test_empty_answer_is_valid(self) -> None:
        """Модель имеет право не найти ничего интересного."""
        assert RankingResponse.model_validate_json('{"verdicts": []}').verdicts == []


# --------------------------------------------------------------- сводка
class TestSummarizer:
    def test_speech_is_clipped_in_the_middle(self) -> None:
        """Конец фразы часто и есть соль момента — панчлайн, признание,
        приказ. Обрезав хвост, мы выкинули бы именно его."""
        job, video = uuid4(), uuid4()
        text = "начало " + "х" * MAX_SPEECH_CHARS + " финал"
        scene = make_scene(0, job_id=job, video_id=video, text=text)

        excerpt = summarize(scene).transcript_excerpt

        assert excerpt.startswith("начало")
        assert excerpt.endswith("финал")
        assert len(excerpt) <= MAX_SPEECH_CHARS + 5

    def test_missing_features_are_not_mentioned(self) -> None:
        """«Лиц: нет данных» стоит токенов и не сообщает ничего."""
        job, video = uuid4(), uuid4()
        scene = make_scene(0, job_id=job, video_id=video, text="реплика")

        block = render(summarize(scene))

        assert "Видео:" not in block
        assert "реплика" in block

    def test_header_carries_index_and_timecodes(self) -> None:
        """Модель обязана вернуть их без изменений — чем меньше ей нужно
        преобразовывать, тем меньше поводов ошибиться."""
        job, video = uuid4(), uuid4()
        scene = make_scene(4, job_id=job, video_id=video, text="a")

        header = render(summarize(scene)).splitlines()[0]

        assert "Сцена 4" in header
        assert "00:00:40" in header and "00:00:48" in header

    def test_scene_with_speech_is_always_sent(self) -> None:
        job, video = uuid4(), uuid4()
        scene = make_scene(0, job_id=job, video_id=video, text="хоть что-то")

        assert is_worth_ranking(scene)

    def test_silent_and_flat_scene_is_filtered_out(self) -> None:
        """Экономия ровно здесь: платим за каждую отправленную сцену."""
        job, video = uuid4(), uuid4()
        scene = make_scene(0, job_id=job, video_id=video, features={"vision.motion.mean": 0.01})

        assert not is_worth_ranking(scene)

    def test_silent_but_loud_scene_is_sent(self) -> None:
        job, video = uuid4(), uuid4()
        scene = make_scene(0, job_id=job, video_id=video, features={"audio.event.laughter": 0.8})

        assert is_worth_ranking(scene)

    def test_video_context_has_no_cache_invalidators(self) -> None:
        """Ни job_id, ни даты: у Claude этот блок кешируется, и любое
        меняющееся значение обнулило бы кеш на каждом запросе."""
        block = format_video_context(
            VideoContext(
                title="film.mp4",
                duration_sec=120.0,
                language="ru",
                profile="default",
                scene_count=12,
            )
        )

        assert "job" not in block.lower()
        assert "2026" not in block


# --------------------------------------------------------------- батчи
class TestBatching:
    def test_order_is_preserved(self) -> None:
        """Соседние сцены дают модели контекст — перемешивание его убило бы."""
        summaries = [
            SceneSummary(index=i, range=TimeRange(i, i + 1), signals={}, transcript_excerpt="")
            for i in range(10)
        ]

        batches = split_batches(summaries, 4)

        assert [item.index for item in batches[0]] == [0, 1, 2, 3]
        assert [item.index for item in batches[-1]] == [8, 9]

    async def test_one_failed_batch_does_not_lose_the_others(self) -> None:
        """То же правило, что у анализаторов: частичный результат полезнее
        пустого."""
        summaries = [
            SceneSummary(
                index=i,
                range=TimeRange(i * 10.0, i * 10.0 + 8.0),
                signals={},
                transcript_excerpt="",
            )
            for i in range(4)
        ]

        class Flaky:
            def __init__(self) -> None:
                self.seen = 0

            async def rank_scenes(self, scenes, context):  # type: ignore[no-untyped-def]
                self.seen += 1
                if self.seen == 1:
                    raise ExternalServiceError("шлюз лёг", service="test")
                return [
                    LLMVerdict(
                        scene_index=scenes[0].index,
                        range=scenes[0].range,
                        score=50.0,
                        category="other",
                        reason="ok",
                    )
                ]

            async def generate_metadata(self, *a, **k):  # type: ignore[no-untyped-def]
                raise NotImplementedError

        outcome = await rank_in_batches(
            Flaky(),  # type: ignore[arg-type]
            summaries,
            _context(),
            batch_size=2,
        )

        assert outcome.batches == 2
        assert len(outcome.failed) == 1
        assert len(outcome.verdicts) == 1
        assert not outcome.all_failed

    async def test_all_batches_failed_is_visible(self) -> None:
        summaries = [
            SceneSummary(index=0, range=TimeRange(0.0, 8.0), signals={}, transcript_excerpt="")
        ]

        outcome = await rank_in_batches(
            FakeProvider(error=ExternalServiceError("нет сети", service="test")),  # type: ignore[arg-type]
            summaries,
            _context(),
            batch_size=10,
        )

        assert outcome.all_failed


def _context() -> VideoContext:
    return VideoContext(
        title=None, duration_sec=60.0, language="ru", profile="default", scene_count=4
    )


# ------------------------------------------------------------- сценарий
class TestRankScenes:
    def setup_method(self) -> None:
        self.job_id = uuid4()
        self.video_id = uuid4()
        self.scenes = [
            make_scene(index, job_id=self.job_id, video_id=self.video_id, text=f"реплика {index}")
            for index in range(3)
        ]
        scene_repo = FakeSceneRepository()
        scene_repo.scenes = list(self.scenes)
        self.uow = FakeUnitOfWork(scenes=scene_repo)
        self.uow.jobs.jobs[self.job_id] = Job(
            id=self.job_id,
            video_id=self.video_id,
            user_id=uuid4(),
            status=JobStatus.RUNNING,
        )
        self.uow.videos.videos[self.video_id] = Video(
            id=self.video_id,
            user_id=uuid4(),
            source_uri="file:///x.mp4",
            source_kind=SourceKind.LOCAL_FILE,
            status=VideoStatus.READY,
            storage_key="videos/x.mp4",
            checksum_sha256="a" * 64,
            info=MediaInfo(duration_sec=60.0, width=1920, height=1080, fps=25.0),
        )

    def build(self, ranker: FakeRanker, *, enabled: bool = True) -> RankScenes:
        return RankScenes(
            uow_factory=uow_factory(self.uow),
            ranker_factory=lambda: ranker,  # type: ignore[arg-type,return-value]
            enabled=enabled,
        )

    async def test_verdicts_land_on_scenes(self) -> None:
        outcome = RankingOutcome(
            verdicts=(
                LLMVerdict(
                    scene_index=1,
                    range=self.scenes[1].range,
                    score=88.0,
                    category="funny",
                    reason="смешно",
                ),
            ),
            scenes_sent=3,
            batches=1,
            cost={"requests": 1},
        )

        result = await self.build(FakeRanker(outcome)).execute(self.job_id)

        stored = {scene.index: scene for scene in self.uow.scenes.scenes}
        assert result.scenes_scored == 1
        assert stored[1].llm_score == 88.0
        assert stored[1].category == "funny"
        assert stored[0].llm_score is None, "неоценённая сцена должна остаться без оценки"

    async def test_cost_survives_the_stage_mark(self) -> None:
        """Дефект, найденный живым прогоном: отметка этапа пишет джоб,
        прочитанный в начале сценария, и затирала расход, записанный до неё."""
        outcome = RankingOutcome(scenes_sent=3, batches=1, cost={"requests": 2, "cost_usd": 0.0})

        await self.build(FakeRanker(outcome)).execute(self.job_id)

        assert self.uow.jobs.jobs[self.job_id].cost == {"llm": {"requests": 2, "cost_usd": 0.0}}

    async def test_unavailable_model_does_not_fail_the_job(self) -> None:
        """Главное свойство этапа: двухчасовая обработка не выбрасывается
        из-за пятисекундного сбоя чужого API."""
        ranker = FakeRanker(error=ExternalServiceError("503", service="openrouter"))

        result = await self.build(ranker).execute(self.job_id)

        job = self.uow.jobs.jobs[self.job_id]
        assert result.degraded
        assert job.status is JobStatus.PARTIAL
        assert job.stages[JobStage.LLM_RANK].status == "succeeded", (
            "этап должен закрыться, иначе прогресс джоба не доедет до конца"
        )

    async def test_disabled_llm_is_skipped_not_failed(self) -> None:
        result = await self.build(FakeRanker(), enabled=False).execute(self.job_id)

        assert result.skipped
        assert self.uow.jobs.jobs[self.job_id].stages[JobStage.LLM_RANK].status == "skipped"

    async def test_second_run_is_free(self) -> None:
        ranker = FakeRanker(RankingOutcome(scenes_sent=3, batches=1))
        usecase = self.build(ranker)

        await usecase.execute(self.job_id)
        second = await usecase.execute(self.job_id)

        assert second.reused
        assert ranker.calls == 1

    async def test_failed_batches_make_the_job_partial(self) -> None:
        outcome = RankingOutcome(scenes_sent=3, batches=2, failed=("batch 1: таймаут",))

        await self.build(FakeRanker(outcome)).execute(self.job_id)

        assert self.uow.jobs.jobs[self.job_id].status is JobStatus.PARTIAL

    async def test_no_scenes_is_skipped(self) -> None:
        self.uow.scenes.scenes = []

        result = await self.build(FakeRanker()).execute(self.job_id)

        assert result.skipped


# ------------------------------------------------- фильтр перед отправкой
class TestPrepare:
    def test_only_worthy_scenes_are_prepared(self) -> None:
        job, video = uuid4(), uuid4()
        scenes = [
            make_scene(0, job_id=job, video_id=video, text="есть речь"),
            make_scene(1, job_id=job, video_id=video, features={"vision.motion.mean": 0.001}),
            make_scene(2, job_id=job, video_id=video, features={"audio.event.scream": 0.9}),
        ]

        summaries = prepare(scenes)

        assert [item.index for item in summaries] == [0, 2]
