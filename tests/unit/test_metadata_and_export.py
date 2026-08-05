"""Метаданные клипов, манифест, итоговый статус и Range-запросы.

Четыре разные вещи в одном файле, потому что все они — завершение работы
над джобом: подписать результат, описать его, назвать статусом и отдать
клиенту. Ни одна не требует ни ffmpeg, ни модели, ни базы.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from api.ranges import parse_range
from core.domain.analysis import AnnotationOutcome
from core.domain.entities import (
    Asset,
    Clip,
    ClipMetadata,
    Job,
    StageState,
    Video,
    compute_progress,
)
from core.domain.enums import (
    AssetKind,
    AssetOwnerType,
    ClipStatus,
    JobStage,
    JobStatus,
    SourceKind,
    VideoStatus,
)
from core.domain.errors import ExternalServiceError
from core.domain.media import MediaInfo, Transcript
from core.domain.values import TimeRange
from core.usecases.finalize import final_status
from core.usecases.metadata import GenerateMetadata, merge_cost
from export.manifest import MANIFEST_VERSION, build_manifest

pytestmark = pytest.mark.unit


def make_job(**stages: str) -> Job:
    job = Job(id=uuid4(), video_id=uuid4(), user_id=uuid4(), status=JobStatus.RUNNING)
    return replace(
        job,
        stages={
            JobStage(name): StageState(stage=JobStage(name), status=status)
            for name, status in stages.items()
        },
    )


def make_clip(
    rank: int = 1, *, status: ClipStatus = ClipStatus.READY, job_id: UUID | None = None
) -> Clip:
    return Clip(
        id=uuid4(),
        job_id=job_id or uuid4(),
        video_id=uuid4(),
        rank=rank,
        range=TimeRange(rank * 10.0, rank * 10.0 + 20.0),
        score=90.0 - rank,
        category="emotional",
        reason="повод",
        status=status,
    )


# ------------------------------------------------------------- Range
class TestRangeParsing:
    @pytest.mark.parametrize(
        ("header", "expected"),
        [
            ("bytes=0-99", (0, 99)),
            ("bytes=100-", (100, 999)),
            ("bytes=-200", (800, 999)),
            ("bytes=0-99999", (0, 999)),  # хвост за концом просто обрезается
            ("bytes=0-0", (0, 0)),  # один байт — тоже диапазон
        ],
    )
    def test_supported_forms(self, header: str, expected: tuple[int, int]) -> None:
        span = parse_range(header, 1000)
        assert span is not None
        assert (span.start, span.end) == expected

    def test_length_counts_both_ends(self) -> None:
        """Границы в HTTP включительные — расхождение со срезами Python
        стоит одного лишнего байта и битого видео у клиента."""
        span = parse_range("bytes=0-0", 1000)
        assert span is not None
        assert span.length == 1

    @pytest.mark.parametrize(
        "header",
        [None, "", "items=0-10", "bytes=abc", "bytes=-", "bytes=500-100"],
    )
    def test_unclear_request_means_whole_file(self, header: str | None) -> None:
        """Непонятный заголовок — повод отдать файл целиком, а не ошибку:
        Range необязателен, и ломать клиента на нём незачем."""
        assert parse_range(header, 1000) is None

    def test_start_beyond_the_end_is_an_error(self) -> None:
        """Единственный случай, где по RFC полагается 416."""
        with pytest.raises(ValueError, match="за концом файла"):
            parse_range("bytes=2000-", 1000)

    def test_content_range_header(self) -> None:
        span = parse_range("bytes=10-19", 1000)
        assert span is not None
        assert span.content_range(1000) == "bytes 10-19/1000"


# ---------------------------------------------------------- манифест
class TestManifest:
    def _video(self) -> Video:
        return Video(
            id=uuid4(),
            user_id=uuid4(),
            source_uri="file:///media/in/x.mp4",
            source_kind=SourceKind.LOCAL_FILE,
            status=VideoStatus.READY,
            storage_key="videos/x/source.mp4",
            checksum_sha256="0" * 64,
            info=MediaInfo(duration_sec=154.2, width=640, height=360),
        )

    def test_contains_every_clip_with_its_files(self) -> None:
        clip = make_clip()
        asset = Asset(
            id=uuid4(),
            owner_type=AssetOwnerType.CLIP,
            owner_id=clip.id,
            kind=AssetKind.CLIP_V,
            storage_key="jobs/x/clips/001_v.mp4",
            mime="video/mp4",
            size_bytes=1234,
        )

        payload = build_manifest(make_job(), self._video(), [clip], {clip.id: [asset]})

        assert payload["manifest_version"] == MANIFEST_VERSION
        entry = payload["clips"][0]
        assert entry["rank"] == 1
        assert entry["assets"]["clip_v"]["storage_key"] == "jobs/x/clips/001_v.mp4"
        assert entry["assets"]["clip_v"]["download_url"].endswith(str(asset.id))

    def test_metadata_lands_in_the_manifest(self) -> None:
        clip = replace(
            make_clip(),
            metadata=ClipMetadata(
                title="Заголовок", description="Описание", hashtags=("#тег",), hook="Хук"
            ),
        )
        payload = build_manifest(make_job(), self._video(), [clip], {})
        entry = payload["clips"][0]
        assert entry["title"] == "Заголовок"
        assert entry["hashtags"] == ["#тег"]

    def test_time_is_given_both_ways(self) -> None:
        """Человеку тайм-код, коду секунды — как и в API."""
        payload = build_manifest(make_job(), self._video(), [make_clip()], {})
        entry = payload["clips"][0]
        assert entry["start"] == "00:00:10.000"
        assert entry["start_sec"] == 10.0

    def test_is_json_serialisable(self) -> None:
        """Манифест уезжает файлом — в нём не должно остаться доменных типов."""
        import json

        payload = build_manifest(make_job(), self._video(), [make_clip()], {})
        assert json.loads(json.dumps(payload, ensure_ascii=False))


# ------------------------------------------------------- итоговый статус
class TestFinalStatus:
    def test_everything_worked(self) -> None:
        job = make_job(detect_scenes="succeeded", render_clips="succeeded")
        assert final_status(job, [make_clip()]) is JobStatus.SUCCEEDED

    def test_no_ready_clips_is_a_failure(self) -> None:
        """Пустая папка — не успех, как бы ни отработали этапы."""
        job = make_job(detect_scenes="succeeded")
        assert final_status(job, [make_clip(status=ClipStatus.FAILED)]) is JobStatus.FAILED
        assert final_status(job, []) is JobStatus.FAILED

    def test_skipped_stage_makes_it_partial(self) -> None:
        job = make_job(detect_scenes="succeeded", llm_rank="skipped")
        assert final_status(job, [make_clip()]) is JobStatus.PARTIAL

    def test_broken_clip_makes_it_partial(self) -> None:
        job = make_job(render_clips="succeeded")
        clips = [make_clip(1), make_clip(2, status=ClipStatus.FAILED)]
        assert final_status(job, clips) is JobStatus.PARTIAL


# --------------------------------------------------- сценарий метаданных
class FakeAnnotator:
    def __init__(self, outcome: AnnotationOutcome | None = None, error: Exception | None = None):
        self.outcome = outcome or AnnotationOutcome()
        self.error = error
        self.seen: list[int] = []

    async def annotate(
        self,
        clips: Sequence[Clip],
        transcript: Transcript,
        language: str | None = None,
    ) -> AnnotationOutcome:
        if self.error is not None:
            raise self.error
        self.seen = [clip.rank for clip in clips]
        return self.outcome


class FakeJobs:
    def __init__(self, job: Job) -> None:
        self.job = job

    async def get(self, job_id: UUID) -> Job:
        return self.job

    async def update(self, job: Job) -> Job:
        self.job = job
        return job

    async def write_stage(
        self,
        job_id: UUID,
        stage: JobStage,
        state: StageState,
        *,
        status: JobStatus,
        error: dict[str, object] | None = None,
    ) -> Job | None:
        """То же, что в настоящем репозитории: меняем только своё поле."""
        job = self.job if getattr(self, "job", None) is not None else None
        if job is None or job.id != job_id:
            return None
        states = dict(job.stages)
        states[stage] = state
        self.job = replace(
            job,
            stage=stage,
            stages=states,
            progress=compute_progress(states),
            status=status,
            error=error if error is not None else job.error,
        )
        return self.job


class FakeVideos:
    def __init__(self, video: Video) -> None:
        self.video = video

    async def get(self, video_id: UUID) -> Video:
        return self.video


class FakeClips:
    def __init__(self, clips: list[Clip]) -> None:
        self.clips = {clip.id: clip for clip in clips}

    async def list_for_job(self, job_id: UUID) -> Sequence[Clip]:
        return sorted(self.clips.values(), key=lambda clip: clip.rank)

    async def update(self, clip: Clip) -> Clip:
        self.clips[clip.id] = clip
        return clip


class FakeTranscripts:
    async def get_for_job(self, job_id: UUID) -> Transcript:
        return Transcript(language="ru")


class FakeUow:
    def __init__(self, jobs: FakeJobs, videos: FakeVideos, clips: FakeClips) -> None:
        self.jobs = jobs
        self.videos = videos
        self.clips = clips
        self.transcripts = FakeTranscripts()

    async def __aenter__(self) -> FakeUow:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def commit(self) -> None:
        return None


@pytest.fixture
def metadata_scenario() -> dict[str, object]:
    video = Video(
        id=uuid4(),
        user_id=uuid4(),
        source_uri="file:///media/in/x.mp4",
        source_kind=SourceKind.LOCAL_FILE,
        status=VideoStatus.READY,
        storage_key="videos/x/source.mp4",
        checksum_sha256="0" * 64,
    )
    job = Job(id=uuid4(), video_id=video.id, user_id=video.user_id, status=JobStatus.RUNNING)
    clips = [make_clip(rank, job_id=job.id) for rank in (1, 2)]
    uow = FakeUow(FakeJobs(job), FakeVideos(video), FakeClips(clips))
    return {"job_id": job.id, "uow": uow, "clips": clips}


def metadata_for(clip: Clip) -> ClipMetadata:
    return ClipMetadata(
        title=f"Заголовок {clip.rank}",
        description="Описание",
        hashtags=("#один", "#два", "#три"),
        hook="Хук",
    )


class TestGenerateMetadata:
    def _build(self, scenario: dict[str, object], annotator: FakeAnnotator) -> GenerateMetadata:
        return GenerateMetadata(
            uow_factory=lambda: scenario["uow"],  # type: ignore[arg-type,return-value]
            annotator_factory=lambda: annotator,  # type: ignore[arg-type,return-value]
        )

    @pytest.mark.asyncio
    async def test_metadata_lands_on_clips(self, metadata_scenario: dict[str, object]) -> None:
        clips: list[Clip] = metadata_scenario["clips"]  # type: ignore[assignment]
        annotator = FakeAnnotator(
            AnnotationOutcome(
                metadata={clip.id: metadata_for(clip) for clip in clips},
                requested=len(clips),
                cost={"requests": 2, "cost_usd": 0.0},
            )
        )

        result = await self._build(metadata_scenario, annotator).execute(
            metadata_scenario["job_id"]  # type: ignore[arg-type]
        )

        assert result.annotated == 2
        uow: FakeUow = metadata_scenario["uow"]  # type: ignore[assignment]
        assert all(clip.metadata is not None for clip in uow.clips.clips.values())

    @pytest.mark.asyncio
    async def test_failed_clips_are_not_annotated(
        self, metadata_scenario: dict[str, object]
    ) -> None:
        """Заголовок к несобравшемуся файлу — строка, которую некуда показать."""
        uow: FakeUow = metadata_scenario["uow"]  # type: ignore[assignment]
        broken = next(iter(uow.clips.clips.values()))
        uow.clips.clips[broken.id] = replace(broken, status=ClipStatus.FAILED)
        annotator = FakeAnnotator()

        await self._build(metadata_scenario, annotator).execute(
            metadata_scenario["job_id"]  # type: ignore[arg-type]
        )

        assert broken.rank not in annotator.seen

    @pytest.mark.asyncio
    async def test_unavailable_model_degrades_instead_of_failing(
        self, metadata_scenario: dict[str, object]
    ) -> None:
        """Не хватает подписей к готовым файлам — это не повод валить джоб."""
        annotator = FakeAnnotator(error=ExternalServiceError("нет связи", service="openrouter"))

        result = await self._build(metadata_scenario, annotator).execute(
            metadata_scenario["job_id"]  # type: ignore[arg-type]
        )

        assert result.degraded
        uow: FakeUow = metadata_scenario["uow"]  # type: ignore[assignment]
        assert uow.jobs.job.stages[JobStage.GENERATE_METADATA].status == "succeeded"

    @pytest.mark.asyncio
    async def test_cost_is_added_to_what_the_job_already_spent(
        self, metadata_scenario: dict[str, object]
    ) -> None:
        """Регресс живого дефекта: расход двух LLM-этапов должен лечь
        под один ключ суммой. Первый прогон записал его плоско рядом
        с вложенным от ранжирования, и в поле оказалось два набора
        чисел одновременно."""
        uow: FakeUow = metadata_scenario["uow"]  # type: ignore[assignment]
        uow.jobs.job = replace(uow.jobs.job, cost={"llm": {"requests": 5, "prompt_tokens": 100}})
        annotator = FakeAnnotator(
            AnnotationOutcome(requested=2, cost={"requests": 2, "prompt_tokens": 30})
        )

        await self._build(metadata_scenario, annotator).execute(
            metadata_scenario["job_id"]  # type: ignore[arg-type]
        )

        assert uow.jobs.job.cost == {"llm": {"requests": 7, "prompt_tokens": 130}}

    @pytest.mark.asyncio
    async def test_disabled_llm_skips_the_stage(self, metadata_scenario: dict[str, object]) -> None:
        usecase = GenerateMetadata(
            uow_factory=lambda: metadata_scenario["uow"],  # type: ignore[arg-type,return-value]
            annotator_factory=lambda: FakeAnnotator(),  # type: ignore[arg-type,return-value]
            enabled=False,
        )

        result = await usecase.execute(metadata_scenario["job_id"])  # type: ignore[arg-type]

        assert result.skipped
        uow: FakeUow = metadata_scenario["uow"]  # type: ignore[assignment]
        assert uow.jobs.job.stages[JobStage.GENERATE_METADATA].status == "skipped"


class TestCostMerging:
    def test_numbers_add_up(self) -> None:
        merged = merge_cost({"llm": {"requests": 5}}, {"requests": 2})
        assert merged["llm"] == {"requests": 7}

    def test_first_write_creates_the_key(self) -> None:
        assert merge_cost({}, {"requests": 2}) == {"llm": {"requests": 2}}

    def test_lists_are_concatenated(self) -> None:
        """Список неудач — не число, складывать его надо иначе."""
        merged = merge_cost({"llm": {"failed": ["a"]}}, {"failed": ["b"]})
        assert merged["llm"] == {"failed": ["a", "b"]}

    def test_other_keys_survive(self) -> None:
        merged = merge_cost({"cpu_sec": 42, "llm": {"requests": 1}}, {"requests": 1})
        assert merged["cpu_sec"] == 42


def test_manifest_path_is_deterministic() -> None:
    """Повторная финализация переписывает файл, а не плодит копии."""
    from core.domain.keys import manifest_key

    job_id = uuid4()
    assert manifest_key(job_id) == manifest_key(job_id)
    assert str(job_id) in manifest_key(job_id)
    assert Path(manifest_key(job_id)).suffix == ".json"
