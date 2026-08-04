"""Сценарий ``DetectScenes`` целиком — на фейках, без единого видеофайла.

Именно ради этого сценарий не знает ни про ffmpeg, ни про PySceneDetect,
ни про SQLAlchemy: подставив фейковые порты, можно проверить всю логику
этапа — идемпотентность, отметки прогресса, поведение при сбое — за
миллисекунды вместо минут реального разбиения.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest
from tests.fakes import (
    FakeKeyframeExtractor,
    FakeSceneDetector,
    FakeSourceProvider,
    FakeStorage,
    FakeUnitOfWork,
    FakeVideoRepository,
    make_media_info,
    uow_factory,
)

from core.domain.entities import Job, Video
from core.domain.enums import (
    AssetKind,
    AssetOwnerType,
    JobStage,
    JobStatus,
    SourceKind,
    VideoStatus,
)
from core.domain.errors import NotFoundError, ProcessingError, ValidationError
from core.domain.media import LocalMedia
from core.domain.values import TimeRange
from core.usecases.detect_scenes import DetectScenes, SceneDetectionSettings

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

USER_ID = UUID("11111111-1111-1111-1111-111111111111")


@pytest.fixture
def media(tmp_path: Path) -> LocalMedia:
    path = tmp_path / "source.mp4"
    path.write_bytes(b"not-a-real-video")
    return LocalMedia(
        path=path,
        info=make_media_info(duration_sec=120.0),
        source_uri="upload://video",
        owned=False,
    )


@pytest.fixture
def video() -> Video:
    return Video(
        id=uuid4(),
        user_id=USER_ID,
        source_uri="upload://video",
        source_kind=SourceKind.UPLOAD,
        status=VideoStatus.READY,
        storage_key="videos/x/source.mp4",
        checksum_sha256="a" * 64,
        info=make_media_info(duration_sec=120.0),
    )


def make_job(video: Video) -> Job:
    return Job(
        id=uuid4(),
        video_id=video.id,
        user_id=video.user_id,
        status=JobStatus.QUEUED,
        params_hash="hash",
    )


def build(
    *,
    video: Video,
    job: Job,
    media: LocalMedia,
    tmp_path: Path,
    detector: FakeSceneDetector | None = None,
    keyframes: FakeKeyframeExtractor | None = None,
    settings: SceneDetectionSettings | None = None,
) -> tuple[DetectScenes, FakeUnitOfWork, FakeStorage, FakeSourceProvider]:
    uow = FakeUnitOfWork(videos=FakeVideoRepository({video.id: video}))
    uow.jobs.jobs[job.id] = job
    storage = FakeStorage()
    provider = FakeSourceProvider(media)

    usecase = DetectScenes(
        uow_factory=uow_factory(uow),
        detector=detector or FakeSceneDetector([TimeRange(0.0, 60.0), TimeRange(60.0, 120.0)]),
        keyframes=keyframes or FakeKeyframeExtractor(),
        storage=storage,
        provider_resolver=lambda uri: provider,
        workdir=tmp_path / "work",
        settings=settings or SceneDetectionSettings(),
    )
    return usecase, uow, storage, provider


class TestHappyPath:
    async def test_scenes_are_stored_with_keyframes(
        self, video: Video, media: LocalMedia, tmp_path: Path
    ) -> None:
        job = make_job(video)
        usecase, uow, storage, _ = build(video=video, job=job, media=media, tmp_path=tmp_path)

        result = await usecase.execute(job.id)

        assert result.scene_count == 2
        assert result.keyframe_count == 2
        assert not result.reused

        scenes = await uow.scenes.list_for_job(job.id)
        assert [scene.index for scene in scenes] == [0, 1]
        assert [scene.range for scene in scenes] == [TimeRange(0.0, 60.0), TimeRange(60.0, 120.0)]
        assert all(scene.video_id == video.id for scene in scenes)
        # Ключ детерминированный (док 04, §4.6): повторный прогон перезапишет
        # тот же объект, а не создаст второй.
        assert scenes[0].keyframe_key == f"jobs/{job.id}/keyframes/000000.jpg"
        assert set(storage.objects) == {
            f"jobs/{job.id}/keyframes/000000.jpg",
            f"jobs/{job.id}/keyframes/000001.jpg",
        }

    async def test_keyframe_is_taken_from_the_middle_of_the_scene(
        self, video: Video, media: LocalMedia, tmp_path: Path
    ) -> None:
        """Первый кадр после склейки часто ещё тёмный — как превью бесполезен."""
        job = make_job(video)
        keyframes = FakeKeyframeExtractor()
        usecase, *_ = build(
            video=video, job=job, media=media, tmp_path=tmp_path, keyframes=keyframes
        )

        await usecase.execute(job.id)

        assert keyframes.moments == [30.0, 90.0]

    async def test_assets_are_registered_for_the_job(
        self, video: Video, media: LocalMedia, tmp_path: Path
    ) -> None:
        job = make_job(video)
        usecase, uow, *_ = build(video=video, job=job, media=media, tmp_path=tmp_path)

        await usecase.execute(job.id)

        assets = await uow.assets.list_for_owner(AssetOwnerType.JOB, job.id)
        assert len(assets) == 2
        assert {asset["kind"] for asset in assets} == {AssetKind.KEYFRAME}
        assert all(asset["mime"] == "image/jpeg" for asset in assets)

    async def test_media_is_always_released(
        self, video: Video, media: LocalMedia, tmp_path: Path
    ) -> None:
        job = make_job(video)
        usecase, _, _, provider = build(video=video, job=job, media=media, tmp_path=tmp_path)

        await usecase.execute(job.id)

        assert (provider.acquired, provider.released) == (1, 1)

    async def test_temporary_frames_are_cleaned_up(
        self, video: Video, media: LocalMedia, tmp_path: Path
    ) -> None:
        """Кадры уехали в хранилище — копия на диске воркера не нужна.
        На тысяче сцен это сотни мегабайт, которые никто потом не уберёт."""
        job = make_job(video)
        usecase, *_ = build(video=video, job=job, media=media, tmp_path=tmp_path)

        await usecase.execute(job.id)

        assert not (tmp_path / "work" / "keyframes" / str(job.id)).exists()

    async def test_failed_keyframe_does_not_break_the_stage(
        self, video: Video, media: LocalMedia, tmp_path: Path
    ) -> None:
        """Превью — вспомогательный артефакт. Один битый кадр не повод
        терять разбиение, ради которого этап и затевался."""
        job = make_job(video)
        usecase, uow, _, _ = build(
            video=video,
            job=job,
            media=media,
            tmp_path=tmp_path,
            keyframes=FakeKeyframeExtractor(fail_indexes=frozenset({0})),
        )

        result = await usecase.execute(job.id)

        assert result.scene_count == 2
        assert result.keyframe_count == 1
        scenes = await uow.scenes.list_for_job(job.id)
        assert scenes[0].keyframe_key is None
        assert scenes[1].keyframe_key is not None

    async def test_keyframes_can_be_disabled(
        self, video: Video, media: LocalMedia, tmp_path: Path
    ) -> None:
        job = make_job(video)
        usecase, _, storage, _ = build(
            video=video,
            job=job,
            media=media,
            tmp_path=tmp_path,
            settings=SceneDetectionSettings(keyframes_enabled=False),
        )

        result = await usecase.execute(job.id)

        assert result.keyframe_count == 0
        assert storage.objects == {}


class TestStageState:
    async def test_stage_is_marked_succeeded_and_progress_advances(
        self, video: Video, media: LocalMedia, tmp_path: Path
    ) -> None:
        job = make_job(video)
        usecase, uow, *_ = build(video=video, job=job, media=media, tmp_path=tmp_path)

        await usecase.execute(job.id)

        stored = uow.jobs.jobs[job.id]
        state = stored.stages[JobStage.DETECT_SCENES]
        assert state.status == "succeeded"
        assert state.duration_sec is not None
        assert stored.status is JobStatus.RUNNING
        # Вес detect_scenes = 7 из 100 (док 04, §4.8).
        assert stored.progress == pytest.approx(10.0)

    async def test_failure_marks_job_failed_and_reraises(
        self, video: Video, media: LocalMedia, tmp_path: Path
    ) -> None:
        """Детекция сцен не деградирует, а валит джоб (док 04, §4.5):
        без сцен дальше нечего анализировать."""
        job = make_job(video)
        usecase, uow, _, provider = build(
            video=video,
            job=job,
            media=media,
            tmp_path=tmp_path,
            detector=FakeSceneDetector(error=ProcessingError("детектор упал")),
        )

        with pytest.raises(ProcessingError, match="детектор упал"):
            await usecase.execute(job.id)

        stored = uow.jobs.jobs[job.id]
        assert stored.status is JobStatus.FAILED
        assert stored.stages[JobStage.DETECT_SCENES].status == "failed"
        assert stored.error is not None
        assert stored.error["stage"] == JobStage.DETECT_SCENES.value
        assert stored.progress == pytest.approx(0.0)
        # Файл отпущен даже на исключении, иначе временные копии копятся.
        assert provider.released == 1

    async def test_second_run_does_not_recompute(
        self, video: Video, media: LocalMedia, tmp_path: Path
    ) -> None:
        """``acks_late`` означает, что убитый воркер вернёт задачу в очередь
        и она выполнится повторно при готовом результате первой."""
        job = make_job(video)
        detector = FakeSceneDetector([TimeRange(0.0, 60.0), TimeRange(60.0, 120.0)])
        usecase, uow, *_ = build(
            video=video, job=job, media=media, tmp_path=tmp_path, detector=detector
        )

        await usecase.execute(job.id)
        result = await usecase.execute(job.id)

        assert result.reused
        assert result.scene_count == 2
        assert len(detector.calls) == 1, "детектор не должен вызываться повторно"
        assert len(await uow.scenes.list_for_job(job.id)) == 2

    async def test_rerun_after_failure_replaces_scenes(
        self, video: Video, media: LocalMedia, tmp_path: Path
    ) -> None:
        """Повтор незавершённого этапа переписывает результат, а не дублирует:
        у таблицы уникальный индекс (job_id, index)."""
        job = make_job(video)
        detector = FakeSceneDetector([TimeRange(0.0, 60.0), TimeRange(60.0, 120.0)])
        usecase, uow, *_ = build(
            video=video, job=job, media=media, tmp_path=tmp_path, detector=detector
        )

        await usecase.execute(job.id)
        # Ручной сброс этапа — то же, что сделает POST /jobs/{id}/retry.
        uow.jobs.jobs[job.id] = uow.jobs.jobs[job.id].with_stage(JobStage.DETECT_SCENES, "pending")
        detector.scenes = [TimeRange(0.0, 120.0)]

        result = await usecase.execute(job.id)

        assert result.scene_count == 1
        scenes = await uow.scenes.list_for_job(job.id)
        assert [scene.range for scene in scenes] == [TimeRange(0.0, 120.0)]


class TestGuards:
    async def test_unknown_job(self, video: Video, media: LocalMedia, tmp_path: Path) -> None:
        job = make_job(video)
        usecase, *_ = build(video=video, job=job, media=media, tmp_path=tmp_path)

        with pytest.raises(NotFoundError):
            await usecase.execute(uuid4())

    async def test_deleted_video_is_rejected(
        self, video: Video, media: LocalMedia, tmp_path: Path
    ) -> None:
        deleted = Video(
            id=video.id,
            user_id=video.user_id,
            source_uri=video.source_uri,
            source_kind=video.source_kind,
            status=VideoStatus.DELETED,
            storage_key=video.storage_key,
            checksum_sha256=video.checksum_sha256,
            info=video.info,
        )
        job = make_job(deleted)
        usecase, *_ = build(video=deleted, job=job, media=media, tmp_path=tmp_path)

        with pytest.raises(ValidationError, match="недоступно"):
            await usecase.execute(job.id)

    async def test_detector_breaking_the_contract_is_caught(
        self, video: Video, media: LocalMedia, tmp_path: Path
    ) -> None:
        """Порт обещает покрытие окна целиком, но выполнять обещание будет
        чужой код. Дыра, не замеченная здесь, всплыла бы этапа через три —
        необъяснимым провалом в аналитике."""
        job = make_job(video)
        usecase, *_ = build(
            video=video,
            job=job,
            media=media,
            tmp_path=tmp_path,
            detector=FakeSceneDetector([TimeRange(0.0, 50.0), TimeRange(60.0, 120.0)]),
        )

        with pytest.raises(ProcessingError, match="Разрыв"):
            await usecase.execute(job.id)

    async def test_detector_returning_nothing_is_caught(
        self, video: Video, media: LocalMedia, tmp_path: Path
    ) -> None:
        job = make_job(video)
        usecase, *_ = build(
            video=video,
            job=job,
            media=media,
            tmp_path=tmp_path,
            detector=FakeSceneDetector([]),
        )

        with pytest.raises(ProcessingError, match="ни одной сцены"):
            await usecase.execute(job.id)
