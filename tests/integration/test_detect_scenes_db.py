"""Этап детекции сцен против живой БД.

Юнит-тесты сценария работают на фейковых репозиториях и проверяют логику.
Здесь проверяется то, чего фейки увидеть не могут: что сцены реально ложатся
в Postgres, что JSONB со состояниями этапов переживает круг «домен → БД →
домен», что уникальный индекс ``(job_id, index)`` не мешает повторному
прогону, а прогресс джоба виден снаружи транзакции этапа.

Детектор и извлечение кадров здесь фейковые: настоящие проверяются на
настоящем видео в ``test_scene_detection_real.py``. Смешивать одно с другим —
значит получить тест, который падает по двум несвязанным причинам.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from tests.fakes import (
    FakeKeyframeExtractor,
    FakeSceneDetector,
    FakeSourceProvider,
    make_media_info,
)

from core.container import Container
from core.domain.entities import Job, Video
from core.domain.enums import (
    AssetKind,
    AssetOwnerType,
    JobStage,
    JobStatus,
    SourceKind,
    VideoStatus,
)
from core.domain.media import LocalMedia
from core.domain.values import TimeRange
from core.usecases.detect_scenes import DetectScenes, SceneDetectionSettings

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

SCENES = [TimeRange(0.0, 40.0), TimeRange(40.0, 80.0), TimeRange(80.0, 120.0)]


@pytest.fixture
async def video(container: Container) -> Video:
    async with container.uow() as uow:
        user = await uow.users.create(email=f"{uuid4()}@test.local", api_key_hash=str(uuid4()))
        stored = await uow.videos.add(
            Video(
                id=uuid4(),
                user_id=user.id,
                source_uri="upload://x",
                source_kind=SourceKind.UPLOAD,
                status=VideoStatus.READY,
                storage_key="videos/x/source.mp4",
                checksum_sha256="b" * 64,
                info=make_media_info(duration_sec=120.0),
            )
        )
        await uow.commit()
    return stored


@pytest.fixture
async def job(container: Container, video: Video) -> Job:
    async with container.uow() as uow:
        stored = await uow.jobs.add(
            Job(
                id=uuid4(),
                video_id=video.id,
                user_id=video.user_id,
                status=JobStatus.QUEUED,
                params={"profile": "default"},
                params_hash="deadbeef",
            )
        )
        await uow.commit()
    return stored


def build(
    container: Container,
    video: Video,
    tmp_path: Path,
    *,
    detector: FakeSceneDetector | None = None,
) -> DetectScenes:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fake")
    media = LocalMedia(
        path=source,
        info=make_media_info(duration_sec=120.0),
        source_uri=video.source_uri,
        owned=False,
    )
    provider = FakeSourceProvider(media)

    return DetectScenes(
        uow_factory=container.uow,
        detector=detector or FakeSceneDetector(SCENES),
        keyframes=FakeKeyframeExtractor(),
        storage=container.storage,
        provider_resolver=lambda uri: provider,
        workdir=tmp_path / "work",
        settings=SceneDetectionSettings(),
    )


class TestPersistence:
    async def test_scenes_and_assets_land_in_the_database(
        self, container: Container, video: Video, job: Job, tmp_path: Path
    ) -> None:
        usecase = build(container, video, tmp_path)

        result = await usecase.execute(job.id)

        assert result.scene_count == 3
        async with container.uow() as uow:
            scenes = await uow.scenes.list_for_job(job.id)
            records = await uow.assets.list_for_owner(AssetOwnerType.JOB, job.id)

        # Поля читаются УЖЕ ПОСЛЕ закрытия сессии, и это регресс дефекта,
        # найденного живым прогоном Этапа 8: репозиторий отдавал наружу
        # строки ORM, а те после коммита протухают и падают
        # `DetachedInstanceError` при обращении к любому полю.
        assets = [(asset.kind, asset.storage_key) for asset in records]

        assert [scene.index for scene in scenes] == [0, 1, 2]
        assert [scene.range for scene in scenes] == SCENES
        assert all(scene.video_id == video.id for scene in scenes)
        assert len(assets) == 3
        assert {kind for kind, _ in assets} == {AssetKind.KEYFRAME}
        assert {key for _, key in assets} == {
            f"jobs/{job.id}/keyframes/{index:06d}.jpg" for index in range(3)
        }

    async def test_keyframes_are_written_to_storage(
        self, container: Container, video: Video, job: Job, tmp_path: Path
    ) -> None:
        usecase = build(container, video, tmp_path)

        await usecase.execute(job.id)

        for index in range(3):
            assert await container.storage.exists(f"jobs/{job.id}/keyframes/{index:06d}.jpg")

    async def test_stage_state_survives_the_round_trip(
        self, container: Container, video: Video, job: Job, tmp_path: Path
    ) -> None:
        """JSONB со состояниями этапов — единственное место, где прогресс
        джоба переживает перезапуск процесса. Если он не читается обратно,
        конвейер теряет способность продолжить с середины."""
        usecase = build(container, video, tmp_path)

        await usecase.execute(job.id)

        async with container.uow() as uow:
            stored = await uow.jobs.get(job.id)

        assert stored is not None
        state = stored.stages[JobStage.DETECT_SCENES]
        assert state.status == "succeeded"
        assert state.duration_sec is not None and state.duration_sec >= 0
        assert stored.stage is JobStage.DETECT_SCENES
        assert stored.progress == pytest.approx(7.0)  # вес этапа из STAGE_WEIGHTS
        assert stored.params == {"profile": "default"}

    async def test_rerun_replaces_scenes_without_violating_unique_index(
        self, container: Container, video: Video, job: Job, tmp_path: Path
    ) -> None:
        """Повторный прогон незавершённого этапа обязан переписать сцены.
        Без предварительной очистки он упёрся бы в uq_scenes_job_index."""
        detector = FakeSceneDetector(SCENES)
        usecase = build(container, video, tmp_path, detector=detector)
        await usecase.execute(job.id)

        async with container.uow() as uow:
            stored = await uow.jobs.get(job.id)
            assert stored is not None
            await uow.jobs.update(stored.with_stage(JobStage.DETECT_SCENES, "pending"))
            await uow.commit()

        detector.scenes = [TimeRange(0.0, 60.0), TimeRange(60.0, 120.0)]
        result = await usecase.execute(job.id)

        assert result.scene_count == 2
        async with container.uow() as uow:
            scenes = await uow.scenes.list_for_job(job.id)
        assert [scene.range for scene in scenes] == detector.scenes

    async def test_second_run_of_a_finished_stage_is_free(
        self, container: Container, video: Video, job: Job, tmp_path: Path
    ) -> None:
        detector = FakeSceneDetector(SCENES)
        usecase = build(container, video, tmp_path, detector=detector)

        await usecase.execute(job.id)
        result = await usecase.execute(job.id)

        assert result.reused
        assert result.scene_count == 3
        assert len(detector.calls) == 1

    async def test_failed_stage_is_recorded(
        self, container: Container, video: Video, job: Job, tmp_path: Path
    ) -> None:
        from core.domain.errors import FFmpegError

        usecase = build(
            container, video, tmp_path, detector=FakeSceneDetector(error=FFmpegError("scdet упал"))
        )

        with pytest.raises(FFmpegError):
            await usecase.execute(job.id)

        async with container.uow() as uow:
            stored = await uow.jobs.get(job.id)
            count = await uow.scenes.count_for_job(job.id)

        assert stored is not None
        assert stored.status is JobStatus.FAILED
        assert stored.error is not None and stored.error["type"] == "FFmpegError"
        assert count == 0


class TestSceneRepository:
    async def test_bulk_add_count_and_delete(
        self, container: Container, video: Video, job: Job, tmp_path: Path
    ) -> None:
        usecase = build(container, video, tmp_path)
        await usecase.execute(job.id)

        async with container.uow() as uow:
            assert await uow.scenes.count_for_job(job.id) == 3
            await uow.scenes.delete_for_job(job.id)
            await uow.commit()

        async with container.uow() as uow:
            assert await uow.scenes.count_for_job(job.id) == 0

    async def test_top_by_score_puts_unscored_last(
        self, container: Container, video: Video, job: Job, tmp_path: Path
    ) -> None:
        """Скоринг появится на Этапе 7, но порядок сортировки важен уже
        сейчас: NULL-скор не должен вытеснять посчитанные сцены наверх."""
        from dataclasses import replace

        usecase = build(container, video, tmp_path)
        await usecase.execute(job.id)

        async with container.uow() as uow:
            scenes = list(await uow.scenes.list_for_job(job.id))
            await uow.scenes.bulk_update(
                [replace(scenes[1], final_score=91.0), replace(scenes[2], final_score=42.0)]
            )
            await uow.commit()

        async with container.uow() as uow:
            top = await uow.scenes.top_by_score(job.id, limit=3)

        assert [scene.index for scene in top] == [1, 2, 0]
