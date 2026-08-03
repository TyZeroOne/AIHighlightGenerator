"""Транскрипт против живой БД.

Фейковый репозиторий хранит доменные объекты как есть и потому не может
поймать главное: переживает ли пословная разметка круг «домен → JSONB →
домен», не ломается ли повторная запись, доезжает ли текст до колонки сцены.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from tests.fakes import (
    FakeAudioExtractor,
    FakeSourceProvider,
    FakeTranscriber,
    make_media_info,
    make_transcript,
)

from core.container import Container
from core.domain.entities import Job, Scene, Video
from core.domain.enums import (
    AssetKind,
    AssetOwnerType,
    JobStage,
    JobStatus,
    SourceKind,
    VideoStatus,
)
from core.domain.keys import audio_key
from core.domain.media import LocalMedia
from core.domain.values import TimeRange
from core.usecases.transcribe import ExtractAudio, Transcribe

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.fixture
async def job(container: Container) -> Job:
    async with container.uow() as uow:
        user = await uow.users.create(email=f"{uuid4()}@test.local", api_key_hash=str(uuid4()))
        video = await uow.videos.add(
            Video(
                id=uuid4(),
                user_id=user.id,
                source_uri="upload://x",
                source_kind=SourceKind.UPLOAD,
                status=VideoStatus.READY,
                storage_key="videos/x/source.mp4",
                checksum_sha256="e" * 64,
                info=make_media_info(duration_sec=120.0),
            )
        )
        stored = await uow.jobs.add(
            Job(
                id=uuid4(),
                video_id=video.id,
                user_id=user.id,
                status=JobStatus.RUNNING,
                params_hash=str(uuid4()),
            )
        )
        await uow.scenes.bulk_add(
            [
                Scene(
                    id=uuid4(),
                    video_id=video.id,
                    job_id=stored.id,
                    index=index,
                    range=TimeRange(index * 60.0, (index + 1) * 60.0),
                )
                for index in range(2)
            ]
        )
        await uow.commit()
    return stored


def build_transcribe(
    container: Container, tmp_path: Path, transcriber: FakeTranscriber
) -> Transcribe:
    return Transcribe(
        uow_factory=container.uow,
        transcriber=transcriber,
        storage=container.storage,
        workdir=tmp_path / "work",
    )


async def put_audio(container: Container, job: Job, tmp_path: Path) -> None:
    source = tmp_path / "audio.wav"
    source.write_bytes(b"RIFF" + b"\x00" * 3200)
    await container.storage.put(audio_key(job.id), source)


class TestPersistence:
    async def test_words_survive_the_round_trip(
        self, container: Container, job: Job, tmp_path: Path
    ) -> None:
        """Пословные тайминги лежат в JSONB под короткими ключами.
        Если чтение их не восстановит, Этап 9 останется без караоке."""
        await put_audio(container, job, tmp_path)
        transcriber = FakeTranscriber(make_transcript((5.0, 8.0, "держись они прямо за нами")))
        usecase = build_transcribe(container, tmp_path, transcriber)

        await usecase.execute(job.id)

        async with container.uow() as uow:
            stored = await uow.transcripts.get_for_job(job.id)

        segment = stored.segments[0]
        assert [word.text for word in segment.words] == ["держись", "они", "прямо", "за", "нами"]
        assert segment.words[0].start == pytest.approx(5.0, abs=0.01)
        assert segment.words[-1].end == pytest.approx(8.0, abs=0.01)
        assert segment.language == "ru"
        assert segment.no_speech_prob == pytest.approx(0.05)

    async def test_scene_text_is_written(
        self, container: Container, job: Job, tmp_path: Path
    ) -> None:
        await put_audio(container, job, tmp_path)
        usecase = build_transcribe(
            container,
            tmp_path,
            FakeTranscriber(make_transcript((10.0, 14.0, "первая"), (70.0, 74.0, "вторая"))),
        )

        result = await usecase.execute(job.id)

        assert result.scenes_with_text == 2
        async with container.uow() as uow:
            scenes = await uow.scenes.list_for_job(job.id)
        assert [scene.transcript_text for scene in scenes] == ["первая", "вторая"]

    async def test_rerun_replaces_instead_of_appending(
        self, container: Container, job: Job, tmp_path: Path
    ) -> None:
        await put_audio(container, job, tmp_path)
        transcriber = FakeTranscriber(make_transcript((10.0, 14.0, "первый прогон")))
        usecase = build_transcribe(container, tmp_path, transcriber)
        await usecase.execute(job.id)

        async with container.uow() as uow:
            stored = await uow.jobs.get(job.id)
            assert stored is not None
            await uow.jobs.update(stored.with_stage(JobStage.TRANSCRIBE, "pending"))
            await uow.commit()

        transcriber.transcript = make_transcript((10.0, 14.0, "второй прогон"))
        await usecase.execute(job.id)

        async with container.uow() as uow:
            count = await uow.transcripts.count_for_job(job.id)
            final = await uow.transcripts.get_for_job(job.id)
        assert count == 1
        assert final.segments[0].text == "второй прогон"

    async def test_stage_progress_is_persisted(
        self, container: Container, job: Job, tmp_path: Path
    ) -> None:
        await put_audio(container, job, tmp_path)
        usecase = build_transcribe(
            container, tmp_path, FakeTranscriber(make_transcript((10.0, 14.0, "раз")))
        )

        await usecase.execute(job.id)

        async with container.uow() as uow:
            stored = await uow.jobs.get(job.id)
        assert stored is not None
        assert stored.stages[JobStage.TRANSCRIBE].status == "succeeded"
        # Вес transcribe — 25 из 100 (док 04, §4.8).
        assert stored.progress == pytest.approx(25.0)


class TestAudioStage:
    async def test_audio_asset_has_ttl_and_metadata(
        self, container: Container, job: Job, tmp_path: Path
    ) -> None:
        """Час несжатого аудио — 115 МБ. Артефакт обязан иметь срок жизни,
        иначе диск закончится раньше, чем кто-нибудь это заметит."""
        source = tmp_path / "source.mp4"
        source.write_bytes(b"video")
        media = LocalMedia(
            path=source,
            info=make_media_info(duration_sec=120.0),
            source_uri="upload://x",
            owned=False,
        )
        usecase = ExtractAudio(
            uow_factory=container.uow,
            extractor=FakeAudioExtractor(duration_sec=120.0),
            storage=container.storage,
            provider_resolver=lambda uri: FakeSourceProvider(media),
            retention_hours=24,
        )

        result = await usecase.execute(job.id)

        assert result.storage_key == audio_key(job.id)
        assert await container.storage.exists(audio_key(job.id))
        async with container.uow() as uow:
            assets = [
                (asset.kind, asset.expires_at is not None)
                for asset in await uow.assets.list_for_owner(AssetOwnerType.JOB, job.id)
            ]
        assert assets == [(AssetKind.AUDIO_WAV, True)]

    async def test_transcribe_without_audio_is_skipped(
        self, container: Container, job: Job, tmp_path: Path
    ) -> None:
        """Аудио в хранилище нет — значит, предыдущий этап его пропустил.
        Пустой транскрипт и `skipped`, а не падение."""
        transcriber = FakeTranscriber(make_transcript((10.0, 14.0, "не должно случиться")))
        usecase = build_transcribe(container, tmp_path, transcriber)

        result = await usecase.execute(job.id)

        assert result.skipped
        assert not transcriber.calls
        async with container.uow() as uow:
            stored = await uow.jobs.get(job.id)
            count = await uow.transcripts.count_for_job(job.id)
        assert stored is not None
        assert stored.stages[JobStage.TRANSCRIBE].status == "skipped"
        assert count == 0
