"""Сценарии речевого этапа на фейках: ни ffmpeg, ни модели, ни базы.

Проверяется то, что должно работать одинаково при любой реализации портов:
идемпотентность, отметки этапов, пропуск видео без звука, привязка текста
к сценам и то, что временные файлы за собой убираются.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest
from tests.fakes import (
    FakeAudioExtractor,
    FakeSourceProvider,
    FakeStorage,
    FakeTranscriber,
    FakeUnitOfWork,
    FakeVideoRepository,
    make_media_info,
    make_transcript,
    uow_factory,
)

from core.domain.entities import Job, Scene, Video
from core.domain.enums import (
    AssetKind,
    AssetOwnerType,
    JobStage,
    JobStatus,
    SourceKind,
    VideoStatus,
)
from core.domain.errors import FFmpegError, ProcessingError
from core.domain.keys import audio_key
from core.domain.media import LocalMedia, Transcript
from core.domain.values import TimeRange
from core.usecases.transcribe import ExtractAudio, Transcribe

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

USER_ID = UUID("22222222-2222-2222-2222-222222222222")


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


def make_video(*, with_audio: bool = True) -> Video:
    return Video(
        id=uuid4(),
        user_id=USER_ID,
        source_uri="upload://video",
        source_kind=SourceKind.UPLOAD,
        status=VideoStatus.READY,
        storage_key="videos/x/source.mp4",
        checksum_sha256="c" * 64,
        info=make_media_info(duration_sec=120.0, audio_codec="aac" if with_audio else None),
    )


def make_job(video: Video) -> Job:
    return Job(
        id=uuid4(),
        video_id=video.id,
        user_id=video.user_id,
        status=JobStatus.QUEUED,
        params_hash="hash",
    )


def make_scenes(video: Video, job: Job) -> list[Scene]:
    return [
        Scene(id=uuid4(), video_id=video.id, job_id=job.id, index=0, range=TimeRange(0.0, 60.0)),
        Scene(id=uuid4(), video_id=video.id, job_id=job.id, index=1, range=TimeRange(60.0, 120.0)),
    ]


def build_uow(video: Video, job: Job, scenes: list[Scene] | None = None) -> FakeUnitOfWork:
    uow = FakeUnitOfWork(videos=FakeVideoRepository({video.id: video}))
    uow.jobs.jobs[job.id] = job
    if scenes:
        uow.scenes.scenes.extend(scenes)
    return uow


class TestExtractAudio:
    async def test_audio_lands_in_storage_with_an_asset(
        self, media: LocalMedia, tmp_path: Path
    ) -> None:
        video = make_video()
        job = make_job(video)
        uow = build_uow(video, job)
        storage = FakeStorage()

        usecase = ExtractAudio(
            uow_factory=uow_factory(uow),
            extractor=FakeAudioExtractor(),
            storage=storage,
            provider_resolver=lambda uri: FakeSourceProvider(media),
        )

        result = await usecase.execute(job.id)

        assert result.storage_key == audio_key(job.id)
        assert not result.skipped
        assert audio_key(job.id) in storage.objects
        assets = await uow.assets.list_for_owner(AssetOwnerType.JOB, job.id)
        assert [asset["kind"] for asset in assets] == [AssetKind.AUDIO_WAV]
        assert uow.jobs.jobs[job.id].stages[JobStage.EXTRACT_AUDIO].status == "succeeded"

    async def test_temporary_wav_is_deleted(self, media: LocalMedia, tmp_path: Path) -> None:
        """Час аудио — это 115 МБ. Копия на диске воркера после выгрузки
        в хранилище не нужна никому."""
        video = make_video()
        job = make_job(video)
        uow = build_uow(video, job)
        provider = FakeSourceProvider(media)

        usecase = ExtractAudio(
            uow_factory=uow_factory(uow),
            extractor=FakeAudioExtractor(),
            storage=FakeStorage(),
            provider_resolver=lambda uri: provider,
        )

        await usecase.execute(job.id)

        assert not media.path.with_suffix(".wav").exists()
        assert provider.released == 1

    async def test_video_without_audio_is_skipped_not_failed(
        self, media: LocalMedia, tmp_path: Path
    ) -> None:
        """Видео без звука — штатная ситуация: конвейер идёт дальше
        по одному видеоряду (док 04, §4.5)."""
        video = make_video(with_audio=False)
        job = make_job(video)
        uow = build_uow(video, job)
        extractor = FakeAudioExtractor()

        usecase = ExtractAudio(
            uow_factory=uow_factory(uow),
            extractor=extractor,
            storage=FakeStorage(),
            provider_resolver=lambda uri: FakeSourceProvider(media),
        )

        result = await usecase.execute(job.id)

        assert result.skipped and result.storage_key is None
        assert not extractor.calls, "ffmpeg не должен запускаться ради известного ответа"
        stored = uow.jobs.jobs[job.id]
        assert stored.stages[JobStage.EXTRACT_AUDIO].status == "skipped"
        assert stored.status is JobStatus.RUNNING
        # Пропущенный этап засчитывается в прогресс целиком: иначе джоб
        # застрял бы навсегда из-за работы, которой не будет.
        assert stored.progress == pytest.approx(5.0)

    async def test_missing_track_discovered_by_ffmpeg_is_also_skip(
        self, media: LocalMedia, tmp_path: Path
    ) -> None:
        """ffprobe мог не заметить дорожку, а ffmpeg — не найти её.
        Это по-прежнему «звука нет», а не «этап сломался»."""
        video = make_video(with_audio=True)
        job = make_job(video)
        uow = build_uow(video, job)

        usecase = ExtractAudio(
            uow_factory=uow_factory(uow),
            extractor=FakeAudioExtractor(error=FFmpegError("В медиафайле нет аудиодорожки")),
            storage=FakeStorage(),
            provider_resolver=lambda uri: FakeSourceProvider(media),
        )

        result = await usecase.execute(job.id)

        assert result.skipped
        assert uow.jobs.jobs[job.id].stages[JobStage.EXTRACT_AUDIO].status == "skipped"

    async def test_real_failure_marks_job_failed(self, media: LocalMedia, tmp_path: Path) -> None:
        video = make_video()
        job = make_job(video)
        uow = build_uow(video, job)

        usecase = ExtractAudio(
            uow_factory=uow_factory(uow),
            extractor=FakeAudioExtractor(error=FFmpegError("кодек не поддерживается")),
            storage=FakeStorage(),
            provider_resolver=lambda uri: FakeSourceProvider(media),
        )

        with pytest.raises(FFmpegError):
            await usecase.execute(job.id)

        stored = uow.jobs.jobs[job.id]
        assert stored.status is JobStatus.FAILED
        assert stored.stages[JobStage.EXTRACT_AUDIO].status == "failed"

    async def test_second_run_does_not_re_extract(self, media: LocalMedia, tmp_path: Path) -> None:
        video = make_video()
        job = make_job(video)
        uow = build_uow(video, job)
        extractor = FakeAudioExtractor()
        usecase = ExtractAudio(
            uow_factory=uow_factory(uow),
            extractor=extractor,
            storage=FakeStorage(),
            provider_resolver=lambda uri: FakeSourceProvider(media),
        )

        await usecase.execute(job.id)
        result = await usecase.execute(job.id)

        assert result.reused
        assert len(extractor.calls) == 1


class TestTranscribe:
    def build(
        self,
        *,
        video: Video,
        job: Job,
        scenes: list[Scene] | None,
        tmp_path: Path,
        transcriber: FakeTranscriber,
        with_audio: bool = True,
    ) -> tuple[Transcribe, FakeUnitOfWork]:
        uow = build_uow(video, job, scenes)
        storage = FakeStorage()
        if with_audio:
            source = tmp_path / "in.wav"
            source.write_bytes(b"RIFF" + b"\x00" * 3200)
            storage.objects[audio_key(job.id)] = source.read_bytes()

        usecase = Transcribe(
            uow_factory=uow_factory(uow),
            transcriber=transcriber,
            storage=storage,
            workdir=tmp_path / "work",
        )
        return usecase, uow

    async def test_segments_are_stored_and_bound_to_scenes(self, tmp_path: Path) -> None:
        video = make_video()
        job = make_job(video)
        transcriber = FakeTranscriber(
            make_transcript(
                (5.0, 8.0, "первая реплика"),
                (70.0, 74.0, "вторая реплика"),
            )
        )
        usecase, uow = self.build(
            video=video,
            job=job,
            scenes=make_scenes(video, job),
            tmp_path=tmp_path,
            transcriber=transcriber,
        )

        result = await usecase.execute(job.id)

        assert result.segment_count == 2
        assert result.language == "ru"
        assert result.scenes_with_text == 2
        stored = await uow.transcripts.get_for_job(job.id)
        assert [segment.text for segment in stored.segments] == [
            "первая реплика",
            "вторая реплика",
        ]
        by_index = {scene.index: scene for scene in await uow.scenes.list_for_job(job.id)}
        assert by_index[0].transcript_text == "первая реплика"
        assert by_index[1].transcript_text == "вторая реплика"

    async def test_reply_crossing_a_cut_lands_in_both_scenes(self, tmp_path: Path) -> None:
        """Фраза, начатая до склейки и законченная после, принадлежит обеим
        сценам: обрезать её по границе — значит потерять смысл в каждой."""
        video = make_video()
        job = make_job(video)
        usecase, uow = self.build(
            video=video,
            job=job,
            scenes=make_scenes(video, job),
            tmp_path=tmp_path,
            transcriber=FakeTranscriber(make_transcript((58.0, 63.0, "через границу"))),
        )

        await usecase.execute(job.id)

        texts = [scene.transcript_text for scene in await uow.scenes.list_for_job(job.id)]
        assert texts == ["через границу", "через границу"]

    async def test_scene_without_speech_gets_none(self, tmp_path: Path) -> None:
        """None, а не пустая строка: «здесь не говорили» и «здесь сказали
        ничего» — разные вещи, и в SQL они тоже разные."""
        video = make_video()
        job = make_job(video)
        usecase, uow = self.build(
            video=video,
            job=job,
            scenes=make_scenes(video, job),
            tmp_path=tmp_path,
            transcriber=FakeTranscriber(make_transcript((5.0, 8.0, "только в первой"))),
        )

        await usecase.execute(job.id)

        by_index = {scene.index: scene for scene in await uow.scenes.list_for_job(job.id)}
        assert by_index[1].transcript_text is None

    async def test_progress_is_reported(self, tmp_path: Path) -> None:
        """Этап весит 25 из 100 и идёт минутами: без дробного прогресса
        снаружи он выглядит как зависание."""
        video = make_video()
        job = make_job(video)
        transcriber = FakeTranscriber(make_transcript((1.0, 2.0, "раз")))
        usecase, _ = self.build(
            video=video, job=job, scenes=None, tmp_path=tmp_path, transcriber=transcriber
        )

        await usecase.execute(job.id)

        assert transcriber.progress_reports == [0.5, 1.0]

    async def test_downloaded_audio_is_deleted(self, tmp_path: Path) -> None:
        video = make_video()
        job = make_job(video)
        usecase, _ = self.build(
            video=video,
            job=job,
            scenes=None,
            tmp_path=tmp_path,
            transcriber=FakeTranscriber(make_transcript((1.0, 2.0, "раз"))),
        )

        await usecase.execute(job.id)

        assert not (tmp_path / "work" / "audio" / f"{job.id}.wav").exists()

    async def test_no_audio_gives_empty_transcript_not_an_error(self, tmp_path: Path) -> None:
        """DoD этапа: видео без аудио → пустой транскрипт, конвейер не падает."""
        video = make_video(with_audio=False)
        job = make_job(video)
        transcriber = FakeTranscriber(make_transcript((1.0, 2.0, "не должно случиться")))
        usecase, uow = self.build(
            video=video,
            job=job,
            scenes=make_scenes(video, job),
            tmp_path=tmp_path,
            transcriber=transcriber,
            with_audio=False,
        )

        result = await usecase.execute(job.id)

        assert result.skipped
        assert result.segment_count == 0
        assert not transcriber.calls, "модель не должна вызываться без аудио"
        assert (await uow.transcripts.get_for_job(job.id)).is_empty
        stored = uow.jobs.jobs[job.id]
        assert stored.stages[JobStage.TRANSCRIBE].status == "skipped"
        assert stored.status is JobStatus.RUNNING

    async def test_model_failure_marks_job_failed(self, tmp_path: Path) -> None:
        video = make_video()
        job = make_job(video)
        usecase, uow = self.build(
            video=video,
            job=job,
            scenes=None,
            tmp_path=tmp_path,
            transcriber=FakeTranscriber(error=ProcessingError("модель упала")),
        )

        with pytest.raises(ProcessingError):
            await usecase.execute(job.id)

        stored = uow.jobs.jobs[job.id]
        assert stored.status is JobStatus.FAILED
        assert stored.stages[JobStage.TRANSCRIBE].status == "failed"

    async def test_second_run_reuses_the_result(self, tmp_path: Path) -> None:
        video = make_video()
        job = make_job(video)
        transcriber = FakeTranscriber(make_transcript((1.0, 2.0, "раз")))
        usecase, _ = self.build(
            video=video, job=job, scenes=None, tmp_path=tmp_path, transcriber=transcriber
        )

        await usecase.execute(job.id)
        result = await usecase.execute(job.id)

        assert result.reused
        assert result.segment_count == 1
        assert len(transcriber.calls) == 1

    async def test_rerun_replaces_previous_transcript(self, tmp_path: Path) -> None:
        """Повтор незавершённого этапа переписывает транскрипт, а не дописывает:
        иначе каждое слово получило бы по второму экземпляру."""
        video = make_video()
        job = make_job(video)
        transcriber = FakeTranscriber(make_transcript((1.0, 2.0, "первый прогон")))
        usecase, uow = self.build(
            video=video, job=job, scenes=None, tmp_path=tmp_path, transcriber=transcriber
        )

        await usecase.execute(job.id)
        uow.jobs.jobs[job.id] = uow.jobs.jobs[job.id].with_stage(JobStage.TRANSCRIBE, "pending")
        transcriber.transcript = make_transcript((1.0, 2.0, "второй прогон"))

        result = await usecase.execute(job.id)

        assert result.segment_count == 1
        stored = await uow.transcripts.get_for_job(job.id)
        assert [segment.text for segment in stored.segments] == ["второй прогон"]

    async def test_empty_transcript_from_model_is_not_an_error(self, tmp_path: Path) -> None:
        """Музыка без речи: модель честно вернула пустоту, и это результат."""
        video = make_video()
        job = make_job(video)
        usecase, uow = self.build(
            video=video,
            job=job,
            scenes=make_scenes(video, job),
            tmp_path=tmp_path,
            transcriber=FakeTranscriber(Transcript()),
        )

        result = await usecase.execute(job.id)

        assert result.segment_count == 0
        assert result.scenes_with_text == 0
        assert uow.jobs.jobs[job.id].stages[JobStage.TRANSCRIBE].status == "succeeded"
