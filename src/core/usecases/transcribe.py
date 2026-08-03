"""Сценарии речевого этапа: извлечение аудио и распознавание.

Два отдельных этапа, а не один, и это не дробление ради дробления:

* извлечение аудио — это ffmpeg на CPU, минуты на длинном файле;
* распознавание — это модель на GPU, где очередь короткая и дорогая.

Разделив их, мы даём конвейеру запустить распознавание сразу, как только
готово аудио, и параллельно считать по тому же аудио громкость и паузы
(Этап 5). Слитый этап держал бы GPU-слот всё время работы ffmpeg.

Оба сценария знают только порты: ``AudioExtractor``, ``Transcriber``,
``ObjectStorage``, репозитории. Ни faster-whisper, ни ffmpeg здесь не
упоминаются — их подставляет композиционный корень.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import UUID

from core.domain.entities import Job, Video
from core.domain.enums import AssetKind, AssetOwnerType, JobStage
from core.domain.errors import FFmpegError, ObjectNotFoundError
from core.domain.keys import audio_key
from core.domain.media import AudioTrack, Transcript
from core.domain.values import MediaWindow
from core.logging import get_logger
from core.ports.media import AudioExtractor
from core.ports.repository import UnitOfWorkFactory
from core.ports.source import SourceProvider
from core.ports.speech import ProgressCallback, Transcriber
from core.ports.storage import ObjectStorage
from core.usecases.stages import StageTracker

__all__ = [
    "AudioExtractionResult",
    "ExtractAudio",
    "Transcribe",
    "TranscriptionResult",
]

log = get_logger(__name__)

type ProviderResolver = Callable[[str], SourceProvider]


@dataclass(frozen=True, slots=True)
class AudioExtractionResult:
    job_id: UUID
    storage_key: str | None
    duration_sec: float
    #: Видео без звука — не ошибка: конвейер идёт дальше по видеопризнакам.
    skipped: bool = False
    reused: bool = False


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    job_id: UUID
    segment_count: int
    word_count: int
    language: str | None
    scenes_with_text: int
    duration_sec: float
    skipped: bool = False
    reused: bool = False


class ExtractAudio:
    """Этап ``extract_audio``: дорожка видео → WAV 16 кГц в хранилище."""

    def __init__(
        self,
        *,
        uow_factory: UnitOfWorkFactory,
        extractor: AudioExtractor,
        storage: ObjectStorage,
        provider_resolver: ProviderResolver,
        retention_hours: int | None = 24,
    ) -> None:
        self._uow_factory = uow_factory
        self._extractor = extractor
        self._storage = storage
        self._resolve_provider = provider_resolver
        self._retention_hours = retention_hours
        self._tracker = StageTracker(uow_factory=uow_factory, stage=JobStage.EXTRACT_AUDIO)

    async def execute(self, job_id: UUID) -> AudioExtractionResult:
        started = time.monotonic()
        job, video = await self._tracker.load(job_id)

        if job.is_stage_done(JobStage.EXTRACT_AUDIO):
            key = audio_key(job_id)
            exists = await self._storage.exists(key)
            return AudioExtractionResult(
                job_id=job_id,
                storage_key=key if exists else None,
                duration_sec=0.0,
                skipped=not exists,
                reused=True,
            )

        # Видео без звука — штатная ситуация (док 04, §4.5). Проверяем ДО
        # отметки «этап начался»: запускать ffmpeg, чтобы он честно сообщил
        # об отсутствии дорожки, — это минута работы ради известного ответа.
        if video.info is not None and not video.info.has_audio:
            await self._tracker.skip(job, "в видео нет аудиодорожки")
            return AudioExtractionResult(
                job_id=job_id, storage_key=None, duration_sec=0.0, skipped=True
            )

        await self._tracker.begin(job)
        try:
            result = await self._run(job, video, started)
        except Exception as error:
            await self._tracker.fail(job, error, duration_sec=round(time.monotonic() - started, 3))
            raise

        # Отсутствие дорожки могло обнаружиться уже внутри: тогда этап
        # отмечен пропущенным, и переписывать эту отметку успехом нельзя —
        # иначе следующий этап решит, что аудио где-то есть.
        if not result.skipped:
            await self._tracker.succeed(job, duration_sec=result.duration_sec)
        return result

    async def _run(self, job: Job, video: Video, started: float) -> AudioExtractionResult:
        provider = self._resolve_provider(video.source_uri)
        media = await provider.acquire(video.source_uri)
        track: AudioTrack | None = None
        try:
            window = MediaWindow(source_id=video.source_uri, range=media.full_range)
            try:
                track = await self._extractor.extract(media, window)
            except FFmpegError as error:
                # ffprobe мог не заметить дорожку, а ffmpeg — не найти её.
                # Это по-прежнему «звука нет», а не «этап сломался».
                if "нет аудиодорожки" in str(error):
                    await self._tracker.skip(job, "аудиодорожка не найдена при извлечении")
                    return AudioExtractionResult(
                        job_id=job.id, storage_key=None, duration_sec=0.0, skipped=True
                    )
                raise

            key = audio_key(job.id)
            await self._storage.put(key, track.path, content_type="audio/wav")
            size = track.path.stat().st_size

            async with self._uow_factory() as uow:
                await uow.assets.add(
                    owner_type=AssetOwnerType.JOB,
                    owner_id=job.id,
                    kind=AssetKind.AUDIO_WAV,
                    storage_key=key,
                    mime="audio/wav",
                    size_bytes=size,
                    meta={
                        "sample_rate": track.sample_rate,
                        "channels": track.channels,
                        "duration_sec": round(track.duration_sec, 3),
                    },
                    # Час несжатого аудио — это 115 МБ. Держать его вечно
                    # незачем: после анализа он не нужен никому.
                    ttl_hours=self._retention_hours,
                )
                await uow.commit()
        finally:
            await provider.release(media)
            if track is not None:
                # Копия на диске воркера больше не нужна: файл в хранилище,
                # а распознавание может пойти вообще на другой машине.
                track.path.unlink(missing_ok=True)

        log.info("audio.stored", job_id=str(job.id), key=key, size_bytes=size)
        return AudioExtractionResult(
            job_id=job.id,
            storage_key=key,
            duration_sec=round(time.monotonic() - started, 3),
        )


class Transcribe:
    """Этап ``transcribe``: аудио → сегменты речи в БД и текст в сценах."""

    def __init__(
        self,
        *,
        uow_factory: UnitOfWorkFactory,
        transcriber: Transcriber,
        storage: ObjectStorage,
        workdir: Path,
        language: str | None = None,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        self._uow_factory = uow_factory
        self._transcriber = transcriber
        self._storage = storage
        self._workdir = workdir
        self._language = language or None
        self._sample_rate = sample_rate
        self._channels = channels
        self._tracker = StageTracker(uow_factory=uow_factory, stage=JobStage.TRANSCRIBE)

    async def execute(self, job_id: UUID) -> TranscriptionResult:
        started = time.monotonic()
        job, video = await self._tracker.load(job_id)

        if job.is_stage_done(JobStage.TRANSCRIBE):
            async with self._uow_factory() as uow:
                existing = await uow.transcripts.count_for_job(job_id)
            return TranscriptionResult(
                job_id=job_id,
                segment_count=existing,
                word_count=0,
                language=None,
                scenes_with_text=0,
                duration_sec=0.0,
                reused=True,
            )

        key = audio_key(job_id)
        if not await self._storage.exists(key):
            # Аудио нет — значит, предыдущий этап его пропустил (видео без
            # звука). Пустой транскрипт здесь правильнее исключения: конвейер
            # обязан дойти до конца и по одному видеоряду.
            await self._tracker.skip(job, "аудио для распознавания отсутствует")
            await self._store(job, video, Transcript())
            return TranscriptionResult(
                job_id=job_id,
                segment_count=0,
                word_count=0,
                language=None,
                scenes_with_text=0,
                duration_sec=0.0,
                skipped=True,
            )

        await self._tracker.begin(job)
        try:
            result = await self._run(job, video, key, started)
        except Exception as error:
            await self._tracker.fail(job, error, duration_sec=round(time.monotonic() - started, 3))
            raise

        await self._tracker.succeed(job, duration_sec=result.duration_sec)
        return result

    async def _run(self, job: Job, video: Video, key: str, started: float) -> TranscriptionResult:
        destination = self._workdir / "audio" / f"{job.id}.wav"
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            await self._storage.get(key, destination)
        except ObjectNotFoundError:
            raise
        size = destination.stat().st_size

        track = AudioTrack(
            path=destination,
            sample_rate=self._sample_rate,
            channels=self._channels,
            duration_sec=_pcm_duration(size, self._sample_rate, self._channels),
        )

        try:
            transcript = await self._transcriber.transcribe(
                track,
                language=self._language,
                progress=_progress_bridge(lambda fraction: self._tracker.progress(job, fraction)),
            )
        finally:
            destination.unlink(missing_ok=True)

        scenes_with_text = await self._store(job, video, transcript)

        words = sum(len(segment.words) for segment in transcript.segments)
        log.info(
            "transcript.stored",
            job_id=str(job.id),
            segments=len(transcript.segments),
            words=words,
            language=transcript.language,
            scenes_with_text=scenes_with_text,
        )
        return TranscriptionResult(
            job_id=job.id,
            segment_count=len(transcript.segments),
            word_count=words,
            language=transcript.language,
            scenes_with_text=scenes_with_text,
            duration_sec=round(time.monotonic() - started, 3),
        )

    async def _store(self, job: Job, video: Video, transcript: Transcript) -> int:
        """Сегменты в ``transcript_segments`` + текст в ``scenes``.

        Денормализация текста в сцену — сознательная (док 03): LLM на Этапе 6
        получает сцену целиком, и джойн к транскрипту на каждую из тысячи сцен
        обойдётся дороже, чем дублирование текста.
        """
        async with self._uow_factory() as uow:
            await uow.transcripts.replace_for_job(job.id, transcript, video.id)

            scenes = await uow.scenes.list_for_job(job.id)
            updated = [
                replace(scene, transcript_text=transcript.text_within(scene.range) or None)
                for scene in scenes
            ]
            if updated:
                await uow.scenes.bulk_update(updated)
            await uow.commit()

        return sum(1 for scene in updated if scene.transcript_text)


def _progress_bridge(report: Callable[[float], Coroutine[Any, Any, None]]) -> ProgressCallback:
    """Мост из потока модели в event loop.

    Модель считает в отдельном потоке и сообщает прогресс оттуда. Трогать
    сессию БД из чужого потока нельзя, поэтому отчёт планируется в тот цикл,
    которому сессия принадлежит. Ошибку отчёта глотаем: не сообщить прогресс
    неприятно, уронить из-за этого двадцатиминутную транскрипцию — хуже.
    """
    loop = asyncio.get_running_loop()

    def callback(fraction: float) -> None:
        try:
            asyncio.run_coroutine_threadsafe(report(fraction), loop)
        except RuntimeError as error:  # pragma: no cover — цикл уже остановлен
            log.debug("transcribe.progress_dropped", error=str(error))

    return callback


def _pcm_duration(size_bytes: int, sample_rate: int = 16000, channels: int = 1) -> float:
    payload = max(size_bytes - 44, 0)
    bytes_per_second = sample_rate * channels * 2
    return payload / bytes_per_second if bytes_per_second else 0.0
