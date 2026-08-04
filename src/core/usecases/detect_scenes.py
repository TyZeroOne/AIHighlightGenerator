"""Сценарий разбиения видео на сцены — первый настоящий этап конвейера.

Что здесь происходит по шагам:

1. читаем джоб и видео, проверяем, что этап ещё не пройден (идемпотентность);
2. получаем файл на диск через ``SourceProvider`` — сценарий не знает, откуда
   он берётся: локальный том, наше хранилище или (в будущем) стрим;
3. зовём ``SceneDetector`` — сценарий не знает, PySceneDetect это или ffmpeg;
4. вытаскиваем превью-кадр на сцену и кладём в хранилище;
5. пишем сцены и артефакты в БД одной транзакцией.

Ни одного упоминания ffmpeg, SQLAlchemy или S3 — только порты. Именно поэтому
весь сценарий целиком проверяется юнит-тестами на фейках, без видеофайлов.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

from core.domain.entities import Job, Scene, Video
from core.domain.enums import AssetKind, AssetOwnerType, JobStage
from core.domain.errors import ProcessingError
from core.domain.keys import keyframe_key
from core.domain.media import LocalMedia
from core.domain.values import EPSILON, MediaWindow, TimeRange
from core.logging import get_logger
from core.ports.media import KeyframeExtractor, SceneDetector
from core.ports.repository import UnitOfWorkFactory
from core.ports.source import SourceProvider
from core.ports.storage import ObjectStorage
from core.usecases.stages import StageTracker

__all__ = ["DetectScenes", "SceneDetectionResult", "SceneDetectionSettings"]

log = get_logger(__name__)

STAGE = JobStage.DETECT_SCENES

#: Как резолвится провайдер под конкретный URI. Функция, а не реестр:
#: сценарий не должен знать ни про классы провайдеров, ни про то, что их
#: вообще несколько. Связывание — работа композиционного корня.
type ProviderResolver = Callable[[str], SourceProvider]


@dataclass(frozen=True, slots=True)
class SceneDetectionSettings:
    keyframes_enabled: bool = True
    keyframe_mime: str = "image/jpeg"
    #: Сколько часов хранить превью сцен. Неделя: их показывает API
    #: в разборе скоринга, а после того как джоб отсмотрен, они мусор.
    retention_hours: int | None = 168


@dataclass(frozen=True, slots=True)
class SceneDetectionResult:
    job_id: UUID
    scene_count: int
    keyframe_count: int
    duration_sec: float
    #: ``True`` — этап уже был пройден в этом джобе, работа не повторялась.
    reused: bool = False


class DetectScenes:
    """Этап ``detect_scenes``.

    Этот этап — из тех трёх, что не деградируют, а валят джоб (док 04, §4.5):
    без сцен дальше нечего анализировать, нечего скорить и не из чего резать.
    Поэтому исключение здесь не глотается, а помечает джоб провалившимся
    и летит наверх — Celery решит, ретраить его или нет, по типу ошибки.
    """

    def __init__(
        self,
        *,
        uow_factory: UnitOfWorkFactory,
        detector: SceneDetector,
        keyframes: KeyframeExtractor,
        storage: ObjectStorage,
        provider_resolver: ProviderResolver,
        workdir: Path,
        settings: SceneDetectionSettings | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._detector = detector
        self._keyframes = keyframes
        self._storage = storage
        self._resolve_provider = provider_resolver
        self._workdir = workdir
        self._settings = settings or SceneDetectionSettings()
        self._tracker = StageTracker(uow_factory=uow_factory, stage=STAGE)

    async def execute(self, job_id: UUID) -> SceneDetectionResult:
        started = time.monotonic()
        job, video = await self._tracker.load(job_id)

        if job.is_stage_done(STAGE):
            # Задача могла вернуться в очередь после убийства воркера
            # (acks_late), и повторное выполнение — штатная ситуация.
            # Пересчитывать сцены заново значило бы тратить минуты CPU
            # на результат, который уже лежит в базе.
            async with self._uow_factory() as uow:
                existing = await uow.scenes.count_for_job(job_id)
            log.info("scenes.stage_already_done", job_id=str(job_id), scenes=existing)
            return SceneDetectionResult(
                job_id=job_id,
                scene_count=existing,
                keyframe_count=0,
                duration_sec=0.0,
                reused=True,
            )

        await self._tracker.begin(job)

        try:
            result = await self._run(job, video, started)
        except Exception as error:
            await self._tracker.fail(job, error, duration_sec=round(time.monotonic() - started, 3))
            raise

        await self._tracker.succeed(job, duration_sec=result.duration_sec)
        log.info(
            "scenes.stored",
            job_id=str(job_id),
            scenes=result.scene_count,
            keyframes=result.keyframe_count,
            duration_sec=result.duration_sec,
        )
        return result

    # --------------------------------------------------------------- шаги
    async def _run(self, job: Job, video: Video, started: float) -> SceneDetectionResult:
        provider = self._resolve_provider(video.source_uri)
        media = await provider.acquire(video.source_uri)
        try:
            window = MediaWindow(
                source_id=video.source_uri,
                range=media.full_range,
                index=0,
                is_final=True,
            )
            ranges = await self._detector.detect(media, window)
            _validate_coverage(ranges, window.range)

            keyframes = await self._extract_keyframes(job.id, media, ranges)
        finally:
            # Провайдер сам решает, удалять файл или нет: исходник
            # пользователя трогать нельзя, скачанную копию — обязательно.
            await provider.release(media)

        scenes = [
            Scene(
                id=uuid4(),
                video_id=video.id,
                job_id=job.id,
                index=index,
                range=span,
                keyframe_key=keyframes.get(index),
            )
            for index, span in enumerate(ranges)
        ]

        async with self._uow_factory() as uow:
            # Сначала удаляем прежние сцены: у таблицы уникальный индекс
            # (job_id, index), и повторный прогон этапа без очистки упёрся бы
            # в конфликт вместо того, чтобы переписать результат.
            await uow.scenes.delete_for_job(job.id)
            await uow.scenes.bulk_add(scenes)
            for index, key in sorted(keyframes.items()):
                await uow.assets.add(
                    owner_type=AssetOwnerType.JOB,
                    owner_id=job.id,
                    kind=AssetKind.KEYFRAME,
                    storage_key=key,
                    mime=self._settings.keyframe_mime,
                    meta={"scene_index": index, "at_sec": round(ranges[index].midpoint, 3)},
                    ttl_hours=self._settings.retention_hours,
                )
            await uow.commit()

        return SceneDetectionResult(
            job_id=job.id,
            scene_count=len(scenes),
            keyframe_count=len(keyframes),
            duration_sec=round(time.monotonic() - started, 3),
        )

    async def _extract_keyframes(
        self,
        job_id: UUID,
        media: LocalMedia,
        ranges: Sequence[TimeRange],
    ) -> dict[int, str]:
        """Кадр на сцену → хранилище. Возвращает ключи по индексам сцен."""
        if not self._settings.keyframes_enabled or not ranges:
            return {}

        # Середина сцены, а не начало: первый кадр после склейки часто ещё
        # тёмный (затухание предыдущего плана) и как превью бесполезен.
        moments = [span.midpoint for span in ranges]
        target = self._workdir / "keyframes" / str(job_id)
        frames = await self._keyframes.extract(media, moments, target)

        keys: dict[int, str] = {}
        try:
            for index, frame in enumerate(frames):
                if frame is None:
                    continue
                key = keyframe_key(job_id, index)
                await self._storage.put(key, frame, content_type=self._settings.keyframe_mime)
                keys[index] = key
        finally:
            await asyncio.to_thread(_cleanup, target)

        return keys


def _validate_coverage(ranges: Sequence[TimeRange], window: TimeRange) -> None:
    """Проверка контракта порта на границе с внешней реализацией.

    Контракт «сцены покрывают окно целиком, без дыр и пересечений» записан
    в ``SceneDetector``, но выполнять его будет чужой код — сегодня наш,
    завтра сторонний плагин. Дешёвая проверка здесь превращает нарушение
    в понятную ошибку этапа вместо необъяснимых дыр в аналитике через
    три этапа конвейера.
    """
    if not ranges:
        raise ProcessingError("Детектор не вернул ни одной сцены", window=str(window))

    if abs(ranges[0].start - window.start) > 1e-3 or abs(ranges[-1].end - window.end) > 1e-3:
        raise ProcessingError(
            "Сцены не покрывают окно целиком",
            window=str(window),
            first=str(ranges[0]),
            last=str(ranges[-1]),
        )

    for previous, current in itertools.pairwise(ranges):
        if abs(current.start - previous.end) > EPSILON:
            raise ProcessingError(
                "Разрыв или пересечение между сценами",
                previous=str(previous),
                current=str(current),
            )


def _cleanup(directory: Path) -> None:
    """Убрать временные кадры. Файлы уже в хранилище, копия на диске воркера
    не нужна — а на тысяче сцен это сотни мегабайт."""
    if not directory.exists():
        return
    for item in directory.iterdir():
        item.unlink(missing_ok=True)
    directory.rmdir()
