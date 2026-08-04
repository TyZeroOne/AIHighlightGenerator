"""Этап ``render_clips``: отобранные отрезки → файлы.

Первый этап конвейера, который производит не решения, а результат.
До него всё было обратимо: пересчитать признаки, переоценить сцены,
переотобрать моменты — вопрос процессорного времени. Здесь появляются
файлы, которые пользователь скачает.

## Что происходит с каждым клипом

1. горизонтальный файл — точная резка из исходника;
2. вертикальный 1080×1920 — по траектории кадрирования (если включено);
3. превью-кадр;
4. всё это в хранилище, ссылки — в ``assets``, статус — в ``clips``.

## Три уровня деградации, а не один

Этап входит в тройку тех, что валят джоб (док 04, §4.5), — но «валят»
здесь означает «если не получилось **ничего**».

- **Не собрался вертикальный кадр или превью** — клип всё равно готов.
  Горизонтальный файл есть, и отдавать пользователю ошибку вместо него
  было бы странно.
- **Не собрался один клип** — остальные готовы. Клип помечается
  ``failed`` с причиной, этап продолжается. Девять файлов из десяти —
  это девять файлов, а не провал.
- **Не собрался ни один** — вот это провал этапа: дальше по конвейеру
  (субтитры, экспорт) работать не с чем.

## Почему параллельность ограничена, а не «по числу ядер»

Один x264 занимает все доступные ядра сам. Десять параллельных
процессов не ускорят работу — они поделят те же ядра, умножат
потребление памяти и заставят диск читать исходник из десяти мест
одновременно. Два-три процесса дают выигрыш на перемотке и разборе
контейнера, дальше начинается конкуренция. Значение настраивается
(``RENDER_CONCURRENCY``), замер — в разборе Этапа 8.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from core.domain.entities import Clip, Job, Video
from core.domain.enums import AssetKind, AssetOwnerType, JobStage
from core.domain.errors import ProcessingError
from core.domain.keys import clip_key, thumbnail_key
from core.domain.media import LocalMedia
from core.domain.render import VERTICAL_HEIGHT, VERTICAL_WIDTH, RenderedClip, RenderSpec
from core.logging import get_logger
from core.ports.media import ClipRenderer, CropPlanner
from core.ports.repository import UnitOfWorkFactory
from core.ports.source import SourceProvider
from core.ports.storage import ObjectStorage
from core.usecases.stages import StageTracker

__all__ = ["RenderClips", "RenderResult", "RenderSettings", "thumbnail_moment"]

log = get_logger(__name__)

STAGE = JobStage.RENDER_CLIPS

type ProviderResolver = Callable[[str], SourceProvider]

#: Доля клипа, с которой берётся превью. Не начало: первые кадры часто
#: заняты остатком предыдущей склейки или затемнением. Не середина —
#: там на диалоговых клипах чаще всего пауза с закрытыми глазами.
THUMBNAIL_FRACTION = 1 / 3


@dataclass(frozen=True, slots=True)
class RenderSettings:
    spec: RenderSpec = field(default_factory=RenderSpec)
    vertical: bool = True
    thumbnails: bool = True
    concurrency: int = 2
    video_mime: str = "video/mp4"
    image_mime: str = "image/jpeg"


@dataclass(frozen=True, slots=True)
class RenderResult:
    job_id: UUID
    clips_total: int
    rendered: int
    failed: int
    vertical: int
    thumbnails: int
    duration_sec: float
    #: Каким режимом собрана вертикаль, по числу клипов: track/static/blurpad.
    #: Не отладочная мелочь: доля ``blurpad`` — это доля клипов, в которых
    #: не нашлось ни одного лица, а по ней видно, годится ли материал
    #: для вертикального формата вообще.
    modes: dict[str, int] = field(default_factory=dict)
    warnings: Sequence[str] = field(default_factory=tuple)
    skipped: bool = False
    reused: bool = False


class RenderClips:
    """Сценарий этапа нарезки."""

    def __init__(
        self,
        *,
        uow_factory: UnitOfWorkFactory,
        renderer: ClipRenderer,
        storage: ObjectStorage,
        provider_resolver: ProviderResolver,
        workdir: Path,
        planner: CropPlanner | None = None,
        settings: RenderSettings | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._renderer = renderer
        self._storage = storage
        self._resolve_provider = provider_resolver
        self._workdir = workdir
        self._planner = planner
        self._settings = settings or RenderSettings()
        self._tracker = StageTracker(uow_factory=uow_factory, stage=STAGE)

    async def execute(self, job_id: UUID) -> RenderResult:
        started = time.monotonic()
        job, video = await self._tracker.load(job_id)

        if job.is_stage_done(STAGE):
            async with self._uow_factory() as uow:
                existing = await uow.clips.list_for_job(job_id)
            return RenderResult(
                job_id=job_id,
                clips_total=len(existing),
                rendered=sum(1 for clip in existing if clip.status.value == "ready"),
                failed=0,
                vertical=0,
                thumbnails=0,
                duration_sec=0.0,
                reused=True,
            )

        async with self._uow_factory() as uow:
            clips = list(await uow.clips.list_for_job(job_id))

        if not clips:
            await self._tracker.skip(job, "нет клипов: этап отбора не выполнялся")
            return RenderResult(job_id, 0, 0, 0, 0, 0, 0.0, skipped=True)

        await self._tracker.begin(job)

        try:
            result = await self._run(job, video, clips, started)
        except Exception as error:
            await self._tracker.fail(job, error, duration_sec=_elapsed(started))
            raise

        await self._tracker.succeed(job, duration_sec=result.duration_sec)
        log.info(
            "render.done",
            job_id=str(job_id),
            rendered=result.rendered,
            failed=result.failed,
            modes=result.modes,
            duration_sec=result.duration_sec,
        )
        return result

    # --------------------------------------------------------------- шаги
    async def _run(
        self,
        job: Job,
        video: Video,
        clips: Sequence[Clip],
        started: float,
    ) -> RenderResult:
        provider = self._resolve_provider(video.source_uri)
        media = await provider.acquire(video.source_uri)
        target = self._workdir / "clips" / str(job.id)
        target.mkdir(parents=True, exist_ok=True)

        try:
            rendered = await self._render_all(job, media, clips, target)
            await self._publish(job, clips, rendered)
        finally:
            await provider.release(media)
            await asyncio.to_thread(_cleanup, target)

        ok = [item for item in rendered if item.ok]
        if not ok:
            raise ProcessingError(
                "Не удалось собрать ни одного клипа",
                job_id=str(job.id),
                clips=len(clips),
                first_error=next((item.error for item in rendered if item.error), None),
            )

        modes: dict[str, int] = {}
        for item in ok:
            if item.reframe_mode:
                modes[item.reframe_mode] = modes.get(item.reframe_mode, 0) + 1

        return RenderResult(
            job_id=job.id,
            clips_total=len(clips),
            rendered=len(ok),
            failed=len(rendered) - len(ok),
            vertical=sum(1 for item in ok if item.vertical is not None),
            thumbnails=sum(1 for item in ok if item.thumbnail is not None),
            duration_sec=_elapsed(started),
            modes=modes,
            warnings=tuple(warning for item in rendered for warning in item.warnings),
        )

    async def _render_all(
        self,
        job: Job,
        media: LocalMedia,
        clips: Sequence[Clip],
        target: Path,
    ) -> list[RenderedClip]:
        semaphore = asyncio.Semaphore(max(self._settings.concurrency, 1))
        done = 0
        lock = asyncio.Lock()

        async def worker(clip: Clip) -> RenderedClip:
            nonlocal done
            async with semaphore:
                result = await self._render_one(media, clip, target)
            async with lock:
                done += 1
                # Прогресс пишется по мере готовности клипов: этап идёт
                # минутами, и без этого джоб выглядит снаружи замершим.
                await self._tracker.progress(job, done / len(clips))
            return result

        return list(await asyncio.gather(*(worker(clip) for clip in clips)))

    async def _render_one(self, media: LocalMedia, clip: Clip, target: Path) -> RenderedClip:
        spec = self._settings.spec
        stem = f"{clip.rank:03d}"
        try:
            horizontal = await self._renderer.cut(media, clip.range, spec, target / f"{stem}_h.mp4")
        except Exception as error:
            log.warning("render.clip_failed", rank=clip.rank, error=str(error))
            return RenderedClip(clip_id=clip.id, error=f"{type(error).__name__}: {error}")

        warnings: list[str] = []
        vertical: Path | None = None
        mode: str | None = None
        if self._settings.vertical and self._planner is not None:
            vertical, mode, warning = await self._make_vertical(horizontal, clip, target)
            if warning:
                warnings.append(warning)

        thumbnail: Path | None = None
        if self._settings.thumbnails:
            try:
                thumbnail = await self._renderer.thumbnail(
                    horizontal,
                    thumbnail_moment(clip.range.duration),
                    target / f"{stem}.jpg",
                )
            except Exception as error:
                warnings.append(f"клип {clip.rank}: превью — {error}")

        return RenderedClip(
            clip_id=clip.id,
            horizontal=horizontal,
            vertical=vertical,
            thumbnail=thumbnail,
            duration_sec=clip.range.duration,
            reframe_mode=mode,
            warnings=tuple(warnings),
        )

    async def _make_vertical(
        self,
        horizontal: Path,
        clip: Clip,
        target: Path,
    ) -> tuple[Path | None, str | None, str | None]:
        """Вертикаль — или причина, по которой её нет.

        Провал здесь не отменяет клип: горизонтальный файл уже собран,
        и терять его из-за неудавшегося кадрирования незачем.
        """
        if self._planner is None:  # pragma: no cover — проверено вызывающим
            return None, None, None
        try:
            track = await self._planner.plan(
                horizontal,
                target_width=VERTICAL_WIDTH,
                target_height=VERTICAL_HEIGHT,
            )
            path, mode = await self._renderer.reframe_vertical(
                horizontal,
                track,
                self._settings.spec,
                target / f"{clip.rank:03d}_v.mp4",
            )
        except Exception as error:
            log.warning("render.vertical_failed", rank=clip.rank, error=str(error))
            return None, None, f"клип {clip.rank}: вертикаль — {error}"
        return path, mode, None

    async def _publish(
        self,
        job: Job,
        clips: Sequence[Clip],
        rendered: Sequence[RenderedClip],
    ) -> None:
        """Файлы в хранилище, ссылки и статусы — в базу.

        Загрузка идёт до транзакции, запись — одной транзакцией после.
        Наоборот было бы хуже: запись в базе, что файл готов, при упавшей
        загрузке — это ссылка в никуда, а лишний файл в хранилище без
        записи всего лишь занимает место и убирается по TTL.
        """
        by_id = {clip.id: clip for clip in clips}
        uploads: list[tuple[Clip, RenderedClip, list[tuple[AssetKind, str, str, int]]]] = []

        for item in rendered:
            clip = by_id[item.clip_id]
            if not item.ok:
                uploads.append((clip, item, []))
                continue
            assets = await self._upload(job.id, clip, item)
            uploads.append((clip, item, assets))

        async with self._uow_factory() as uow:
            for clip, item, assets in uploads:
                updated = clip.rendered() if item.ok else clip.broken(item.error or "неизвестно")
                await uow.clips.update(updated)
                # Перезапуск этапа не должен плодить вторые записи об
                # ассетах на те же ключи — старые удаляются целиком.
                await uow.assets.delete_for_owner(AssetOwnerType.CLIP, clip.id)
                for kind, key, mime, size in assets:
                    await uow.assets.add(
                        owner_type=AssetOwnerType.CLIP,
                        owner_id=clip.id,
                        kind=kind,
                        storage_key=key,
                        mime=mime,
                        size_bytes=size,
                        meta={"rank": clip.rank, "reframe_mode": item.reframe_mode},
                    )
            await uow.commit()

    async def _upload(
        self,
        job_id: UUID,
        clip: Clip,
        item: RenderedClip,
    ) -> list[tuple[AssetKind, str, str, int]]:
        settings = self._settings
        plan: list[tuple[Path | None, AssetKind, str, str]] = [
            (item.horizontal, AssetKind.CLIP_H, clip_key(job_id, clip.rank), settings.video_mime),
            (
                item.vertical,
                AssetKind.CLIP_V,
                clip_key(job_id, clip.rank, vertical=True),
                settings.video_mime,
            ),
            (
                item.thumbnail,
                AssetKind.THUMBNAIL,
                thumbnail_key(job_id, clip.rank),
                settings.image_mime,
            ),
        ]

        stored: list[tuple[AssetKind, str, str, int]] = []
        for path, kind, key, mime in plan:
            if path is None:
                continue
            size = path.stat().st_size
            await self._storage.put(key, path, content_type=mime)
            stored.append((kind, key, mime, size))
        return stored


def thumbnail_moment(duration_sec: float) -> float:
    """Момент внутри клипа, с которого берётся превью."""
    return max(duration_sec * THUMBNAIL_FRACTION, 0.0)


def _cleanup(directory: Path) -> None:
    """Убрать временные файлы клипов: они уже в хранилище.

    Десять клипов по 45 секунд — это сотни мегабайт на диске воркера,
    и оставлять их означает, что через сутки работы диск кончится.
    """
    if not directory.exists():
        return
    for item in directory.iterdir():
        item.unlink(missing_ok=True)
    directory.rmdir()


def _elapsed(started: float) -> float:
    return round(time.monotonic() - started, 3)
