"""Уборка: истёкшие артефакты и зависшие джобы.

Два сценария, которые никто не вызывает руками. Их запускает расписание,
и именно поэтому они написаны как всё остальное — через порты, с тестами
на фейках: код, который выполняется в три часа ночи без свидетелей,
проверить потом будет некому.

## Почему файлы удаляются раньше, чем записи

Порядок обратный тому, что кажется естественным. Сначала пропадает файл,
и только потом — запись о нём.

Причина в том, какой из двух видов рассогласования переживается. Запись
без файла — это ошибка «файл не найден» при скачивании: неприятно, но
починимо и заметно. Файл без записи — это байты, которые занимают место
и о которых никто не знает: их не найдёт ни одна уборка, потому что
искать она умеет только по записям.

## Почему зависший джоб — это отдельная проблема

Воркер, убитый по OOM или вместе с нодой, не успевает отметить провал.
Джоб остаётся ``running`` навсегда: снаружи он неотличим от живого, ждать
его можно бесконечно, а места в квоте он занимает как настоящий.

Признак — не время создания (двухчасовое видео обрабатывается сорок минут
и это норма), а время последнего изменения записи: живой джоб трогает её
на каждой отметке этапа.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from uuid import UUID

from core.domain.enums import JobStatus
from core.logging import get_logger
from core.ports.repository import UnitOfWorkFactory
from core.ports.storage import ObjectStorage

__all__ = ["CleanupExpiredAssets", "CleanupResult", "ReapResult", "ReapStuckJobs"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CleanupResult:
    examined: int = 0
    removed: int = 0
    failed: int = 0
    duration_sec: float = 0.0


@dataclass(frozen=True, slots=True)
class ReapResult:
    reaped: tuple[UUID, ...] = ()
    duration_sec: float = 0.0

    @property
    def count(self) -> int:
        return len(self.reaped)


class CleanupExpiredAssets:
    """Удалить артефакты, у которых вышел срок хранения."""

    def __init__(
        self,
        *,
        uow_factory: UnitOfWorkFactory,
        storage: ObjectStorage,
        batch_size: int = 500,
    ) -> None:
        self._uow_factory = uow_factory
        self._storage = storage
        self._batch_size = batch_size

    async def execute(self) -> CleanupResult:
        started = time.monotonic()

        async with self._uow_factory() as uow:
            expired = list(await uow.assets.expired(limit=self._batch_size))

        if not expired:
            return CleanupResult(duration_sec=_elapsed(started))

        removed: list[UUID] = []
        failed = 0
        for asset in expired:
            try:
                await self._storage.delete(asset.storage_key)
            except Exception as error:
                # Не падаем на одном файле: остальные удалить всё ещё нужно,
                # а этот попадёт в следующую уборку — запись о нём осталась.
                failed += 1
                log.warning("cleanup.delete_failed", key=asset.storage_key, error=str(error))
                continue
            removed.append(asset.id)

        async with self._uow_factory() as uow:
            await uow.assets.mark_deleted(removed)
            await uow.commit()

        log.info("cleanup.done", examined=len(expired), removed=len(removed), failed=failed)
        return CleanupResult(
            examined=len(expired),
            removed=len(removed),
            failed=failed,
            duration_sec=_elapsed(started),
        )


class ReapStuckJobs:
    """Пометить провалившимися джобы, которые давно не подают признаков жизни."""

    def __init__(
        self,
        *,
        uow_factory: UnitOfWorkFactory,
        timeout_sec: float,
        limit: int = 100,
    ) -> None:
        self._uow_factory = uow_factory
        self._timeout_sec = timeout_sec
        self._limit = limit

    async def execute(self) -> ReapResult:
        started = time.monotonic()

        async with self._uow_factory() as uow:
            stuck = list(await uow.jobs.stuck(older_than_sec=self._timeout_sec, limit=self._limit))
            for job in stuck:
                await uow.jobs.update(
                    replace(
                        job,
                        status=JobStatus.FAILED,
                        error={
                            "code": "STUCK",
                            "stage": job.stage.value if job.stage else None,
                            # Формулировка честная: мы не знаем, что случилось.
                            # Знаем только, что запись давно не менялась, —
                            # и что ждать дальше бессмысленно.
                            "message": (
                                f"Джоб не подавал признаков жизни дольше "
                                f"{self._timeout_sec / 3600:.1f} ч и снят по таймауту. "
                                "Вероятная причина — воркер, убитый до отметки о провале."
                            ),
                        },
                    )
                )
            if stuck:
                await uow.commit()

        if stuck:
            log.warning("reaper.jobs_failed", count=len(stuck), ids=[str(job.id) for job in stuck])
        return ReapResult(reaped=tuple(job.id for job in stuck), duration_sec=_elapsed(started))


def _elapsed(started: float) -> float:
    return round(time.monotonic() - started, 3)
