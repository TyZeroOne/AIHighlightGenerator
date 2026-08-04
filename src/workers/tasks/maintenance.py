"""Периодические задачи: уборка артефактов и снятие зависших джобов.

Запускаются расписанием (``celery beat``), а не человеком. Отсюда два
требования, которых нет у обычных задач конвейера.

**Никаких ретраев.** Задача, упавшая по расписанию, повторится сама
через час. Повтор внутри же означал бы шторм попыток на недоступном
хранилище — ровно тогда, когда ему и так плохо.

**Никакого ``acks_late``.** Уборка идемпотентна по построению, но её
двойной запуск на одном наборе файлов — это лишние обращения к S3,
а не польза. Пропустить один час уборки безопаснее, чем выполнить её
дважды.
"""

from __future__ import annotations

from typing import Any

from celery import Task

from core.container import get_container
from core.logging import get_logger
from workers.app import celery_app
from workers.runtime import run_async

__all__ = ["cleanup_expired_assets", "reap_stuck_jobs"]

log = get_logger(__name__)


class MaintenanceTask(Task):
    """База для задач по расписанию.

    Намеренно не ``PipelineTask``: у той ретраи, ``acks_late`` и отметки
    этапов джоба — всё то, чего у уборки нет и быть не должно.
    """

    max_retries = 0
    acks_late = False
    ignore_result = True


@celery_app.task(
    base=MaintenanceTask,
    name="workers.tasks.maintenance.cleanup_expired_assets",
)
def cleanup_expired_assets() -> dict[str, Any]:
    """Удалить файлы, у которых вышел срок хранения.

    Час аудио в WAV — это 115 МБ, которые после анализа не нужны никому.
    Без уборки диск кончается за неделю работы, и замечают это в тот
    момент, когда очередной джоб падает на записи файла.
    """
    result = run_async(get_container().cleanup_assets.execute())
    return {
        "examined": result.examined,
        "removed": result.removed,
        "failed": result.failed,
        "duration_sec": result.duration_sec,
    }


@celery_app.task(
    base=MaintenanceTask,
    name="workers.tasks.maintenance.reap_stuck_jobs",
)
def reap_stuck_jobs() -> dict[str, Any]:
    """Снять джобы, которые давно не подают признаков жизни.

    Воркер, убитый по OOM, не успевает отметить провал: джоб остаётся
    ``running`` навсегда и снаружи неотличим от живого.
    """
    result = run_async(get_container().reap_stuck_jobs.execute())
    return {
        "reaped": [str(job_id) for job_id in result.reaped],
        "count": result.count,
        "duration_sec": result.duration_sec,
    }
