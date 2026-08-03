"""Базовый класс задач конвейера.

Вся общая механика — ретраи, трассировка, отметки этапов — живёт здесь,
а не копируется в каждую задачу. Конкретная задача остаётся коротким
куском по существу дела.
"""

from __future__ import annotations

from typing import Any, ClassVar

from celery import Task

from core.domain.enums import JobStage
from core.domain.errors import ExternalServiceError, TransientError
from core.logging import bind_trace, get_logger

__all__ = ["PipelineTask"]

log = get_logger(__name__)


class PipelineTask(Task):
    """База для всех задач конвейера.

    Ключевое решение — **избирательный ретрай**. Ретраить всё подряд вредно:
    задача, упавшая из-за битого файла, будет падать одинаково пять раз
    подряд, занимая воркер и отодвигая живые задачи. Поэтому повторяются
    только ``TransientError`` и ``ExternalServiceError`` — те, где повтор
    имеет шанс сработать.
    """

    #: Этап конвейера, который представляет задача. Переопределяется наследником.
    stage: ClassVar[JobStage | None] = None

    autoretry_for = (TransientError, ExternalServiceError)
    retry_backoff = True  # 1с, 2с, 4с, 8с... вместо шторма повторов
    retry_backoff_max = 600
    retry_jitter = True  # разброс, чтобы упавшие одновременно задачи
    # не пошли на повтор в один и тот же момент
    max_retries = 5
    acks_late = True
    track_started = True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        job_id = kwargs.get("job_id") or (args[0] if args else None)
        with bind_trace(job_id=job_id, stage=self.stage.value if self.stage else None):
            return super().__call__(*args, **kwargs)

    def on_failure(
        self,
        exc: Exception,
        task_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        einfo: Any,
    ) -> None:
        log.error(
            "task.failed",
            task=self.name,
            task_id=task_id,
            stage=self.stage.value if self.stage else None,
            error_type=type(exc).__name__,
            error=str(exc),
        )

    def on_retry(
        self,
        exc: Exception,
        task_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        einfo: Any,
    ) -> None:
        log.warning(
            "task.retry",
            task=self.name,
            task_id=task_id,
            attempt=self.request.retries + 1,
            max_retries=self.max_retries,
            error=str(exc),
        )

    def on_success(
        self,
        retval: Any,
        task_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        log.debug("task.succeeded", task=self.name, task_id=task_id)
