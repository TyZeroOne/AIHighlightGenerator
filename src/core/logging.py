"""Структурное логирование и сквозной ``trace_id``.

Обычный текстовый лог бесполезен, когда один джоб проходит через API,
пять Celery-задач и три процесса. Нужен машиночитаемый формат и общий
идентификатор, по которому события собираются в одну историю.

``contextvars`` вместо передачи ``trace_id`` параметром: он автоматически
наследуется вложенными вызовами и корректно работает в asyncio, где
несколько задач исполняются в одном потоке.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from uuid import UUID, uuid4

import structlog

__all__ = ["bind_trace", "configure_logging", "get_logger", "get_trace_id"]

_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_job_id: ContextVar[str | None] = ContextVar("job_id", default=None)
_stage: ContextVar[str | None] = ContextVar("stage", default=None)

_configured = False


def _inject_context(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Процессор structlog: добавляет контекст в каждую запись автоматически."""
    for key, var in (("trace_id", _trace_id), ("job_id", _job_id), ("stage", _stage)):
        value = var.get()
        if value is not None:
            event_dict.setdefault(key, value)
    return event_dict


def configure_logging(*, level: str = "INFO", fmt: str = "console") -> None:
    """Настроить логирование. Вызывается один раз при старте процесса."""
    global _configured
    if _configured:
        return

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _inject_context,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    processors.append(
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    # Приводим сторонние библиотеки (uvicorn, sqlalchemy) к тому же уровню,
    # иначе они пишут своим форматом мимо нашего конвейера.
    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=level.upper())
    for noisy in ("uvicorn.access", "sqlalchemy.engine", "asyncio"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, logging.getLevelName(level.upper())))

    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]


def get_trace_id() -> str | None:
    return _trace_id.get()


@contextmanager
def bind_trace(
    *,
    trace_id: str | UUID | None = None,
    job_id: str | UUID | None = None,
    stage: str | None = None,
) -> Iterator[str]:
    """Привязать контекст к текущей задаче/запросу.

    Всё, что залогируется внутри блока — включая вложенные функции и
    ``await``-вызовы — автоматически получит эти поля.
    """
    resolved = str(trace_id) if trace_id is not None else str(uuid4())
    tokens = [
        _trace_id.set(resolved),
        _job_id.set(str(job_id) if job_id is not None else _job_id.get()),
        _stage.set(stage if stage is not None else _stage.get()),
    ]
    try:
        yield resolved
    finally:
        for var, token in zip((_trace_id, _job_id, _stage), tokens, strict=True):
            var.reset(token)
