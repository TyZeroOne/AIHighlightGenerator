"""Служебные задачи: проверка, что конвейер вообще жив.

На Этапе 1 конвейера ещё нет, но инфраструктура очередей есть — и её нужно
уметь проверить, не дожидаясь Этапа 3. Отсюда ``ping`` и ``echo``.
"""

from __future__ import annotations

import platform
import time
from typing import Any

from core.logging import get_logger
from workers.app import celery_app
from workers.base import PipelineTask

__all__ = ["echo", "ping"]

log = get_logger(__name__)


@celery_app.task(base=PipelineTask, name="workers.tasks.system.ping")
def ping() -> dict[str, Any]:
    """Проверка живости воркера. Возвращает данные о процессе, который её выполнил.

    Полезнее простого ``"pong"``: по ``hostname`` видно, какой именно воркер
    взял задачу, — при отладке маршрутизации по очередям это первое, что нужно.
    """
    log.info("task.ping")
    return {
        "status": "pong",
        "hostname": platform.node(),
        "python": platform.python_version(),
        "timestamp": time.time(),
    }


@celery_app.task(base=PipelineTask, name="workers.tasks.system.echo")
def echo(payload: dict[str, Any]) -> dict[str, Any]:
    """Возвращает полученное. Нужна для проверки сериализации и цепочек задач."""
    log.info("task.echo", keys=sorted(payload))
    return payload
