"""Мост между синхронным Celery и асинхронным ядром.

Задачи Celery — обычные синхронные функции, сценарии — корутины. Соединить
их можно тремя способами, и выбор здесь не косметический.

``asyncio.run()`` на каждую задачу — самый очевидный и самый неправильный:
он создаёт и закрывает event loop каждый раз, а пул соединений SQLAlchemy
привязан к тому циклу, в котором соединение открылось. Вторая задача в том же
процессе получит из пула соединение чужого, уже закрытого цикла — и упадёт
на ``Future attached to a different loop``. Ошибка не воспроизводится на
первом запуске и потому особенно неприятна.

``asgiref.sync.async_to_sync`` тянет зависимость ради того же самого.

Остаётся то, что здесь: **один постоянный event loop на процесс-воркер**.
Пул живёт вместе с процессом, соединения переиспользуются между задачами,
а prefork-модель Celery гарантирует, что задачи в процессе идут по одной —
конкуренции за цикл нет.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

from core.logging import get_logger

__all__ = ["close_loop", "get_loop", "run_async"]

log = get_logger(__name__)

_loop: asyncio.AbstractEventLoop | None = None


def get_loop() -> asyncio.AbstractEventLoop:
    """Event loop процесса. Создаётся при первом обращении."""
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
    return _loop


def run_async[T](coro: Coroutine[Any, Any, T]) -> T:
    """Выполнить корутину в постоянном цикле процесса."""
    return get_loop().run_until_complete(coro)


def close_loop() -> None:
    """Закрыть цикл при остановке воркера.

    Вызывается после закрытия контейнера: сначала отпускаем соединения БД,
    потом гасим цикл, в котором они жили. Обратный порядок оставил бы
    незакрытые сокеты и предупреждения в логе остановки.
    """
    global _loop
    if _loop is not None and not _loop.is_closed():
        _loop.close()
    _loop = None
