"""Поток событий прогресса (Server-Sent Events).

## Почему SSE, а не WebSocket и не поллинг

Поллинг статуса раз в секунду на сотне клиентов — это сотня запросов
в секунду, каждый со своей аутентификацией и своим запросом к базе,
ради ответа, который в 90 % случаев не изменился.

WebSocket решил бы это, но он двусторонний, а нам нужно одно
направление: сервер рассказывает, клиент слушает. За двусторонность
пришлось бы платить сложностью — своим протоколом поверх, обработкой
переподключений, отдельным путём через прокси.

SSE — это обычный HTTP-ответ, который не заканчивается. Работает через
любой прокси, переподключается браузером сам, читается в четыре строки
на клиенте.

## Что мешает SSE работать

**Буферизация.** Прокси между нами и клиентом (nginx, traefik, ELB)
по умолчанию копит ответ, чтобы отдать его одним куском. Для потока
это означает, что клиент не увидит ничего до самого конца — то есть
ровно противоположное задуманному. Лечится заголовком
``X-Accel-Buffering: no``, который nginx понимает напрямую, и
``Cache-Control: no-cache`` для всех остальных.

**Обрыв клиента.** Закрытая вкладка не сообщает о себе никак: генератор
продолжает опрашивать базу для никого. Поэтому обрыв проверяется явно
на каждом цикле.

**Молчание.** Прокси и балансировщики закрывают соединение, по которому
ничего не идёт, обычно через 30–60 секунд. Джоб на этапе распознавания
речи молчит минутами — поэтому в тишину уходит комментарий-heartbeat:
строка, начинающаяся с двоеточия, которую спецификация велит
игнорировать, а прокси считает трафиком.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any, Final

__all__ = ["HEARTBEAT_SEC", "POLL_SEC", "SSE_HEADERS", "format_event", "heartbeat"]

#: Как часто спрашиваем состояние. Секунда — компромисс: чаще не имеет
#: смысла (этапы длятся минутами), реже заметно на глаз.
POLL_SEC: Final = 1.0

#: Как часто отправляем комментарий в тишину, чтобы прокси не закрыл
#: соединение. 15 секунд — вдвое меньше типичного таймаута в 30.
HEARTBEAT_SEC: Final = 15.0

SSE_HEADERS: Final[dict[str, str]] = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # Понимает nginx; для остальных прокси работает Cache-Control.
    "X-Accel-Buffering": "no",
}


def format_event(event: str, payload: dict[str, Any]) -> str:
    """Одно событие в формате SSE.

    Двойной перевод строки в конце — не стиль, а разделитель событий:
    без него клиент считает следующее событие продолжением текущего.
    ``ensure_ascii=False`` — чтобы русские названия этапов и ошибки
    оставались читаемыми в потоке.
    """
    body = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {body}\n\n"


def heartbeat() -> str:
    """Комментарий, который клиент игнорирует, а прокси считает трафиком."""
    return ": ping\n\n"


async def stream_events(
    *,
    snapshot: Callable[[], Coroutine[Any, Any, dict[str, Any] | None]],
    is_disconnected: Callable[[], Coroutine[Any, Any, bool]],
    sleep: Callable[[float], Coroutine[Any, Any, None]],
    changed: Callable[[dict[str, Any] | None, dict[str, Any]], bool],
    is_final: Callable[[dict[str, Any]], bool],
) -> AsyncIterator[str]:
    """Общий цикл потока: опроси, сравни, отправь.

    Все зависимости переданы функциями — не ради красоты, а потому что
    иначе этот цикл нельзя проверить: он состоит из времени, обрывов
    соединения и запросов к базе. С функциями он проверяется списком
    состояний и фейковым сном.
    """
    previous: dict[str, Any] | None = None
    silence = 0.0

    while True:
        if await is_disconnected():
            return

        current = await snapshot()
        if current is None:
            yield format_event("error", {"message": "Джоб исчез"})
            return

        if changed(previous, current):
            yield format_event("progress", current)
            previous = current
            silence = 0.0

        if is_final(current):
            yield format_event("done", current)
            return

        if silence >= HEARTBEAT_SEC:
            yield heartbeat()
            silence = 0.0

        await sleep(POLL_SEC)
        silence += POLL_SEC
