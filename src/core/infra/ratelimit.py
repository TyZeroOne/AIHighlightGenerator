"""Ограничитель частоты запросов: Redis и память процесса.

## Фиксированное окно, а не скользящее

Скользящее окно точнее: оно не даёт всплеска на стыке, когда клиент
успевает сделать лимит запросов в конце одной минуты и столько же
в начале следующей. Но стоит оно хранения меток каждого запроса
(sorted set с обрезкой) вместо одного целого числа.

Всплеск вдвое на границе окна нас не убивает: лимит защищает от
случайного цикла в чужом скрипте и от перебора ключей, а не от
распределённой атаки, — для второго нужен совсем другой рубеж, и он
не в приложении.

## Почему счётчик в Redis, а не в памяти

Реплик API несколько. Счётчик в памяти означал бы, что реальный лимит
умножается на их число, причём непредсказуемо: балансировщик распределяет
запросы как хочет. Общий счётчик — единственный вариант, дающий тот
лимит, который написан в конфиге.

Реализация в памяти оставлена для тестов и для одиночного запуска
без Redis.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import TYPE_CHECKING, Final

from core.logging import get_logger
from core.ports.limits import RateVerdict

if TYPE_CHECKING:
    from redis.asyncio import Redis

__all__ = ["InMemoryRateLimiter", "RedisRateLimiter"]

log = get_logger(__name__)

#: Префикс ключей. Общий с остальным приложением Redis используется и под
#: брокер, и под результаты — без префикса ключи ограничителя смешались бы
#: с очередями Celery.
KEY_PREFIX: Final = "ratelimit:"


class RedisRateLimiter:
    """Счётчик в Redis. Реализация порта ``RateLimiter``."""

    def __init__(self, client: Redis) -> None:
        self._client = client

    async def hit(self, key: str, *, limit: int, window_sec: int) -> RateVerdict:
        """Инкремент со сроком жизни — атомарно, одним конвейером.

        ``INCR`` и ``EXPIRE`` идут вместе намеренно: между ними процесс
        может умереть, и ключ без срока жизни остался бы навсегда —
        а вместе с ним и запрет для клиента.

        Срок ставится **на каждом** запросе, а не только на первом.
        Так окно скользит вслед за активностью, и клиент, который стучит
        непрерывно, не получает бесплатный сброс от того, что ключ
        случайно пережил предыдущее окно.
        """
        redis_key = f"{KEY_PREFIX}{key}"
        try:
            async with self._client.pipeline(transaction=True) as pipe:
                pipe.incr(redis_key)
                pipe.expire(redis_key, window_sec, nx=True)
                pipe.ttl(redis_key)
                count, _, ttl = await pipe.execute()
        except Exception as error:
            # Отказ Redis не должен закрывать API. Ограничитель — защита
            # от перебора, а не от несанкционированного доступа; работать
            # без него хуже, чем не работать вовсе.
            log.warning("ratelimit.unavailable", error=str(error))
            return RateVerdict(allowed=True, remaining=limit, retry_after_sec=0, limit=limit)

        used = int(count)
        retry_after = int(ttl) if int(ttl) > 0 else window_sec
        return RateVerdict(
            allowed=used <= limit,
            remaining=max(limit - used, 0),
            retry_after_sec=retry_after,
            limit=limit,
        )


class InMemoryRateLimiter:
    """Тот же счётчик в памяти процесса — для тестов и одиночного запуска.

    Честно непригоден для нескольких реплик: у каждой будет свой счётчик,
    и фактический лимит умножится на их число. Это записано здесь, а не
    в документации, потому что читать будут отсюда.
    """

    def __init__(self, clock: object | None = None) -> None:
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._now = clock or time.monotonic

    async def hit(self, key: str, *, limit: int, window_sec: int) -> RateVerdict:
        now = float(self._now())  # type: ignore[operator]
        window_start = now - window_sec
        hits = [moment for moment in self._hits[key] if moment > window_start]
        hits.append(now)
        self._hits[key] = hits

        used = len(hits)
        oldest = hits[0]
        return RateVerdict(
            allowed=used <= limit,
            remaining=max(limit - used, 0),
            retry_after_sec=max(int(oldest + window_sec - now) + 1, 1),
            limit=limit,
        )
