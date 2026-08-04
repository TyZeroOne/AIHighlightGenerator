"""Порт ограничителя частоты запросов.

Отдельный порт ради одного метода — потому что реализаций осмысленно
две. В проде счётчик общий на все реплики API и живёт в Redis; в тестах
он в памяти процесса, иначе проверка «на одиннадцатом запросе приходит
429» требовала бы поднятого Redis.

Возврат — не булев флаг, а вердикт с числами: клиенту в заголовках надо
сказать не только «нельзя», но и «сколько осталось» и «когда можно».
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

__all__ = ["RateLimiter", "RateVerdict"]


@dataclass(frozen=True, slots=True)
class RateVerdict:
    allowed: bool
    #: Сколько запросов осталось в текущем окне.
    remaining: int
    #: Через сколько секунд окно сбросится. Уходит в ``Retry-After``.
    retry_after_sec: int
    limit: int = 0


class RateLimiter(Protocol):
    async def hit(self, key: str, *, limit: int, window_sec: int) -> RateVerdict:
        """Учесть запрос и сказать, можно ли его выполнять.

        Учитывать надо **до** ответа, а не после: смысл ограничителя
        в том, чтобы отказ стоил дёшево, а работа не начиналась вовсе.
        """
        ...
