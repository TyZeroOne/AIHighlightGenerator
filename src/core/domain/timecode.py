"""Преобразование тайм-кодов ``HH:MM:SS.mmm`` ↔ секунды.

Живёт в домене, потому что тайм-код — часть языка предметной области:
он приходит от LLM, уходит в API и в субтитры. Никаких зависимостей.
"""

from __future__ import annotations

import re
from typing import Final

__all__ = ["format_timecode", "parse_timecode"]

# HH:MM:SS с опциональными миллисекундами. Часы — 2+ цифры (видео может быть длиннее 99 ч).
_TIMECODE_RE: Final = re.compile(
    r"^(?P<h>\d{2,}):(?P<m>[0-5]\d):(?P<s>[0-5]\d)(?:\.(?P<ms>\d{1,3}))?$"
)

_SECONDS_IN_MINUTE: Final = 60
_SECONDS_IN_HOUR: Final = 3600


def parse_timecode(value: str) -> float:
    """``"00:15:21.480"`` → ``921.48``.

    :raises ValueError: если строка не соответствует формату ``HH:MM:SS[.mmm]``.
    """
    match = _TIMECODE_RE.match(value.strip())
    if match is None:
        raise ValueError(f"Некорректный тайм-код: {value!r}. Ожидается HH:MM:SS[.mmm]")

    hours = int(match["h"])
    minutes = int(match["m"])
    seconds = int(match["s"])
    # "5" означает 500 мс, а не 5 мс — дополняем справа нулями до трёх знаков.
    millis = int((match["ms"] or "0").ljust(3, "0"))

    return hours * _SECONDS_IN_HOUR + minutes * _SECONDS_IN_MINUTE + seconds + millis / 1000.0


def format_timecode(seconds: float, *, with_millis: bool = True) -> str:
    """``921.48`` → ``"00:15:21.480"``.

    :raises ValueError: при отрицательном значении.
    """
    if seconds < 0:
        raise ValueError(f"Тайм-код не может быть отрицательным: {seconds}")

    # Округляем до миллисекунд ДО разбора на части, иначе 59.9996 даст "00:00:59.1000".
    total_millis = round(seconds * 1000)
    millis = total_millis % 1000
    total_seconds = total_millis // 1000

    hours, remainder = divmod(total_seconds, _SECONDS_IN_HOUR)
    minutes, secs = divmod(remainder, _SECONDS_IN_MINUTE)

    base = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{base}.{millis:03d}" if with_millis else base
