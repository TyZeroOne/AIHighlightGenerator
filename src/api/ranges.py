"""Разбор заголовка ``Range`` — перемотка в плеере.

Браузер, открывая видео в ``<video>``, почти никогда не запрашивает файл
целиком. Он просит первые байты, читает заголовок контейнера, а дальше
ходит за кусками — и на каждый запрос ждёт ``206 Partial Content``
с заголовком ``Content-Range``. Ответ ``200`` с полным файлом формально
допустим, но означает, что перемотка работает через полную загрузку.

## Что поддерживается

Один диапазон вида ``bytes=START-END``, включая формы ``bytes=1000-``
(до конца) и ``bytes=-500`` (последние 500 байт). Множественные
диапазоны (``bytes=0-100,200-300``) — нет: они требуют ответа
``multipart/byteranges``, и ни один плеер их не просит.

## Почему разбор отдельным модулем

Он полностью описывается парой «строка + размер файла → пара чисел»,
не зависит ни от FastAPI, ни от хранилища, и все крайние случаи
(перевёрнутый диапазон, выход за конец, мусор вместо цифр) проверяются
таблицей значений.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

__all__ = ["ByteRange", "parse_range"]

_PATTERN: Final = re.compile(r"^bytes=(\d*)-(\d*)$")


@dataclass(frozen=True, slots=True)
class ByteRange:
    """Разобранный диапазон: полуоткрытым его делать нельзя.

    В HTTP границы **включительные**: ``bytes=0-0`` — это один байт.
    Расхождение с питоновскими срезами стоит одного лишнего байта
    в ответе и битого видео у клиента, поэтому длина считается здесь,
    а не на месте использования.
    """

    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    def content_range(self, total: int) -> str:
        return f"bytes {self.start}-{self.end}/{total}"


def parse_range(header: str | None, size: int) -> ByteRange | None:
    """``Range`` + размер файла → диапазон, или ``None``.

    ``None`` означает «отдавай целиком» и возвращается на любой запрос,
    который мы не понимаем. Это намеренно: заголовок необязательный,
    и отвечать ошибкой на непонятную форму значит ломать клиента там,
    где можно просто отдать файл.

    Единственное исключение — диапазон, начинающийся за концом файла:
    по RFC 9110 на него полагается ``416``, и вызывающий отличает этот
    случай по исключению :class:`ValueError`.
    """
    if not header or size <= 0:
        return None

    match = _PATTERN.match(header.strip())
    if match is None:
        return None

    raw_start, raw_end = match.group(1), match.group(2)
    if not raw_start and not raw_end:
        return None

    if not raw_start:
        # ``bytes=-500`` — последние 500 байт. Просят так редко, но
        # именно так mp4-плееры добирают хвост с индексом, если файл
        # собран без faststart.
        length = min(int(raw_end), size)
        return ByteRange(size - length, size - 1) if length else None

    start = int(raw_start)
    if start >= size:
        raise ValueError(f"Диапазон начинается за концом файла: {start} >= {size}")

    end = int(raw_end) if raw_end else size - 1
    end = min(end, size - 1)
    if end < start:
        return None
    return ByteRange(start, end)
