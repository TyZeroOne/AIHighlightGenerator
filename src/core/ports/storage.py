"""Порт файлового хранилища.

Одна и та же абстракция закрывает локальную ФС в разработке и S3/MinIO
в проде. Благодаря ей воркеры не обязаны иметь общий диск с API-сервером —
а значит, их можно разнести по разным машинам.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol

__all__ = ["ObjectStorage"]


class ObjectStorage(Protocol):
    async def put(self, key: str, source: Path, *, content_type: str | None = None) -> str:
        """Положить файл под ключ. Возвращает канонический ключ."""
        ...

    async def get(self, key: str, destination: Path) -> Path:
        """Скачать в локальный путь. Каталог создаётся при необходимости."""
        ...

    def open_stream(self, key: str, *, chunk_size: int = 1 << 20) -> AsyncIterator[bytes]:
        """Потоковое чтение — чтобы отдать 2 ГБ клиенту, не читая их в память.

        Объявлен как обычный ``def``, возвращающий ``AsyncIterator``, а не как
        ``async def``: реализация — асинхронный генератор, а вызов генератора
        сразу отдаёт итератор, без промежуточного ``await``.
        """
        ...

    async def delete(self, key: str) -> None:
        """Удалить. Отсутствие объекта — не ошибка (операция идемпотентна)."""
        ...

    async def exists(self, key: str) -> bool: ...

    async def size(self, key: str) -> int: ...

    async def signed_url(self, key: str, *, ttl_sec: int = 900) -> str | None:
        """Временная прямая ссылка.

        Для S3 — presigned URL, байты идут мимо приложения.
        Для локального хранилища — ``None``: прямых ссылок не бывает,
        API отдаёт файл сам.
        """
        ...

    async def healthcheck(self) -> None:
        """Проверка доступности для ``/health/ready``. Бросает исключение при сбое."""
        ...
