"""Хранилище на локальной файловой системе.

Используется в разработке и в однохостовой установке. В проде вместо него
S3/MinIO — интерфейс тот же, вызывающий код не меняется.
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import AsyncIterator
from pathlib import Path

from core.domain.errors import ObjectNotFoundError, StorageError
from core.logging import get_logger

__all__ = ["LocalStorage"]

log = get_logger(__name__)


class LocalStorage:
    """Реализация ``ObjectStorage`` поверх каталога на диске."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ ключи
    def _path(self, key: str) -> Path:
        """Ключ → путь на диске, с защитой от выхода за пределы корня.

        Ключ может прийти из БД, а туда — из данных пользователя. Ключ вида
        ``../../etc/passwd`` без этой проверки дал бы чтение и запись где угодно.
        Проверяем не строку (её легко обмануть кодированием), а результат
        разрешения пути.
        """
        if not key or key.startswith("/"):
            raise StorageError(f"Некорректный ключ: {key!r}", key=key)

        candidate = (self._root / key).resolve()
        if candidate != self._root and self._root not in candidate.parents:
            raise StorageError(f"Ключ выходит за пределы хранилища: {key!r}", key=key)
        return candidate

    # ------------------------------------------------------------ запись
    async def put(self, key: str, source: Path, *, content_type: str | None = None) -> str:
        target = self._path(key)

        def _copy() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            # Пишем во временный файл рядом и переименовываем: rename в пределах
            # одной ФС атомарен, поэтому читатель никогда не увидит наполовину
            # записанный объект. Прямая запись в target такой гарантии не даёт.
            tmp = target.with_suffix(target.suffix + ".partial")
            shutil.copyfile(source, tmp)
            tmp.replace(target)

        await asyncio.to_thread(_copy)
        log.debug("storage.put", key=key, size=target.stat().st_size)
        return key

    # ------------------------------------------------------------ чтение
    async def get(self, key: str, destination: Path) -> Path:
        source = self._path(key)
        if not source.exists():
            raise ObjectNotFoundError(f"Объект не найден: {key}", key=key)

        def _copy() -> None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)

        await asyncio.to_thread(_copy)
        return destination

    async def local_path(self, key: str) -> Path | None:
        """Путь к объекту на диске, если он там есть.

        Не входит в порт ``ObjectStorage`` — это оптимизация, специфичная для
        локального хранилища: воркеру незачем копировать файл сам в себя.
        Вызывающий код обязан проверять наличие метода, а не полагаться на него.
        """
        path = self._path(key)
        return path if path.exists() else None

    async def _stream(
        self,
        key: str,
        chunk_size: int,
        offset: int,
        length: int | None,
    ) -> AsyncIterator[bytes]:
        path = self._path(key)
        if not path.exists():
            raise ObjectNotFoundError(f"Объект не найден: {key}", key=key)
        # Файл читается в пуле потоков: обычный read блокирует event loop,
        # а отдача 2 ГБ заблокировала бы его надолго.
        handle = await asyncio.to_thread(path.open, "rb")
        try:
            if offset:
                await asyncio.to_thread(handle.seek, offset)
            # ``remaining is None`` означает «до конца файла»; иначе считаем
            # оставшиеся байты и читаем ровно столько, сколько просили.
            remaining = length
            while True:
                want = chunk_size if remaining is None else min(chunk_size, remaining)
                if want <= 0:
                    break
                chunk = await asyncio.to_thread(handle.read, want)
                if not chunk:
                    break
                if remaining is not None:
                    remaining -= len(chunk)
                yield chunk
        finally:
            await asyncio.to_thread(handle.close)

    def open_stream(
        self,
        key: str,
        *,
        chunk_size: int = 1 << 20,
        offset: int = 0,
        length: int | None = None,
    ) -> AsyncIterator[bytes]:
        return self._stream(key, chunk_size, offset, length)

    # ------------------------------------------------------- метаданные
    async def exists(self, key: str) -> bool:
        return await asyncio.to_thread(self._path(key).exists)

    async def size(self, key: str) -> int:
        path = self._path(key)
        if not path.exists():
            raise ObjectNotFoundError(f"Объект не найден: {key}", key=key)
        return (await asyncio.to_thread(path.stat)).st_size

    async def delete(self, key: str) -> None:
        path = self._path(key)

        def _remove() -> None:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)  # отсутствие объекта — не ошибка

        await asyncio.to_thread(_remove)

    async def delete_prefix(self, prefix: str) -> int:
        """Удалить все объекты с общим префиксом. Возвращает число удалённых."""
        base = self._path(prefix)

        def _remove_tree() -> int:
            if not base.exists():
                return 0
            count = sum(1 for path in base.rglob("*") if path.is_file())
            shutil.rmtree(base, ignore_errors=True)
            return count

        return await asyncio.to_thread(_remove_tree)

    async def signed_url(self, key: str, *, ttl_sec: int = 900) -> str | None:
        """Локальное хранилище не умеет прямых ссылок — файл отдаёт API.

        Возвращаем ``None``, а не бросаем исключение: вызывающий код проверяет
        результат и выбирает способ отдачи. Так один и тот же роутер работает
        с обоими бэкендами.
        """
        return None

    async def healthcheck(self) -> None:
        def _probe() -> None:
            self._root.mkdir(parents=True, exist_ok=True)
            marker = self._root / ".healthcheck"
            marker.write_text("ok", encoding="utf-8")
            marker.unlink()

        await asyncio.to_thread(_probe)
