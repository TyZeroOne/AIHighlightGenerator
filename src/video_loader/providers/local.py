"""Провайдер локальных файлов: ``file:///media/in/movie.mkv``.

Сценарий — том с видео смонтирован в контейнер, и загружать файл через HTTP
незачем. Типично для пакетной обработки архива.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import ClassVar
from urllib.parse import unquote, urlparse

from core.domain.enums import SourceKind
from core.domain.errors import SourceError
from core.domain.media import LocalMedia, SourceDescriptor
from core.domain.values import MediaWindow, TimeRange
from core.ports.source import MediaProbe
from core.registry import register_source

__all__ = ["LocalFileProvider"]


@register_source
class LocalFileProvider:
    """Реализация ``SourceProvider`` для файлов на диске воркера.

    Обратите внимание, чего здесь **нет**: ни скачивания, ни временных файлов,
    ни очистки. Для файла ``acquire()`` возвращает путь как есть, а ``release()``
    ничего не делает. Именно поэтому у ``LocalMedia`` есть флаг ``owned`` —
    он отличает «наш временный файл, удалить» от «чужой файл, не трогать».
    """

    scheme: ClassVar[str] = "file"

    def __init__(self, probe: MediaProbe, allowed_roots: tuple[Path, ...] = ()) -> None:
        self._probe = probe
        # Белый список каталогов. Без него URI `file:///etc/shadow` дал бы
        # чтение любого файла на хосте — классический path traversal
        # с полностью валидным на вид запросом.
        self._allowed_roots = tuple(root.resolve() for root in allowed_roots)

    @classmethod
    def supports(cls, uri: str) -> bool:
        return uri.startswith("file://")

    def _resolve(self, uri: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            raise SourceError(f"Не файловый URI: {uri!r}", uri=uri)

        # unquote — потому что пробелы и кириллица в пути приезжают как %20 и %D0.
        raw = unquote(parsed.path)
        # На Windows путь из URI выглядит как "/D:/video.mkv" — ведущий слэш лишний.
        if len(raw) > 2 and raw[0] == "/" and raw[2] == ":":
            raw = raw[1:]

        path = Path(raw).resolve()

        if self._allowed_roots and not any(
            path == root or root in path.parents for root in self._allowed_roots
        ):
            raise SourceError(
                "Путь вне разрешённых каталогов",
                path=str(path),
                allowed=[str(root) for root in self._allowed_roots],
            )
        if not path.exists():
            raise SourceError(f"Файл не найден: {path}", path=str(path))
        if not path.is_file():
            raise SourceError(f"Это не файл: {path}", path=str(path))
        return path

    async def describe(self, uri: str) -> SourceDescriptor:
        path = self._resolve(uri)
        info = await self._probe.probe(path)
        return SourceDescriptor(
            uri=uri,
            kind=SourceKind.LOCAL_FILE,
            title=path.stem,
            info=info,
            is_live=False,
            supports_seek=True,
        )

    async def windows(self, uri: str) -> AsyncIterator[MediaWindow]:
        """У файла ровно одно окно — на всю длительность.

        Для стрима здесь будет бесконечный генератор. Конвейеру разницы нет:
        он в обоих случаях получает последовательность окон.
        """
        descriptor = await self.describe(uri)
        assert descriptor.info is not None  # describe всегда заполняет info
        yield MediaWindow(
            source_id=uri,
            range=TimeRange(0.0, descriptor.info.duration_sec),
            index=0,
            is_final=True,
        )

    async def acquire(self, uri: str, window: TimeRange | None = None) -> LocalMedia:
        path = self._resolve(uri)
        info = await self._probe.probe(path)
        # owned=False: файл принадлежит пользователю, удалять его нельзя.
        return LocalMedia(path=path, info=info, source_uri=uri, owned=False)

    async def release(self, media: LocalMedia) -> None:
        if media.owned:  # для этого провайдера всегда False, но контракт соблюдаем
            media.path.unlink(missing_ok=True)
