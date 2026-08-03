"""Провайдер загруженных видео: ``upload://<video_id>``.

Файл лежит в нашем объектном хранилище. Провайдер отвечает за то, чтобы
он оказался на диске воркера, и за уборку после — если хранилище удалённое,
скачанный файл временный и его надо удалить.
"""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import ClassVar
from uuid import UUID

from core.domain.enums import SourceKind
from core.domain.errors import SourceError
from core.domain.media import LocalMedia, SourceDescriptor
from core.domain.values import MediaWindow, TimeRange
from core.ports.source import MediaProbe
from core.ports.storage import ObjectStorage
from core.registry import register_source

__all__ = ["KeyResolver", "UploadProvider", "parse_upload_uri"]

_SCHEME = "upload://"

#: «Дай ключ хранилища по идентификатору видео». Реализуется репозиторием
#: в проде и обычным словарём в тестах.
type KeyResolver = Callable[[UUID], Awaitable[str]]


def parse_upload_uri(uri: str) -> UUID:
    if not uri.startswith(_SCHEME):
        raise SourceError(f"Не upload-URI: {uri!r}", uri=uri)
    try:
        return UUID(uri.removeprefix(_SCHEME))
    except ValueError as error:
        raise SourceError(f"Некорректный video_id в URI: {uri!r}", uri=uri) from error


@register_source
class UploadProvider:
    """Реализация ``SourceProvider`` для видео из объектного хранилища.

    Провайдеру нужен ключ в хранилище, а в URI лежит только ``video_id``.
    Связь «video_id → storage_key» знает БД. Чтобы провайдер не зависел
    ни от репозиториев, ни от SQLAlchemy, эта связь передаётся функцией
    ``key_resolver`` — контейнер подставляет реализацию, читающую БД,
    а тест подставляет словарь.

    Вычислять ключ самостоятельно нельзя: он включает расширение исходника
    (``videos/<id>/source.mkv``), а его в URI нет.
    """

    scheme: ClassVar[str] = "upload"

    def __init__(
        self,
        storage: ObjectStorage,
        probe: MediaProbe,
        *,
        key_resolver: KeyResolver | None = None,
        workdir: Path | None = None,
    ) -> None:
        self._storage = storage
        self._probe = probe
        self._resolve_key = key_resolver
        self._workdir = workdir or Path(tempfile.gettempdir()) / "highlight-generator"

    async def _storage_key(self, uri: str, explicit: str | None) -> str:
        if explicit is not None:
            return explicit
        if self._resolve_key is None:
            raise SourceError(
                "UploadProvider не знает ключ хранилища: передайте storage_key "
                "или сконфигурируйте key_resolver",
                uri=uri,
            )
        return await self._resolve_key(parse_upload_uri(uri))

    @classmethod
    def supports(cls, uri: str) -> bool:
        return uri.startswith(_SCHEME)

    async def _fetch(self, uri: str, storage_key: str) -> tuple[Path, bool]:
        """Обеспечить наличие файла на диске.

        Возвращает путь и признак «файл наш, надо удалить». Если хранилище
        локальное и умеет отдать путь напрямую — копирования не происходит
        вовсе: копировать 20 ГБ из одного каталога в другой бессмысленно.
        """
        direct = getattr(self._storage, "local_path", None)
        if direct is not None:
            path = await direct(storage_key)
            if path is not None:
                return path, False

        self._workdir.mkdir(parents=True, exist_ok=True)
        suffix = Path(storage_key).suffix or ".bin"
        destination = self._workdir / f"{parse_upload_uri(uri)}{suffix}"
        await self._storage.get(storage_key, destination)
        return destination, True

    async def describe(self, uri: str, storage_key: str | None = None) -> SourceDescriptor:
        media = await self.acquire(uri, storage_key=storage_key)
        try:
            return SourceDescriptor(
                uri=uri,
                kind=SourceKind.UPLOAD,
                info=media.info,
                is_live=False,
                supports_seek=True,
            )
        finally:
            await self.release(media)

    async def windows(self, uri: str, storage_key: str | None = None) -> AsyncIterator[MediaWindow]:
        descriptor = await self.describe(uri, storage_key=storage_key)
        assert descriptor.info is not None
        yield MediaWindow(
            source_id=uri,
            range=TimeRange(0.0, descriptor.info.duration_sec),
            index=0,
            is_final=True,
        )

    async def acquire(
        self,
        uri: str,
        window: TimeRange | None = None,
        *,
        storage_key: str | None = None,
    ) -> LocalMedia:
        key = await self._storage_key(uri, storage_key)
        path, owned = await self._fetch(uri, key)
        info = await self._probe.probe(path)
        return LocalMedia(path=path, info=info, source_uri=uri, owned=owned)

    async def release(self, media: LocalMedia) -> None:
        # Удаляем только то, что скачали сами. Файл, отданный локальным
        # хранилищем напрямую, — это оригинал пользователя.
        if media.owned:
            media.path.unlink(missing_ok=True)
