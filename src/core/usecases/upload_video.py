"""Сценарии работы с видео: загрузка, регистрация, список, удаление.

Здесь — вся логика этапа. Роутер её не содержит: он разбирает HTTP-запрос
и вызывает сценарий. Воркер, CLI или будущий gRPC-интерфейс вызовут те же
сценарии, ничего не дублируя.

Зависимости — только домен и порты. Ни FastAPI, ни SQLAlchemy, ни ffprobe,
ни S3 здесь не упоминаются: их подставляет композиционный корень.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final
from uuid import UUID, uuid4

from core.domain.entities import Video
from core.domain.enums import AssetKind, AssetOwnerType, SourceKind, VideoStatus
from core.domain.errors import NotFoundError, ValidationError
from core.domain.keys import source_key, video_prefix
from core.domain.validation import validate_media
from core.logging import get_logger
from core.ports.repository import UnitOfWorkFactory
from core.ports.source import MediaProbe, SourceProvider
from core.ports.storage import ObjectStorage

__all__ = [
    "DeleteVideo",
    "GetVideo",
    "ListVideos",
    "RegisterLocalVideo",
    "StoredUpload",
    "UploadVideo",
    "VideoLimits",
    "hash_file",
    "stream_to_file",
]

log = get_logger(__name__)

HASH_CHUNK_SIZE: Final = 1 << 20  # 1 МиБ


@dataclass(frozen=True, slots=True)
class VideoLimits:
    allowed_containers: frozenset[str]
    max_duration_sec: int
    max_bytes: int


@dataclass(frozen=True, slots=True)
class StoredUpload:
    """Результат приёма файла на диск — до того, как он попал в хранилище."""

    path: Path
    size_bytes: int
    checksum_sha256: str


async def stream_to_file(
    chunks: AsyncIterator[bytes],
    destination: Path,
    *,
    max_bytes: int,
) -> StoredUpload:
    """Принять поток во временный файл, считая хеш и контролируя размер.

    Три вещи делаются за один проход по данным:

    1. **Запись на диск.** Держать 20 ГБ в памяти невозможно.
    2. **Подсчёт SHA-256.** Второй проход по 20 ГБ ради хеша — это лишние
       минуты дискового чтения.
    3. **Контроль лимита.** Проверка обрывает приём в момент превышения,
       а не после того, как файл целиком занял место. Именно поэтому нельзя
       полагаться только на заголовок ``Content-Length``: его присылает клиент,
       и он может соврать.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0

    try:
        with destination.open("wb") as handle:
            async for chunk in chunks:
                size += len(chunk)
                if size > max_bytes:
                    raise ValidationError(
                        f"Файл превышает лимит {max_bytes / 1024**3:.2f} ГиБ",
                        max_bytes=max_bytes,
                    )
                digest.update(chunk)
                handle.write(chunk)
    except BaseException:
        # Недописанный файл обязателен к удалению: иначе оборванные загрузки
        # молча съедят диск, и никто не поймёт, откуда взялись эти гигабайты.
        destination.unlink(missing_ok=True)
        raise

    if size == 0:
        destination.unlink(missing_ok=True)
        raise ValidationError("Загружен пустой файл")

    return StoredUpload(path=destination, size_bytes=size, checksum_sha256=digest.hexdigest())


async def hash_file(path: Path) -> str:
    """SHA-256 файла. Считается в пуле потоков — 20 ГБ блокировали бы event loop."""

    def _compute() -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(HASH_CHUNK_SIZE):
                digest.update(chunk)
        return digest.hexdigest()

    return await asyncio.to_thread(_compute)


class UploadVideo:
    """Приём загруженного файла: проверка, дедупликация, сохранение."""

    def __init__(
        self,
        *,
        uow_factory: UnitOfWorkFactory,
        storage: ObjectStorage,
        probe: MediaProbe,
        limits: VideoLimits,
    ) -> None:
        self._uow_factory = uow_factory
        self._storage = storage
        self._probe = probe
        self._limits = limits

    async def execute(
        self,
        *,
        user_id: UUID,
        upload: StoredUpload,
        filename: str,
    ) -> tuple[Video, bool]:
        """Вернуть видео и признак «это дубликат уже загруженного».

        Порядок шагов не случаен: сначала **проверка содержимого**, потом
        дедупликация, и только потом запись в хранилище. Класть в хранилище
        файл, который окажется не видео, — значит платить за место и потом
        его чистить.
        """
        info = await self._probe.probe(upload.path)
        validate_media(
            info,
            allowed_containers=self._limits.allowed_containers,
            max_duration_sec=self._limits.max_duration_sec,
            max_bytes=self._limits.max_bytes,
            filename=filename,
        )

        safe_name = PurePosixPath(filename.replace("\\", "/")).name
        extension = PurePosixPath(safe_name).suffix.lstrip(".").lower() or "mp4"

        async with self._uow_factory() as uow:
            existing = await uow.videos.get_by_checksum(user_id, upload.checksum_sha256)
            if existing is not None:
                log.info(
                    "upload.duplicate",
                    video_id=str(existing.id),
                    checksum=upload.checksum_sha256[:12],
                )
                return existing, True

            video_id = uuid4()
            key = source_key(video_id, extension)
            mime = mime_for(extension)

            stored = await uow.videos.add(
                Video(
                    id=video_id,
                    user_id=user_id,
                    source_uri=f"upload://{video_id}",
                    source_kind=SourceKind.UPLOAD,
                    status=VideoStatus.READY,
                    storage_key=key,
                    checksum_sha256=upload.checksum_sha256,
                    info=info,
                    original_filename=safe_name,
                    size_bytes=upload.size_bytes,
                )
            )

            # Запись в хранилище внутри транзакции: если положить файл до неё
            # и транзакция откатится, останется осиротевший объект. Если после
            # коммита — можно получить запись без файла, а это состояние,
            # из которого пользователь не выберется. Осиротевший объект
            # безопаснее: его подберёт уборка.
            await self._storage.put(key, upload.path, content_type=mime)

            await uow.assets.add(
                owner_type=AssetOwnerType.VIDEO,
                owner_id=video_id,
                kind=AssetKind.SOURCE,
                storage_key=key,
                mime=mime,
                size_bytes=upload.size_bytes,
                checksum=upload.checksum_sha256,
                meta={"original_filename": safe_name},
            )
            await uow.commit()

        log.info(
            "upload.stored",
            video_id=str(stored.id),
            size_bytes=upload.size_bytes,
            duration_sec=round(info.duration_sec, 2),
        )
        return stored, False


class RegisterLocalVideo:
    """Регистрация файла, уже лежащего на смонтированном томе.

    Байты не копируются: для архива на 2 ТБ копирование в наше хранилище было бы
    бессмысленной тратой места. В БД пишется ``file://``-URI, а провайдер
    читает файл на месте.

    Провайдер передаётся снаружи — сценарий не знает, что за ним
    ``LocalFileProvider``. Завтра там окажется провайдер сетевой папки,
    и этот файл не изменится.
    """

    def __init__(
        self,
        *,
        uow_factory: UnitOfWorkFactory,
        provider: SourceProvider,
        limits: VideoLimits,
    ) -> None:
        self._uow_factory = uow_factory
        self._provider = provider
        self._limits = limits

    async def execute(self, *, user_id: UUID, source_uri: str) -> tuple[Video, bool]:
        if not self._provider.supports(source_uri):
            raise ValidationError(
                f"URI не поддерживается провайдером: {source_uri!r}", uri=source_uri
            )

        media = await self._provider.acquire(source_uri)
        try:
            validate_media(
                media.info,
                allowed_containers=self._limits.allowed_containers,
                max_duration_sec=self._limits.max_duration_sec,
                max_bytes=self._limits.max_bytes,
                filename=media.path.name,
            )
            checksum = await hash_file(media.path)
            filename = media.path.name
            info = media.info
        finally:
            await self._provider.release(media)

        async with self._uow_factory() as uow:
            existing = await uow.videos.get_by_checksum(user_id, checksum)
            if existing is not None:
                return existing, True

            stored = await uow.videos.add(
                Video(
                    id=uuid4(),
                    user_id=user_id,
                    source_uri=source_uri,
                    source_kind=SourceKind.LOCAL_FILE,
                    status=VideoStatus.READY,
                    # Ключа в нашем хранилище нет — файл не наш. Пустая строка
                    # честнее выдуманного ключа, который никуда не ведёт.
                    storage_key="",
                    checksum_sha256=checksum,
                    info=info,
                    original_filename=filename,
                    size_bytes=info.size_bytes,
                )
            )
            await uow.commit()

        log.info("video.registered", video_id=str(stored.id), source_uri=source_uri)
        return stored, False


class GetVideo:
    def __init__(self, *, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def execute(self, *, user_id: UUID, video_id: UUID) -> Video:
        async with self._uow_factory() as uow:
            video = await uow.videos.get_owned(video_id, user_id)
        if video is None:
            # 404, а не 403: сообщать «объект существует, но не ваш» —
            # значит подтверждать его существование постороннему.
            raise NotFoundError("Видео не найдено", video_id=str(video_id))
        return video


class ListVideos:
    def __init__(self, *, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def execute(
        self,
        *,
        user_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[Sequence[Video], int]:
        async with self._uow_factory() as uow:
            items = await uow.videos.list_for_user(user_id, limit=limit, offset=offset)
            total = await uow.videos.count_for_user(user_id)
        return items, total


class DeleteVideo:
    """Мягкое удаление записи и настоящее удаление файлов."""

    def __init__(self, *, uow_factory: UnitOfWorkFactory, storage: ObjectStorage) -> None:
        self._uow_factory = uow_factory
        self._storage = storage

    async def execute(self, *, user_id: UUID, video_id: UUID) -> None:
        async with self._uow_factory() as uow:
            video = await uow.videos.get_owned(video_id, user_id)
            if video is None:
                raise NotFoundError("Видео не найдено", video_id=str(video_id))

            await uow.videos.mark_deleted(video_id)
            await uow.assets.delete_for_owner(AssetOwnerType.VIDEO, video_id)
            await uow.commit()

        # Файлы удаляем ПОСЛЕ фиксации транзакции. Если наоборот и транзакция
        # откатится — в БД останется живое видео без файлов, состояние
        # безнадёжное. При текущем порядке худший случай — осиротевшие файлы,
        # которые подберёт периодическая уборка.
        if video.source_kind is SourceKind.UPLOAD:
            await self._delete_files(video)

    async def _delete_files(self, video: Video) -> None:
        # delete_prefix есть у обеих наших реализаций, но в порт не вынесен:
        # это оптимизация, а не часть контракта. Отсюда getattr с фоллбеком.
        delete_prefix = getattr(self._storage, "delete_prefix", None)
        try:
            if delete_prefix is not None:
                await delete_prefix(video_prefix(video.id))
            elif video.storage_key:
                await self._storage.delete(video.storage_key)
        except Exception as error:
            # Неудача уборки не должна ломать удаление для пользователя:
            # запись уже помечена удалённой, файлы подберёт фоновая задача.
            log.warning("video.file_cleanup_failed", video_id=str(video.id), error=str(error))


_MIME_BY_EXTENSION: Final[dict[str, str]] = {
    "mp4": "video/mp4",
    "m4v": "video/mp4",
    "mov": "video/quicktime",
    "mkv": "video/x-matroska",
    "webm": "video/webm",
}


def mime_for(extension: str) -> str:
    return _MIME_BY_EXTENSION.get(extension.lstrip(".").lower(), "application/octet-stream")
