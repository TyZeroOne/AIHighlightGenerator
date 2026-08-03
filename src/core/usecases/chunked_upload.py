"""Возобновляемая загрузка большими частями.

**Зачем это нужно.** Обычный multipart на 20 ГБ — плохая идея: обрыв связи
на 95 % означает начать сначала, прокси и балансировщики режут длинные
запросы по таймауту, а прогресс загрузки клиенту неоткуда взять.

**Как устроено.** Клиент делит файл на части, шлёт каждую отдельным запросом
и в конце просит собрать. Части складываются в хранилище по отдельным ключам
и склеиваются на финальном шаге.

Части могут приходить **не по порядку и повторно** — это не ошибка, а норма
при параллельной отправке и ретраях. Отсюда массив полученных номеров в БД
вместо счётчика.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from uuid import UUID

from core.domain.entities import UploadSession, Video
from core.domain.errors import ConflictError, NotFoundError, ValidationError
from core.domain.keys import upload_part_key, upload_prefix
from core.domain.validation import validate_upload_name
from core.logging import get_logger
from core.ports.repository import UnitOfWorkFactory
from core.ports.storage import ObjectStorage
from core.usecases.upload_video import StoredUpload, UploadVideo, stream_to_file

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "MAX_PARTS",
    "CompleteChunkedUpload",
    "InitChunkedUpload",
    "UploadPart",
]

log = get_logger(__name__)

#: 32 МиБ — компромисс: части достаточно крупные, чтобы накладные расходы
#: на запрос были незаметны, и достаточно мелкие, чтобы потеря части при
#: обрыве стоила секунды, а не минуты.
DEFAULT_CHUNK_SIZE: Final = 32 * 1024 * 1024

#: Верхняя граница числа частей. При 20 ГиБ и частях по 32 МиБ выходит ~640,
#: так что 10 000 — с большим запасом. Ограничение нужно, чтобы клиент
#: не мог создать миллион объектов в хранилище кусками по байту.
MAX_PARTS: Final = 10_000


@dataclass(frozen=True, slots=True)
class ChunkedUploadLimits:
    max_bytes: int
    allowed_containers: frozenset[str]
    chunk_size: int = DEFAULT_CHUNK_SIZE
    ttl_hours: int = 24


class InitChunkedUpload:
    """Шаг 1: завести сессию загрузки."""

    def __init__(
        self,
        *,
        uow_factory: UnitOfWorkFactory,
        limits: ChunkedUploadLimits,
    ) -> None:
        self._uow_factory = uow_factory
        self._limits = limits

    async def execute(
        self,
        *,
        user_id: UUID,
        filename: str,
        total_size: int | None = None,
    ) -> UploadSession:
        # Проверяем имя сразу: сообщить «формат не тот» до отправки 20 ГБ
        # дешевле для всех, чем после.
        validate_upload_name(filename, self._limits.allowed_containers)

        if total_size is not None:
            if total_size <= 0:
                raise ValidationError("total_size должен быть положительным")
            if total_size > self._limits.max_bytes:
                raise ValidationError(
                    f"Заявленный размер превышает лимит {self._limits.max_bytes / 1024**3:.2f} ГиБ",
                    total_size=total_size,
                )

        async with self._uow_factory() as uow:
            session = await uow.uploads.create(
                user_id=user_id,
                filename=filename,
                chunk_size=self._limits.chunk_size,
                total_size=total_size,
                ttl_hours=self._limits.ttl_hours,
            )
            await uow.commit()

        log.info("upload.session_created", upload_id=str(session.id), filename=filename)
        return session


class UploadPart:
    """Шаг 2: принять одну часть."""

    def __init__(
        self,
        *,
        uow_factory: UnitOfWorkFactory,
        storage: ObjectStorage,
        limits: ChunkedUploadLimits,
        workdir: Path,
    ) -> None:
        self._uow_factory = uow_factory
        self._storage = storage
        self._limits = limits
        self._workdir = workdir

    async def execute(
        self,
        *,
        user_id: UUID,
        upload_id: UUID,
        part_number: int,
        chunks: AsyncIterator[bytes],
    ) -> UploadSession:
        if not 1 <= part_number <= MAX_PARTS:
            raise ValidationError(
                f"Номер части должен быть в диапазоне 1..{MAX_PARTS}",
                part_number=part_number,
            )

        async with self._uow_factory() as uow:
            session = await uow.uploads.get_owned(upload_id, user_id)
        if session is None:
            raise NotFoundError("Сессия загрузки не найдена", upload_id=str(upload_id))
        if session.completed:
            raise ConflictError("Загрузка уже завершена", upload_id=str(upload_id))

        # Лимит проверяется по уже принятому объёму: клиент, шлющий части
        # бесконечно, упрётся в него, даже если total_size не объявлял.
        remaining = self._limits.max_bytes - session.received_bytes
        if remaining <= 0:
            raise ValidationError("Превышен лимит размера загрузки", upload_id=str(upload_id))

        temp_path = self._workdir / str(upload_id) / f"{part_number:06d}.part"
        received = await stream_to_file(chunks, temp_path, max_bytes=remaining)

        try:
            await self._storage.put(upload_part_key(upload_id, part_number), received.path)
        finally:
            received.path.unlink(missing_ok=True)

        async with self._uow_factory() as uow:
            updated = await uow.uploads.register_part(upload_id, part_number, received.size_bytes)
            await uow.commit()

        log.debug(
            "upload.part_received",
            upload_id=str(upload_id),
            part=part_number,
            size=received.size_bytes,
        )
        return updated


class CompleteChunkedUpload:
    """Шаг 3: склеить части и передать результат обычному сценарию загрузки.

    Финальный шаг намеренно **не дублирует** логику проверки и дедупликации —
    он собирает файл и вызывает ``UploadVideo``. Иначе появились бы два пути
    появления видео в системе с расходящимся поведением, и однажды один
    из них забыли бы обновить.
    """

    def __init__(
        self,
        *,
        uow_factory: UnitOfWorkFactory,
        storage: ObjectStorage,
        upload_video: UploadVideo,
        workdir: Path,
    ) -> None:
        self._uow_factory = uow_factory
        self._storage = storage
        self._upload_video = upload_video
        self._workdir = workdir

    async def execute(self, *, user_id: UUID, upload_id: UUID) -> tuple[Video, bool]:
        async with self._uow_factory() as uow:
            session = await uow.uploads.get_owned(upload_id, user_id)
        if session is None:
            raise NotFoundError("Сессия загрузки не найдена", upload_id=str(upload_id))
        if session.completed:
            raise ConflictError("Загрузка уже завершена", upload_id=str(upload_id))
        if not session.received_parts:
            raise ValidationError("Не получено ни одной части", upload_id=str(upload_id))

        missing = _missing_parts(session.received_parts)
        if missing:
            raise ConflictError(
                f"Не хватает частей: {missing[:10]}",
                upload_id=str(upload_id),
                missing=missing[:10],
                received=len(session.received_parts),
            )

        assembled = await self._assemble(upload_id, session)
        try:
            video, duplicate = await self._upload_video.execute(
                user_id=user_id,
                upload=assembled,
                filename=session.filename,
            )
        finally:
            assembled.path.unlink(missing_ok=True)
            await self._cleanup_parts(upload_id)

        async with self._uow_factory() as uow:
            await uow.uploads.complete(upload_id, video.id)
            await uow.commit()

        log.info("upload.completed", upload_id=str(upload_id), video_id=str(video.id))
        return video, duplicate

    async def _assemble(self, upload_id: UUID, session: UploadSession) -> StoredUpload:
        """Склеить части в один файл, попутно считая хеш."""

        async def _all_parts() -> AsyncIterator[bytes]:
            # Строго по возрастанию номера: порядок отправки значения не имеет,
            # порядок склейки — имеет.
            for part_number in sorted(session.received_parts):
                key = upload_part_key(upload_id, part_number)
                async for chunk in self._storage.open_stream(key):
                    yield chunk

        target = self._workdir / str(upload_id) / "assembled"
        # max_bytes здесь пересчитывать не нужно: суммарный объём уже
        # проверялся при приёме каждой части.
        return await stream_to_file(_all_parts(), target, max_bytes=session.received_bytes + 1)

    async def _cleanup_parts(self, upload_id: UUID) -> None:
        delete_prefix = getattr(self._storage, "delete_prefix", None)
        try:
            if delete_prefix is not None:
                await delete_prefix(upload_prefix(upload_id))
        except Exception as error:
            log.warning("upload.cleanup_failed", upload_id=str(upload_id), error=str(error))


def _missing_parts(received: tuple[int, ...]) -> list[int]:
    """Дыры в нумерации частей.

    Части нумеруются с 1 и должны идти сплошняком: пропуск означает, что
    кусок файла потерялся, и склеивать нельзя — получится битое видео,
    которое упадёт уже на этапе анализа, далеко от места настоящей причины.
    """
    if not received:
        return []
    expected = set(range(1, max(received) + 1))
    return sorted(expected - set(received))
