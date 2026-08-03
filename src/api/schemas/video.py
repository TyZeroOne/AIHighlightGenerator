"""Схемы запросов и ответов для видео.

Отдельные классы, а не доменные сущности напрямую. Причина: доменная модель
меняется по внутренним причинам, а API — это публичный контракт, который
ломать нельзя. Разделив их, мы можем переименовать поле в домене, не сломав
ни одного клиента.
"""

from __future__ import annotations

from datetime import datetime
from typing import Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.domain.entities import UploadSession, Video

__all__ = [
    "PartAccepted",
    "RegisterLocalRequest",
    "UploadSessionResponse",
    "VideoListResponse",
    "VideoResponse",
]


class VideoResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"status": "ready"}]})

    video_id: UUID
    status: str
    source_kind: str
    filename: str | None = None
    size_bytes: int | None = None
    duration_sec: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    video_codec: str | None = None
    audio_codec: str | None = None
    container: str | None = None
    has_audio: bool = False
    checksum_sha256: str
    created_at: datetime | None = None
    #: `true`, если файл уже был загружен раньше и вернулась существующая запись.
    duplicate: bool = False

    @classmethod
    def from_domain(cls, video: Video, *, duplicate: bool = False) -> Self:
        info = video.info
        return cls(
            video_id=video.id,
            status=video.status.value,
            source_kind=video.source_kind.value,
            filename=video.original_filename,
            size_bytes=video.size_bytes or (info.size_bytes if info else None),
            duration_sec=round(info.duration_sec, 3) if info else None,
            width=info.width if info else None,
            height=info.height if info else None,
            fps=round(info.fps, 3) if info and info.fps else None,
            video_codec=info.video_codec if info else None,
            audio_codec=info.audio_codec if info else None,
            container=info.container if info else None,
            has_audio=info.has_audio if info else False,
            checksum_sha256=video.checksum_sha256,
            created_at=video.created_at,
            duplicate=duplicate,
        )


class VideoListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[VideoResponse]


class RegisterLocalRequest(BaseModel):
    source_uri: str = Field(
        description="URI файла на смонтированном томе",
        examples=["file:///media/in/movie.mkv"],
    )

    @field_validator("source_uri")
    @classmethod
    def _must_be_file_uri(cls, value: str) -> str:
        # Ранняя проверка схемы: понятная ошибка валидации вместо
        # «провайдер не найден» из глубины реестра.
        if not value.startswith("file://"):
            raise ValueError("Поддерживается только схема file://")
        return value


class UploadSessionResponse(BaseModel):
    upload_id: UUID
    chunk_size: int
    total_size: int | None = None
    received_parts: list[int]
    received_bytes: int
    #: Номер первой недостающей части — по нему клиент возобновляет загрузку
    #: после обрыва, не пересылая то, что уже дошло.
    next_expected_part: int
    completed: bool

    @classmethod
    def from_domain(cls, session: UploadSession) -> Self:
        return cls(
            upload_id=session.id,
            chunk_size=session.chunk_size,
            total_size=session.total_size,
            received_parts=list(session.received_parts),
            received_bytes=session.received_bytes,
            next_expected_part=session.next_expected_part(),
            completed=session.completed,
        )


class PartAccepted(BaseModel):
    upload_id: UUID
    part_number: int
    received_parts: int
    received_bytes: int
    next_expected_part: int
