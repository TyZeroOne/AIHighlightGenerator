"""Схемы клипов и их файлов.

Клип в этом ответе — это уже не «отрезок, который стоит вырезать»
(так он выглядел на Этапе 7), а результат: файлы, их размеры и статус
сборки. Поэтому схема отдельная от ``scenes.py``, где клип показывался
рядом со сценами как объяснение отбора.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from uuid import UUID

from pydantic import BaseModel, Field

from core.domain.entities import Clip
from core.domain.enums import AssetKind
from core.domain.timecode import format_timecode
from core.ports.repository import AssetRecord

__all__ = ["AssetResponse", "ClipItemResponse", "ClipListResponse"]


class AssetResponse(BaseModel):
    asset_id: UUID
    mime: str
    size_bytes: int
    #: Путь скачивания. Отдаём готовым, чтобы клиенту не приходилось
    #: собирать URL из идентификатора и знать нашу схему маршрутов.
    download_url: str

    @classmethod
    def from_record(cls, record: AssetRecord) -> AssetResponse:
        return cls(
            asset_id=record.id,
            mime=record.mime,
            size_bytes=record.size_bytes,
            download_url=f"/api/v1/download/{record.id}",
        )


class ClipItemResponse(BaseModel):
    clip_id: UUID
    rank: int
    start: str
    end: str
    start_sec: float
    end_sec: float
    duration_sec: float
    score: float
    category: str
    reason: str
    #: ``pending`` — отрезок выбран, файлов ещё нет; ``ready`` — собран;
    #: ``failed`` — не собрался, причина в ``error``.
    status: str
    error: str | None = None
    title: str | None = None
    description: str | None = None
    hashtags: Sequence[str] = ()
    scene_ids: Sequence[UUID] = ()
    #: Файлы по видам: ``clip_h``, ``clip_v``, ``thumbnail`` и далее
    #: субтитры с Этапа 9. Словарь, а не список: клиент почти всегда
    #: спрашивает «дай вертикальный», а не «покажи все».
    assets: Mapping[str, AssetResponse] = Field(default_factory=dict)

    @classmethod
    def from_domain(cls, clip: Clip, assets: Sequence[AssetRecord] = ()) -> ClipItemResponse:
        metadata = clip.metadata
        return cls(
            clip_id=clip.id,
            rank=clip.rank,
            start=format_timecode(clip.range.start),
            end=format_timecode(clip.range.end),
            start_sec=round(clip.range.start, 3),
            end_sec=round(clip.range.end, 3),
            duration_sec=round(clip.range.duration, 3),
            score=clip.score,
            category=clip.category,
            reason=clip.reason,
            status=clip.status.value,
            error=clip.error,
            title=metadata.title if metadata else None,
            description=metadata.description if metadata else None,
            hashtags=list(metadata.hashtags) if metadata else [],
            scene_ids=list(clip.scene_ids),
            assets={
                _kind_name(record.kind): AssetResponse.from_record(record) for record in assets
            },
        )


class ClipListResponse(BaseModel):
    job_id: UUID
    #: Статус джоба целиком. Незавершённый джоб — это 200 с пустым
    #: списком, а не 404: джоб существует, результата пока нет.
    status: str
    total: int
    items: Sequence[ClipItemResponse] = ()


def _kind_name(kind: AssetKind) -> str:
    return kind.value
