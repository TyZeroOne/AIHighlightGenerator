"""ORM-модели (таблицы из док 03).

Это отражение схемы БД, а не домен. Домен — в ``core/domain/entities.py``,
и он ничего не знает об этих классах. Репозитории (Этап 2) переводят одно
в другое.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.domain.enums import (
    AssetKind,
    AssetOwnerType,
    ClipStatus,
    JobMode,
    JobStage,
    JobStatus,
    SourceKind,
    UserRole,
    VideoStatus,
)
from core.infra.db.base import Base, TimestampMixin

__all__ = [
    "AssetORM",
    "ClipORM",
    "JobORM",
    "SceneORM",
    "TranscriptSegmentORM",
    "UploadSessionORM",
    "UserORM",
    "VideoORM",
]


def _enum(enum_cls: type, name: str) -> SAEnum:
    """Нативный enum PostgreSQL из Python-перечисления.

    ``values_callable`` заставляет SQLAlchemy писать в БД *значения*
    (``"succeeded"``), а не *имена* (``"SUCCEEDED"``). Без этого в базе
    оказывается верхний регистр, и любой ручной SQL перестаёт совпадать.
    """
    return SAEnum(enum_cls, name=name, values_callable=lambda e: [m.value for m in e])


class UserORM(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    password_hash: Mapped[str | None] = mapped_column(Text)
    api_key_hash: Mapped[str | None] = mapped_column(String(128), index=True)
    role: Mapped[UserRole] = mapped_column(_enum(UserRole, "user_role"), default=UserRole.USER)
    quota_minutes_month: Mapped[int] = mapped_column(Integer, default=600)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    videos: Mapped[list[VideoORM]] = relationship(back_populates="user", lazy="raise")


class VideoORM(Base, TimestampMixin):
    __tablename__ = "videos"
    __table_args__ = (
        Index("ix_videos_user_created", "user_id", text("created_at DESC")),
        # Дедупликация: один и тот же файл у одного пользователя — одна запись.
        # Частичный индекс, чтобы удалённые видео не мешали повторной загрузке.
        Index(
            "uq_videos_user_checksum",
            "user_id",
            "checksum_sha256",
            unique=True,
            postgresql_where=text("status <> 'deleted'"),
        ),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    source_uri: Mapped[str] = mapped_column(Text, nullable=False)
    source_kind: Mapped[SourceKind] = mapped_column(_enum(SourceKind, "source_kind"))
    original_filename: Mapped[str | None] = mapped_column(Text)
    storage_key: Mapped[str] = mapped_column(Text, nullable=False)
    checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[VideoStatus] = mapped_column(
        _enum(VideoStatus, "video_status"), default=VideoStatus.UPLOADING
    )

    duration_sec: Mapped[Decimal | None] = mapped_column(Numeric(10, 3))
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    fps: Mapped[Decimal | None] = mapped_column(Numeric(7, 3))
    video_codec: Mapped[str | None] = mapped_column(String(32))
    audio_codec: Mapped[str | None] = mapped_column(String(32))
    container: Mapped[str | None] = mapped_column(String(16))
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    has_audio: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    probe: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    error: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    user: Mapped[UserORM] = relationship(back_populates="videos", lazy="raise")
    jobs: Mapped[list[JobORM]] = relationship(back_populates="video", lazy="raise")


class JobORM(Base, TimestampMixin):
    __tablename__ = "jobs"
    __table_args__ = (
        Index("ix_jobs_video_created", "video_id", text("created_at DESC")),
        Index(
            "ix_jobs_active",
            "status",
            postgresql_where=text("status IN ('queued', 'running')"),
        ),
        # Идемпотентность POST /analyze: тот же видос + те же параметры =
        # тот же успешный джоб, а не пересчёт.
        Index(
            "uq_jobs_video_params",
            "video_id",
            "params_hash",
            unique=True,
            postgresql_where=text("status = 'succeeded'"),
        ),
        CheckConstraint("progress >= 0 AND progress <= 100", name="progress_range"),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True, default=uuid4)
    video_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    mode: Mapped[JobMode] = mapped_column(_enum(JobMode, "job_mode"), default=JobMode.BATCH)
    status: Mapped[JobStatus] = mapped_column(
        _enum(JobStatus, "job_status"), default=JobStatus.QUEUED
    )
    stage: Mapped[JobStage | None] = mapped_column(_enum(JobStage, "job_stage"))
    progress: Mapped[Decimal] = mapped_column(Numeric(5, 2), default=Decimal(0))
    stage_states: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    params: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    params_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    celery_root_id: Mapped[str | None] = mapped_column(String(64))
    trace_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True))
    error: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    cost: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    video: Mapped[VideoORM] = relationship(back_populates="jobs", lazy="raise")


class SceneORM(Base):
    __tablename__ = "scenes"
    __table_args__ = (
        UniqueConstraint("job_id", "index", name="uq_scenes_job_index"),
        Index("ix_scenes_job_score", "job_id", text("final_score DESC NULLS LAST")),
        # GIN по JSONB — чтобы искать сцены по значению признака без
        # разворачивания features в отдельную таблицу (обоснование — док 03).
        Index("ix_scenes_features_gin", "features", postgresql_using="gin"),
        CheckConstraint("end_sec > start_sec", name="range_valid"),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True, default=uuid4)
    video_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False
    )
    job_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )
    index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_sec: Mapped[Decimal] = mapped_column(Numeric(10, 3), nullable=False)
    end_sec: Mapped[Decimal] = mapped_column(Numeric(10, 3), nullable=False)
    keyframe_key: Mapped[str | None] = mapped_column(Text)
    features: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    heuristic_score: Mapped[Decimal | None] = mapped_column(Numeric(6, 3))
    score_breakdown: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    llm_score: Mapped[Decimal | None] = mapped_column(Numeric(6, 3))
    llm_category: Mapped[str | None] = mapped_column(String(32))
    llm_reason: Mapped[str | None] = mapped_column(Text)
    final_score: Mapped[Decimal | None] = mapped_column(Numeric(6, 3))
    transcript_text: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )


class TranscriptSegmentORM(Base):
    __tablename__ = "transcript_segments"
    __table_args__ = (Index("ix_transcript_job_start", "job_id", "start_sec"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    video_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False
    )
    job_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )
    start_sec: Mapped[Decimal] = mapped_column(Numeric(10, 3), nullable=False)
    end_sec: Mapped[Decimal] = mapped_column(Numeric(10, 3), nullable=False)
    text_content: Mapped[str] = mapped_column("text", Text, nullable=False)
    words: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    language: Mapped[str | None] = mapped_column(String(5))
    avg_logprob: Mapped[Decimal | None] = mapped_column(Numeric(6, 4))
    no_speech_prob: Mapped[Decimal | None] = mapped_column(Numeric(6, 4))
    speaker: Mapped[str | None] = mapped_column(String(64))


class ClipORM(Base):
    __tablename__ = "clips"
    __table_args__ = (
        UniqueConstraint("job_id", "rank", name="uq_clips_job_rank"),
        CheckConstraint("end_sec > start_sec", name="range_valid"),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True, default=uuid4)
    job_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )
    video_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    start_sec: Mapped[Decimal] = mapped_column(Numeric(10, 3), nullable=False)
    end_sec: Mapped[Decimal] = mapped_column(Numeric(10, 3), nullable=False)
    score: Mapped[Decimal] = mapped_column(Numeric(6, 3), nullable=False)
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str] = mapped_column(Text, default="")
    title: Mapped[str | None] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)
    hashtags: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    status: Mapped[ClipStatus] = mapped_column(
        _enum(ClipStatus, "clip_status"), default=ClipStatus.PENDING
    )
    error: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )


class ClipSceneORM(Base):
    """Связка клип ↔ сцены: один клип может покрывать несколько сцен."""

    __tablename__ = "clip_scenes"

    clip_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("clips.id", ondelete="CASCADE"), primary_key=True
    )
    scene_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("scenes.id", ondelete="CASCADE"), primary_key=True
    )


class UploadSessionORM(Base, TimestampMixin):
    """Состояние возобновляемой чанковой загрузки.

    Почему в БД, а не в Redis: загрузка 20 ГБ идёт минутами и должна пережить
    перезапуск API и переезд клиента на другую реплику. Redis с TTL подошёл бы
    для короткоживущего состояния, но здесь состояние связано с файлами
    в хранилище — и потерять его означает оставить мусор, который никто
    не удалит.
    """

    __tablename__ = "upload_sessions"
    __table_args__ = (
        Index("ix_upload_sessions_user", "user_id", "completed"),
        Index(
            "ix_upload_sessions_expires",
            "expires_at",
            postgresql_where=text("completed = false"),
        ),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_size: Mapped[int] = mapped_column(Integer, nullable=False)
    total_size: Mapped[int | None] = mapped_column(BigInteger)
    #: Номера полученных частей. Массив, а не счётчик: части могут приходить
    #: не по порядку и повторно, и нам нужно знать, чего именно не хватает.
    received_parts: Mapped[list[int]] = mapped_column(
        ARRAY(Integer), nullable=False, server_default=text("'{}'")
    )
    received_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default="0")
    completed: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    video_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AssetORM(Base):
    """Все файловые артефакты в одной таблице.

    Альтернатива — колонки ``clip_path``, ``subtitle_path``, ``thumbnail_path``
    в ``clips``. Отвергнута: каждый новый формат требовал бы миграции,
    а метаданные файла (размер, mime, TTL) пришлось бы дублировать.
    """

    __tablename__ = "assets"
    __table_args__ = (
        Index("ix_assets_owner", "owner_type", "owner_id", "kind"),
        Index("ix_assets_expires", "expires_at", postgresql_where=text("expires_at IS NOT NULL")),
    )

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True, default=uuid4)
    owner_type: Mapped[AssetOwnerType] = mapped_column(_enum(AssetOwnerType, "asset_owner_type"))
    owner_id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), nullable=False)
    kind: Mapped[AssetKind] = mapped_column(_enum(AssetKind, "asset_kind"), nullable=False)
    storage_key: Mapped[str] = mapped_column(Text, nullable=False)
    mime: Mapped[str] = mapped_column(String(128), default="application/octet-stream")
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    checksum_sha256: Mapped[str | None] = mapped_column(String(64))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )
