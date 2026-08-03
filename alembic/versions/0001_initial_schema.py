"""Начальная схема: users, videos, jobs, scenes, transcript_segments, clips, assets

Revision ID: 0001
Revises:
Create Date: Этап 1

Миграция написана вручную, а не автогенерацией. Причина: автогенерация
не умеет корректно создавать нативные enum-типы PostgreSQL до таблиц,
которые на них ссылаются, и путает частичные индексы (`postgresql_where`).
Дальше по проекту автогенерация используется, но её вывод всегда
просматривается глазами.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Значения enum'ов дублируются здесь намеренно: миграция должна быть
# самодостаточной. Если завтра в core.domain.enums добавится значение,
# ЭТА миграция обязана остаться такой, какой была, — иначе повторный
# прогон на чистой базе даст схему, отличную от продовой.
_ENUMS: dict[str, tuple[str, ...]] = {
    "user_role": ("user", "admin"),
    "source_kind": ("upload", "local_file", "youtube", "twitch", "kick"),
    "video_status": ("uploading", "ready", "failed", "deleted"),
    "job_mode": ("batch", "streaming"),
    "job_status": ("queued", "running", "partial", "succeeded", "failed", "cancelled"),
    "job_stage": (
        "ingest",
        "detect_scenes",
        "extract_audio",
        "transcribe",
        "analyze_audio",
        "analyze_speech",
        "analyze_vision",
        "score_scenes",
        "llm_rank",
        "select_highlights",
        "render_clips",
        "generate_metadata",
        "finalize",
    ),
    "clip_status": ("pending", "rendering", "ready", "failed"),
    "asset_owner_type": ("video", "job", "clip"),
    "asset_kind": (
        "source",
        "audio_wav",
        "keyframe",
        "clip_h",
        "clip_v",
        "subtitle_srt",
        "subtitle_ass",
        "thumbnail",
        "manifest",
    ),
}


def _enum(name: str) -> postgresql.ENUM:
    """Ссылка на уже созданный тип: create_type=False, иначе PG ругнётся дублем."""
    return postgresql.ENUM(*_ENUMS[name], name=name, create_type=False)


def upgrade() -> None:
    bind = op.get_bind()
    for name, values in _ENUMS.items():
        postgresql.ENUM(*values, name=name).create(bind, checkfirst=True)

    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("password_hash", sa.Text()),
        sa.Column("api_key_hash", sa.String(128)),
        sa.Column("role", _enum("user_role"), nullable=False, server_default="user"),
        sa.Column("quota_minutes_month", sa.Integer(), nullable=False, server_default="600"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.create_index("ix_users_api_key_hash", "users", ["api_key_hash"])

    op.create_table(
        "videos",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_uri", sa.Text(), nullable=False),
        sa.Column("source_kind", _enum("source_kind"), nullable=False),
        sa.Column("original_filename", sa.Text()),
        sa.Column("storage_key", sa.Text(), nullable=False),
        sa.Column("checksum_sha256", sa.String(64), nullable=False),
        sa.Column("status", _enum("video_status"), nullable=False, server_default="uploading"),
        sa.Column("duration_sec", sa.Numeric(10, 3)),
        sa.Column("width", sa.Integer()),
        sa.Column("height", sa.Integer()),
        sa.Column("fps", sa.Numeric(7, 3)),
        sa.Column("video_codec", sa.String(32)),
        sa.Column("audio_codec", sa.String(32)),
        sa.Column("container", sa.String(16)),
        sa.Column("size_bytes", sa.BigInteger()),
        sa.Column("has_audio", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("probe", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("error", postgresql.JSONB()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], ondelete="CASCADE", name="fk_videos_user_id_users"
        ),
    )
    op.create_index("ix_videos_user_created", "videos", ["user_id", sa.text("created_at DESC")])
    op.create_index(
        "uq_videos_user_checksum",
        "videos",
        ["user_id", "checksum_sha256"],
        unique=True,
        postgresql_where=sa.text("status <> 'deleted'"),
    )

    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("video_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("mode", _enum("job_mode"), nullable=False, server_default="batch"),
        sa.Column("status", _enum("job_status"), nullable=False, server_default="queued"),
        sa.Column("stage", _enum("job_stage")),
        sa.Column("progress", sa.Numeric(5, 2), nullable=False, server_default="0"),
        sa.Column("stage_states", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("params", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("params_hash", sa.String(64), nullable=False),
        sa.Column("celery_root_id", sa.String(64)),
        sa.Column("trace_id", postgresql.UUID(as_uuid=True)),
        sa.Column("error", postgresql.JSONB()),
        sa.Column("cost", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("started_at", sa.DateTime()),
        sa.Column("finished_at", sa.DateTime()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(
            ["video_id"], ["videos.id"], ondelete="CASCADE", name="fk_jobs_video_id_videos"
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], ondelete="CASCADE", name="fk_jobs_user_id_users"
        ),
        sa.CheckConstraint("progress >= 0 AND progress <= 100", name="ck_jobs_progress_range"),
    )
    op.create_index("ix_jobs_video_created", "jobs", ["video_id", sa.text("created_at DESC")])
    op.create_index(
        "ix_jobs_active",
        "jobs",
        ["status"],
        postgresql_where=sa.text("status IN ('queued', 'running')"),
    )
    op.create_index(
        "uq_jobs_video_params",
        "jobs",
        ["video_id", "params_hash"],
        unique=True,
        postgresql_where=sa.text("status = 'succeeded'"),
    )

    op.create_table(
        "scenes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("video_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("index", sa.Integer(), nullable=False),
        sa.Column("start_sec", sa.Numeric(10, 3), nullable=False),
        sa.Column("end_sec", sa.Numeric(10, 3), nullable=False),
        sa.Column("keyframe_key", sa.Text()),
        sa.Column("features", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("heuristic_score", sa.Numeric(6, 3)),
        sa.Column("score_breakdown", postgresql.JSONB()),
        sa.Column("llm_score", sa.Numeric(6, 3)),
        sa.Column("llm_category", sa.String(32)),
        sa.Column("llm_reason", sa.Text()),
        sa.Column("final_score", sa.Numeric(6, 3)),
        sa.Column("transcript_text", sa.Text()),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(
            ["video_id"], ["videos.id"], ondelete="CASCADE", name="fk_scenes_video_id_videos"
        ),
        sa.ForeignKeyConstraint(
            ["job_id"], ["jobs.id"], ondelete="CASCADE", name="fk_scenes_job_id_jobs"
        ),
        sa.UniqueConstraint("job_id", "index", name="uq_scenes_job_index"),
        sa.CheckConstraint("end_sec > start_sec", name="ck_scenes_range_valid"),
    )
    op.create_index(
        "ix_scenes_job_score", "scenes", ["job_id", sa.text("final_score DESC NULLS LAST")]
    )
    op.create_index("ix_scenes_features_gin", "scenes", ["features"], postgresql_using="gin")

    op.create_table(
        "transcript_segments",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("video_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("start_sec", sa.Numeric(10, 3), nullable=False),
        sa.Column("end_sec", sa.Numeric(10, 3), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("words", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("language", sa.String(5)),
        sa.Column("avg_logprob", sa.Numeric(6, 4)),
        sa.Column("no_speech_prob", sa.Numeric(6, 4)),
        sa.Column("speaker", sa.String(64)),
        sa.ForeignKeyConstraint(
            ["video_id"],
            ["videos.id"],
            ondelete="CASCADE",
            name="fk_transcript_segments_video_id_videos",
        ),
        sa.ForeignKeyConstraint(
            ["job_id"], ["jobs.id"], ondelete="CASCADE", name="fk_transcript_segments_job_id_jobs"
        ),
    )
    op.create_index("ix_transcript_job_start", "transcript_segments", ["job_id", "start_sec"])

    op.create_table(
        "clips",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("video_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("start_sec", sa.Numeric(10, 3), nullable=False),
        sa.Column("end_sec", sa.Numeric(10, 3), nullable=False),
        sa.Column("score", sa.Numeric(6, 3), nullable=False),
        sa.Column("category", sa.String(32), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False, server_default=""),
        sa.Column("title", sa.String(200)),
        sa.Column("description", sa.Text()),
        sa.Column("hashtags", postgresql.ARRAY(sa.Text())),
        sa.Column("status", _enum("clip_status"), nullable=False, server_default="pending"),
        sa.Column("error", postgresql.JSONB()),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(
            ["job_id"], ["jobs.id"], ondelete="CASCADE", name="fk_clips_job_id_jobs"
        ),
        sa.ForeignKeyConstraint(
            ["video_id"], ["videos.id"], ondelete="CASCADE", name="fk_clips_video_id_videos"
        ),
        sa.UniqueConstraint("job_id", "rank", name="uq_clips_job_rank"),
        sa.CheckConstraint("end_sec > start_sec", name="ck_clips_range_valid"),
    )

    op.create_table(
        "clip_scenes",
        sa.Column("clip_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("scene_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.ForeignKeyConstraint(
            ["clip_id"], ["clips.id"], ondelete="CASCADE", name="fk_clip_scenes_clip_id_clips"
        ),
        sa.ForeignKeyConstraint(
            ["scene_id"], ["scenes.id"], ondelete="CASCADE", name="fk_clip_scenes_scene_id_scenes"
        ),
    )

    op.create_table(
        "assets",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("owner_type", _enum("asset_owner_type"), nullable=False),
        sa.Column("owner_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("kind", _enum("asset_kind"), nullable=False),
        sa.Column("storage_key", sa.Text(), nullable=False),
        sa.Column(
            "mime", sa.String(128), nullable=False, server_default="application/octet-stream"
        ),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("meta", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("checksum_sha256", sa.String(64)),
        sa.Column("expires_at", sa.DateTime()),
        sa.Column("deleted_at", sa.DateTime()),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_assets_owner", "assets", ["owner_type", "owner_id", "kind"])
    op.create_index(
        "ix_assets_expires",
        "assets",
        ["expires_at"],
        postgresql_where=sa.text("expires_at IS NOT NULL"),
    )


def downgrade() -> None:
    # Порядок обратный созданию: сначала зависимые таблицы, потом типы.
    for table in (
        "assets",
        "clip_scenes",
        "clips",
        "transcript_segments",
        "scenes",
        "jobs",
        "videos",
        "users",
    ):
        op.drop_table(table)

    bind = op.get_bind()
    for name in _ENUMS:
        postgresql.ENUM(name=name).drop(bind, checkfirst=True)
