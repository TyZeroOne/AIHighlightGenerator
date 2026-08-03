"""Таблица upload_sessions — состояние чанковых загрузок

Revision ID: 0002
Revises: 0001
Create Date: Этап 2
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "upload_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("chunk_size", sa.Integer(), nullable=False),
        sa.Column("total_size", sa.BigInteger()),
        # Массив номеров, а не счётчик: части приходят не по порядку и могут
        # повторяться при ретраях, и нам нужно знать, чего именно не хватает.
        sa.Column(
            "received_parts",
            postgresql.ARRAY(sa.Integer()),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column("received_bytes", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("completed", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("video_id", postgresql.UUID(as_uuid=True)),
        sa.Column("expires_at", sa.DateTime()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], ondelete="CASCADE", name="fk_upload_sessions_user_id_users"
        ),
        sa.CheckConstraint("chunk_size > 0", name="ck_upload_sessions_chunk_size_positive"),
    )
    op.create_index("ix_upload_sessions_user", "upload_sessions", ["user_id", "completed"])
    # Частичный индекс: чистить нужно только незавершённые сессии,
    # завершённые остаются как история и в выборку уборщика не попадают.
    op.create_index(
        "ix_upload_sessions_expires",
        "upload_sessions",
        ["expires_at"],
        postgresql_where=sa.text("completed = false"),
    )


def downgrade() -> None:
    op.drop_table("upload_sessions")
