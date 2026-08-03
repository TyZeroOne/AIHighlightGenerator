"""Перевод колонок времени в timestamptz

Revision ID: 0003
Revises: 0002
Create Date: Этап 2

В миграции 0001 часть колонок времени была объявлена как ``TIMESTAMP``
(без часового пояса). Приложение работает со временем в UTC и передаёт
aware-datetime, из-за чего asyncpg падал с «can't subtract offset-naive
and offset-aware datetimes».

Ошибка не «косметическая»: ``TIMESTAMP WITHOUT TIME ZONE`` хранит момент
времени без указания, в каком поясе он записан. Два процесса в разных
поясах запишут в одну колонку разные моменты, и отличить их потом
невозможно. Для системы, где воркеры могут стоять в других зонах,
это гарантированная порча данных.

``USING ... AT TIME ZONE 'UTC'`` трактует уже записанные значения как UTC —
это верно, потому что всё, что писалось до сих пор, писалось в UTC.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_COLUMNS: tuple[tuple[str, str], ...] = (
    ("jobs", "started_at"),
    ("jobs", "finished_at"),
    ("scenes", "created_at"),
    ("clips", "created_at"),
    ("assets", "expires_at"),
    ("assets", "deleted_at"),
    ("assets", "created_at"),
    ("upload_sessions", "expires_at"),
)


def upgrade() -> None:
    for table, column in _COLUMNS:
        op.alter_column(
            table,
            column,
            type_=sa.DateTime(timezone=True),
            postgresql_using=f"{column} AT TIME ZONE 'UTC'",
        )


def downgrade() -> None:
    for table, column in _COLUMNS:
        op.alter_column(
            table,
            column,
            type_=sa.DateTime(),
            postgresql_using=f"{column} AT TIME ZONE 'UTC'",
        )
