"""Асинхронный движок и фабрика сессий SQLAlchemy."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from core.config import Settings

__all__ = ["create_engine", "create_session_factory", "ping_database"]


def create_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(
        settings.database_url,
        echo=settings.db_echo,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_pre_ping=True,  # проверяет соединение перед выдачей: спасает от
        # «MySQL/PG has gone away» после простоя пула
        future=True,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,  # иначе после commit() обращение к полю объекта
        # инициирует новый SELECT — в async-коде это ловушка
        autoflush=False,
    )


@asynccontextmanager
async def session_scope(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Сессия с автоматическим откатом при исключении."""
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def ping_database(engine: AsyncEngine) -> None:
    """Проверка живости для ``/health/ready``. Бросает исключение при сбое."""
    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))
