"""Фикстуры интеграционных тестов: живая БД, контейнер, HTTP-клиент.

Тесты работают против **отдельной базы** ``highlights_test``, а не против
той, где лежат данные разработчика. Схема создаётся один раз на сессию,
таблицы чистятся между тестами.

Почему не миграциями: их обратимость проверяется отдельно
(``alembic upgrade → downgrade → upgrade`` в Docker), а здесь важна скорость —
``create_all`` отрабатывает за доли секунды против секунд на миграции.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from api.deps import generate_api_key
from core.config import Settings
from core.container import Container
from core.domain.entities import User
from core.infra.db.base import Base

TEST_DB_NAME = os.environ.get("TEST_POSTGRES_DB", "highlights_test")


def _settings(tmp_path: Path) -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        app_env="dev",
        log_level="WARNING",
        postgres_host=os.environ.get("POSTGRES_HOST", "localhost"),
        postgres_port=int(os.environ.get("POSTGRES_PORT", "5432")),
        postgres_db=TEST_DB_NAME,
        postgres_user=os.environ.get("POSTGRES_USER", "hg"),
        postgres_password=os.environ.get("POSTGRES_PASSWORD", "hg-dev-password"),
        storage_backend="local",
        storage_local_root=tmp_path / "media",
        workdir=tmp_path / "work",
        max_upload_bytes=8 * 1024 * 1024,
        max_duration_sec=3600,
        local_source_roots_raw=str(tmp_path / "mounted"),
    )


@pytest.fixture(scope="session")
async def _database() -> AsyncIterator[None]:
    """Создать тестовую БД и схему; удалить схему после сессии."""
    admin_dsn = (
        f"postgresql+asyncpg://{os.environ.get('POSTGRES_USER', 'hg')}:"
        f"{os.environ.get('POSTGRES_PASSWORD', 'hg-dev-password')}@"
        f"{os.environ.get('POSTGRES_HOST', 'localhost')}:"
        f"{os.environ.get('POSTGRES_PORT', '5432')}/postgres"
    )

    # CREATE DATABASE нельзя выполнить внутри транзакции — отсюда AUTOCOMMIT.
    admin_engine = create_async_engine(admin_dsn, isolation_level="AUTOCOMMIT")
    try:
        async with admin_engine.connect() as connection:
            exists = await connection.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": TEST_DB_NAME}
            )
            if not exists:
                await connection.execute(text(f'CREATE DATABASE "{TEST_DB_NAME}"'))
    except Exception as error:
        pytest.skip(f"PostgreSQL недоступен: {error}")
    finally:
        await admin_engine.dispose()

    dsn = admin_dsn.rsplit("/", 1)[0] + f"/{TEST_DB_NAME}"
    engine = create_async_engine(dsn)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)
    await engine.dispose()

    yield


@pytest.fixture
async def container(_database: None, tmp_path: Path) -> AsyncIterator[Container]:
    instance = Container(_settings(tmp_path))
    try:
        yield instance
    finally:
        await instance.aclose()


@pytest.fixture(autouse=True)
async def _clean_tables(container: Container) -> AsyncIterator[None]:
    """Пустые таблицы перед каждым тестом.

    TRUNCATE ... CASCADE, а не DROP/CREATE: на порядок быстрее и сбрасывает
    внешние ключи заодно. Тест, оставивший данные, не должен влиять на следующий.
    """
    async with container.engine.begin() as connection:
        names = ", ".join(f'"{table.name}"' for table in Base.metadata.sorted_tables)
        await connection.execute(text(f"TRUNCATE {names} RESTART IDENTITY CASCADE"))
    yield


@pytest.fixture
async def user_with_key(container: Container) -> tuple[User, str]:
    key, key_hash = generate_api_key()
    async with container.uow() as uow:
        user = await uow.users.create(email=f"{uuid4()}@test.local", api_key_hash=key_hash)
        await uow.commit()
    return user, key


@pytest.fixture
async def client(container: Container) -> AsyncIterator[AsyncClient]:
    """HTTP-клиент поверх приложения с подменённым контейнером.

    Подмена через ``dependency_overrides`` — штатный механизм FastAPI:
    приложение остаётся настоящим, меняется только источник зависимостей.
    """
    from api.deps import _container_dep
    from api.main import create_app

    app = create_app()
    app.dependency_overrides[_container_dep] = lambda: container

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def auth(user_with_key: tuple[User, str]) -> dict[str, str]:
    return {"X-API-Key": user_with_key[1]}
