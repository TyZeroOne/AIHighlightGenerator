"""Точка входа FastAPI-приложения."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.errors import install_error_handlers
from api.middleware import TraceMiddleware
from api.routers import clips, health, jobs, system, uploads, videos
from core import __version__
from core.container import get_container
from core.logging import get_logger

__all__ = ["app", "create_app"]

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Старт и остановка приложения.

    ``lifespan`` вместо устаревших ``@app.on_event``: он гарантирует, что
    код завершения выполнится даже при ошибке во время старта, и корректно
    работает с asyncio-контекстом.
    """
    container = get_container()
    log.info("api.startup", version=__version__, env=container.settings.app_env)
    try:
        yield
    finally:
        await container.aclose()
        log.info("api.shutdown")


def create_app() -> FastAPI:
    """Фабрика приложения.

    Фабрика, а не модульный синглтон: тесту нужно поднять приложение
    с подменёнными настройками, не трогая глобальное состояние.
    """
    application = FastAPI(
        title="AI Highlight Generator",
        description="Автоматический поиск интересных моментов в видео и генерация клипов",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    application.add_middleware(TraceMiddleware)
    install_error_handlers(application)

    application.include_router(health.router)
    application.include_router(system.router)
    application.include_router(videos.router)
    application.include_router(uploads.router)
    application.include_router(jobs.router)
    application.include_router(clips.router)

    return application


app = create_app()
