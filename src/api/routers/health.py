"""Health-чеки.

Два разных эндпоинта, и разница между ними принципиальна:

* ``/health``       — liveness. «Процесс жив?» Не трогает зависимости.
                      Оркестратор по нему решает, надо ли перезапустить контейнер.
* ``/health/ready`` — readiness. «Готов принимать трафик?» Проверяет БД, Redis.
                      Балансировщик по нему решает, слать ли сюда запросы.

Если смешать их в один, потеря связи с БД приведёт к бесконечному
перезапуску контейнеров вместо простого вывода их из балансировки.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter, Response, status
from pydantic import BaseModel, Field

from core.container import Container, get_container
from core.infra.db.session import ping_database
from core.logging import get_logger

__all__ = ["router"]

router = APIRouter(tags=["health"])
log = get_logger(__name__)

CHECK_TIMEOUT_SEC = 3.0


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    version: str
    environment: str


class ReadinessResponse(BaseModel):
    status: str = Field(examples=["ready", "degraded"])
    checks: dict[str, str]


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health() -> HealthResponse:
    from core import __version__

    settings = get_container().settings
    return HealthResponse(status="ok", version=__version__, environment=settings.app_env.value)


@router.get("/health/ready", response_model=ReadinessResponse, summary="Readiness probe")
async def readiness(response: Response) -> ReadinessResponse:
    container = get_container()
    checks = await _run_checks(
        {
            "database": lambda: ping_database(container.engine),
            "redis": lambda: _ping_redis(container),
            "storage": lambda: _ping_storage(container),
        }
    )

    ready = all(value == "ok" for value in checks.values())
    if not ready:
        # 503, а не 200 с телом «плохо»: балансировщики смотрят на код ответа.
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(status="ready" if ready else "degraded", checks=checks)


async def _run_checks(
    checks: dict[str, Callable[[], Awaitable[Any]]],
) -> dict[str, str]:
    """Все проверки параллельно и с общим таймаутом.

    Последовательно было бы 3 × timeout в худшем случае — readiness-проба
    успела бы отвалиться по своему таймауту раньше, чем мы ответим.
    """

    async def run(name: str, check: Callable[[], Awaitable[Any]]) -> tuple[str, str]:
        try:
            await asyncio.wait_for(check(), timeout=CHECK_TIMEOUT_SEC)
        except NotImplementedError:
            return name, "not_configured"
        except TimeoutError:
            return name, "timeout"
        except Exception as exc:
            log.warning("health.check_failed", check=name, error=str(exc))
            return name, "error"
        return name, "ok"

    results = await asyncio.gather(*(run(name, check) for name, check in checks.items()))
    return dict(results)


async def _ping_redis(container: Container) -> None:
    from redis.asyncio import Redis

    client: Redis = Redis.from_url(str(container.settings.redis_url))
    try:
        await client.ping()
    finally:
        await client.aclose()


async def _ping_storage(container: Container) -> None:
    """Проверка настроенного хранилища через его собственный healthcheck.

    Роутер не знает, локальная это ФС или S3: обе реализации отвечают
    на один и тот же вызов. Ровно ради этого и заводился порт.
    """
    await container.storage.healthcheck()
