"""Отдача метрик и их сбор с HTTP-слоя.

## Почему шаблон пути, а не сам путь

Метка ``path`` обязана быть шаблоном (``/api/v1/jobs/{job_id}``), а не
конкретным адресом. Иначе каждый новый идентификатор джоба порождает
свой временной ряд, и через сутки работы Prometheus хранит миллион рядов
по одному значению в каждом — это называется взрывом кардинальности
и убивает мониторинг надёжнее, чем его отсутствие.

Шаблон берётся из маршрутизатора: FastAPI кладёт совпавший ``APIRoute``
в scope запроса уже после разбора.

## Почему /metrics не требует ключа

Его опрашивает Prometheus, у которого ключа нет и быть не должно.
Эндпоинт не отдаёт ничего чувствительного — только счётчики и
длительности, — а закрывать его надо не аутентификацией, а тем, что
порт метрик не выставлен наружу.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable
from typing import cast

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request

from core.container import get_container
from core.infra.metrics import HTTP_DURATION, HTTP_REQUESTS, QUEUE_DEPTH, build_registry
from core.logging import get_logger

__all__ = ["MetricsMiddleware", "router"]

log = get_logger(__name__)

router = APIRouter(tags=["system"])

#: Очереди, глубину которых показываем. Списком, а не «всё, что нашлось
#: в Redis»: там же лежат ключи результатов задач, и перебор по маске
#: на живой базе — плохая идея.
_QUEUES = ("cpu", "io", "gpu", "net")


class MetricsMiddleware(BaseHTTPMiddleware):
    """Считает запросы и их длительность."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        started = time.perf_counter()
        response = await call_next(request)

        path = _template(request)
        # Сам /metrics в метрики не попадает: Prometheus опрашивает его
        # каждые несколько секунд, и он занял бы верхнюю строчку в любом
        # отчёте о частоте запросов, ничего о системе не сообщая.
        if path != "/metrics":
            HTTP_REQUESTS.labels(
                method=request.method, path=path, status=str(response.status_code)
            ).inc()
            HTTP_DURATION.labels(method=request.method, path=path).observe(
                time.perf_counter() - started
            )
        return response


@router.get(
    "/metrics",
    summary="Метрики в формате Prometheus",
    response_class=Response,
    include_in_schema=False,
)
async def metrics() -> Response:
    await _refresh_queue_depth()
    return Response(generate_latest(build_registry()), media_type=CONTENT_TYPE_LATEST)


def _template(request: Request) -> str:
    """Шаблон маршрута или ``unmatched``.

    ``unmatched`` — это все 404 на несуществующие пути. Сваливать их
    в одну метку намеренно: сканеры уязвимостей перебирают тысячи адресов,
    и каждый из них не должен становиться отдельным рядом.
    """
    route = request.scope.get("route")
    template = getattr(route, "path", None)
    return str(template) if template else "unmatched"


async def _refresh_queue_depth() -> None:
    """Глубина очередей — на момент опроса.

    Считается здесь, а не пишется задачами: это состояние брокера,
    а не событие приложения. Спрашивать его на каждый scrape дешевле,
    чем поддерживать счётчик, который разъедется с реальностью при
    первой же потере сообщения.

    Ошибка Redis гасится: недоступный брокер — это повод для тревоги
    по другим метрикам, а не причина не отдать те, что есть.
    """
    try:
        from redis.asyncio import Redis

        client: Redis = Redis.from_url(str(get_container().settings.celery_broker_url))
        try:
            for queue in _QUEUES:
                # ``llen`` в асинхронном клиенте типизирован как «или
                # awaitable, или число» ради общего кода с синхронным —
                # приводим явно.
                depth = cast("Awaitable[int]", client.llen(queue))
                QUEUE_DEPTH.labels(queue=queue).set(await depth)
        finally:
            await client.aclose()
    except Exception as error:  # pragma: no cover — зависит от доступности Redis
        log.warning("metrics.queue_depth_failed", error=str(error))
