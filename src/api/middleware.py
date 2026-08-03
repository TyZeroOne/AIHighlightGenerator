"""Middleware: сквозной ``trace_id`` и лог доступа."""

from __future__ import annotations

import time
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from core.logging import bind_trace, get_logger

__all__ = ["TraceMiddleware"]

log = get_logger("api.access")


class TraceMiddleware(BaseHTTPMiddleware):
    """Присваивает запросу ``trace_id`` и логирует результат.

    Идентификатор берётся из заголовка ``X-Request-ID``, если клиент его
    прислал, — так трассировка не рвётся на границе между сервисами.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming = request.headers.get("X-Request-ID")
        started = time.perf_counter()

        with bind_trace(trace_id=incoming or str(uuid4())) as trace_id:
            request.state.trace_id = trace_id
            response = await call_next(request)
            duration_ms = (time.perf_counter() - started) * 1000

            # /health опрашивается каждые несколько секунд — не засоряем лог.
            if not request.url.path.startswith("/health"):
                log.info(
                    "http.request",
                    method=request.method,
                    path=request.url.path,
                    status=response.status_code,
                    duration_ms=round(duration_ms, 2),
                )

            response.headers["X-Request-ID"] = trace_id
            return response
