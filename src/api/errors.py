"""Единая обработка ошибок в формате RFC 7807 (``application/problem+json``).

Зачем стандарт, а не свой формат: RFC 7807 понимают готовые клиенты и
инструменты, а главное — он заставляет отдавать машиночитаемый ``code``
рядом с человекочитаемым ``detail``. Клиент ветвится по ``code``, человек
читает ``detail``.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from core.domain.errors import HighlightError
from core.logging import get_logger, get_trace_id

__all__ = ["install_error_handlers", "problem_response"]

log = get_logger(__name__)

PROBLEM_MEDIA_TYPE = "application/problem+json"
DOCS_BASE = "https://docs.highlight-generator.local/errors"


def problem_response(
    request: Request,
    *,
    status_code: int,
    title: str,
    detail: str,
    code: str,
    **extra: Any,
) -> JSONResponse:
    body: dict[str, Any] = {
        "type": f"{DOCS_BASE}/{code.lower().replace('_', '-')}",
        "title": title,
        "status": status_code,
        "detail": detail,
        "code": code,
        "instance": request.url.path,
        "trace_id": get_trace_id(),
    }
    body.update(extra)
    return JSONResponse(status_code=status_code, content=body, media_type=PROBLEM_MEDIA_TYPE)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(HighlightError)
    async def _domain_error(request: Request, exc: HighlightError) -> JSONResponse:
        # Ожидаемые ошибки логируем как warning: это не сбой системы,
        # а штатный отказ. Сбои (500) пойдут ниже, через _unhandled.
        log.warning(
            "api.domain_error",
            code=exc.code,
            status=exc.http_status,
            detail=exc.message,
            **exc.context,
        )
        return problem_response(
            request,
            status_code=exc.http_status,
            title=exc.__class__.__name__,
            detail=exc.message,
            code=exc.code,
            **({"context": exc.context} if exc.context else {}),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return problem_response(
            request,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            title="Validation error",
            detail="Тело запроса не прошло валидацию",
            code="VALIDATION_ERROR",
            errors=[
                {"field": ".".join(str(p) for p in err["loc"][1:]), "message": err["msg"]}
                for err in exc.errors()
            ],
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return problem_response(
            request,
            status_code=exc.status_code,
            title="HTTP error",
            detail=str(exc.detail),
            code=_HTTP_CODES.get(exc.status_code, "HTTP_ERROR"),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Наружу не отдаём ни текст исключения, ни трейс: это утечка деталей
        # реализации. Клиент получает trace_id и по нему находится всё в логах.
        log.exception("api.unhandled_error", error_type=type(exc).__name__)
        return problem_response(
            request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            title="Internal server error",
            detail="Внутренняя ошибка сервера. Сообщите trace_id в поддержку.",
            code="INTERNAL_ERROR",
        )


_HTTP_CODES = {
    401: "UNAUTHENTICATED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    413: "FILE_TOO_LARGE",
    415: "UNSUPPORTED_MEDIA",
    429: "RATE_LIMITED",
}
