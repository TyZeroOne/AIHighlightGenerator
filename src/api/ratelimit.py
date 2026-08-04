"""Ограничение частоты запросов на границе HTTP.

## Почему зависимость, а не middleware

Middleware видит все запросы, включая ``/health`` и ``/metrics``, —
а их дёргает не пользователь, а Kubernetes и Prometheus, и ограничивать
их означает уронить проверку живости под нагрузкой. Отделять служебные
пути внутри middleware пришлось бы списком строк, который разъедется
с маршрутами при первом же переименовании.

Зависимость подключается там, где нужна, и списка не требует.

## Ключ — пользователь, а не адрес

Адрес за NAT общий у офиса, а за прокси — у всех сразу. Ключ по
пользователю точен по определению: лимит выдан ему, а не его сети.
Обратная сторона — неаутентифицированные запросы ограничителем
не покрыты; их отсекает проверка ключа раньше, и стоит она дешевле.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends

from api.deps import ContainerDep, CurrentUser
from core.domain.errors import QuotaExceededError

__all__ = ["RateLimitError", "RateLimited", "rate_limit"]


class RateLimitError(QuotaExceededError):
    """429 с ``Retry-After``.

    Отдельный класс нужен ради заголовка: обработчик ошибок читает
    ``headers`` и добавляет их в ответ. Без ``Retry-After`` клиент
    не знает, когда пробовать снова, и в лучшем случае начинает
    долбиться раз в секунду.
    """

    def __init__(self, retry_after_sec: int, limit: int) -> None:
        super().__init__(
            f"Превышен лимит {limit} запросов в минуту. Повторите через {retry_after_sec} с.",
            retry_after_sec=retry_after_sec,
            limit=limit,
        )
        self.headers = {"Retry-After": str(retry_after_sec)}


async def _enforce(container: ContainerDep, user: CurrentUser) -> None:
    settings = container.settings
    if settings.rate_limit_rpm <= 0:
        return

    verdict = await container.rate_limiter.hit(
        f"user:{user.id}",
        limit=settings.rate_limit_rpm,
        window_sec=60,
    )
    if not verdict.allowed:
        raise RateLimitError(verdict.retry_after_sec, verdict.limit)


#: Подключается к роутерам, которые делают работу. Служебные пути
#: (``/health``, ``/metrics``) остаются без него намеренно.
RateLimited = Annotated[None, Depends(_enforce)]


def rate_limit() -> Any:
    """Та же проверка, но как зависимость роутера целиком.

    Через ``APIRouter(dependencies=[...])`` она применяется ко всем
    эндпоинтам сразу — и новый эндпоинт получает лимит по факту
    добавления, а не по факту того, что автор о нём вспомнил.

    Возвращаемый тип ``Any``, потому что ``Depends`` — не тип, а фабрика
    маркеров: FastAPI разбирает их сам, а аннотировать точнее нечем.
    """
    return Depends(_enforce)
