"""Зависимости FastAPI: контейнер и аутентификация.

Аутентификация здесь **минимальная и осознанно неполная**: API-ключ,
хеш которого лежит в БД. Полноценная авторизация — JWT, роли, квоты,
rate limiting — это Этап 10.

Почему не отложить её целиком: без пользователя эндпоинты бессмысленны.
У видео есть владелец, проверка владения — часть DoD этапа, а «временный
хардкод user_id» имеет свойство доживать до продакшена.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Annotated

from fastapi import Depends, Header

from core.container import Container, get_container
from core.domain.entities import User
from core.domain.errors import HighlightError

__all__ = [
    "ContainerDep",
    "CurrentUser",
    "generate_api_key",
    "hash_api_key",
]

API_KEY_PREFIX = "hg_"


class UnauthenticatedError(HighlightError):
    http_status = 401
    code = "UNAUTHENTICATED"


def generate_api_key() -> tuple[str, str]:
    """Новый ключ и его хеш. Открытый ключ показывается пользователю один раз."""
    key = f"{API_KEY_PREFIX}{secrets.token_urlsafe(32)}"
    return key, hash_api_key(key)


def hash_api_key(key: str) -> str:
    """SHA-256 без соли — и это здесь правильно.

    Соль и медленные функции (argon2, bcrypt) нужны паролям, потому что
    пароли короткие, осмысленные и перебираются по словарю. API-ключ —
    256 бит случайности: перебирать нечего, а быстрый хеш важен, потому
    что проверка идёт на каждом запросе.

    Для паролей пользователей (Этап 10) будет argon2 — там ситуация обратная.
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


async def _container_dep() -> Container:
    return get_container()


ContainerDep = Annotated[Container, Depends(_container_dep)]


async def _current_user(
    container: ContainerDep,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> User:
    if not x_api_key:
        raise UnauthenticatedError("Требуется заголовок X-API-Key")

    key_hash = hash_api_key(x_api_key)
    async with container.uow() as uow:
        user = await uow.users.get_by_api_key_hash(key_hash)

    if user is None:
        # Сравнение с фиктивным значением занимает столько же времени,
        # сколько успешное: иначе по времени ответа можно было бы отличить
        # «ключа нет» от «ключ есть, но неактивен».
        hmac.compare_digest(key_hash, "0" * 64)
        raise UnauthenticatedError("Неверный API-ключ")
    if not user.is_active:
        raise UnauthenticatedError("Учётная запись отключена")

    return user


CurrentUser = Annotated[User, Depends(_current_user)]
