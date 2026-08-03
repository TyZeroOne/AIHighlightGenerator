"""Общие фикстуры."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

# Настройки читаются до импорта приложения — иначе Settings подхватит
# реальный .env разработчика и тесты станут зависеть от его машины.
os.environ.setdefault("APP_ENV", "dev")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("STORAGE_LOCAL_ROOT", str(Path(__file__).parent / ".media"))

from core import registry
from core.config import get_settings
from core.container import reset_container

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture(autouse=True)
def _isolate_global_state() -> Iterator[None]:
    """Сбрасывает глобальное состояние между тестами.

    Реестр плагинов и контейнер — синглтоны на процесс. Без сброса тест,
    зарегистрировавший фейковый анализатор, ломает следующий за ним.
    Это цена глобального реестра, и она платится здесь один раз.
    """
    yield
    registry.clear_registries()
    reset_container()
    get_settings.cache_clear()


@pytest.fixture
def settings():  # type: ignore[no-untyped-def]
    return get_settings()
