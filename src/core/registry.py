"""Реестры плагинов: анализаторы признаков и источники видео.

Здесь живёт механизм, ради которого затевалась вся архитектура: добавление
нового признака или нового источника не требует правки ядра, конвейера и API.

Три уровня расширения:

1. Модуль внутри репозитория помечает класс декоратором ``@register_analyzer``.
2. Внешний пакет объявляет entry point в группе ``highlight.analyzers``.
3. Конфиг отключает ненужное через ``ANALYZERS_DISABLED``.
"""

from __future__ import annotations

import importlib
from collections.abc import Collection, Iterable, Mapping
from importlib.metadata import entry_points
from typing import Final

from core.domain.errors import DependencyResolutionError, DuplicateRegistrationError
from core.ports.analysis import FeatureAnalyzer
from core.ports.source import SourceProvider

__all__ = [
    "ANALYZER_ENTRY_POINT_GROUP",
    "SOURCE_ENTRY_POINT_GROUP",
    "build_plan",
    "clear_registries",
    "discover",
    "get_analyzer",
    "get_analyzers",
    "get_source_providers",
    "register_analyzer",
    "register_source",
    "resolve_source",
]

ANALYZER_ENTRY_POINT_GROUP: Final = "highlight.analyzers"
SOURCE_ENTRY_POINT_GROUP: Final = "highlight.sources"

#: Встроенные модули, которые импортируются при ``discover()``.
#: Импорт модуля исполняет декораторы — так классы попадают в реестр.
_BUILTIN_MODULES: Final[tuple[str, ...]] = (
    "video_loader.providers",
    "audio.analyzers",
    "speech.analyzers",
    "vision.analyzers",
)

_ANALYZERS: dict[str, type[FeatureAnalyzer]] = {}
_SOURCES: dict[str, type[SourceProvider]] = {}
_discovered = False


# ------------------------------------------------------------ регистрация
def register_analyzer[A: type[FeatureAnalyzer]](cls: A) -> A:
    """Декоратор регистрации анализатора.

    Возвращает класс без изменений — декоратор нужен только ради побочного
    эффекта, поэтому обычное использование класса не ломается.
    """
    key = getattr(cls, "key", "")
    if not key:
        raise DependencyResolutionError(
            f"Анализатор {cls.__name__} не объявил атрибут 'key'",
            analyzer=cls.__name__,
        )
    if key in _ANALYZERS and _ANALYZERS[key] is not cls:
        raise DuplicateRegistrationError(
            f"Анализатор с ключом {key!r} уже зарегистрирован: "
            f"{_ANALYZERS[key].__name__} против {cls.__name__}",
            key=key,
        )
    _ANALYZERS[key] = cls
    return cls


def register_source[S: type[SourceProvider]](cls: S) -> S:
    """Декоратор регистрации источника видео."""
    scheme = getattr(cls, "scheme", "")
    if not scheme:
        raise DependencyResolutionError(
            f"Источник {cls.__name__} не объявил атрибут 'scheme'",
            source=cls.__name__,
        )
    if scheme in _SOURCES and _SOURCES[scheme] is not cls:
        raise DuplicateRegistrationError(
            f"Источник со схемой {scheme!r} уже зарегистрирован: "
            f"{_SOURCES[scheme].__name__} против {cls.__name__}",
            scheme=scheme,
        )
    _SOURCES[scheme] = cls
    return cls


# ------------------------------------------------------------- обнаружение
def discover(*, force: bool = False) -> None:
    """Найти и зарегистрировать все плагины.

    Вызывается один раз при старте процесса (в ``Container``). Повторные
    вызовы бесплатны — отсюда флаг ``_discovered``.
    """
    global _discovered
    if _discovered and not force:
        return

    for module_name in _BUILTIN_MODULES:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            # Нормально: модули появляются постепенно по этапам дорожной карты.
            continue

    for group, register in (
        (ANALYZER_ENTRY_POINT_GROUP, register_analyzer),
        (SOURCE_ENTRY_POINT_GROUP, register_source),
    ):
        for entry_point in entry_points(group=group):
            register(entry_point.load())

    _discovered = True


def clear_registries() -> None:
    """Полная очистка. Только для тестов: реестр — глобальное состояние,
    и без сброса тесты начинают влиять друг на друга."""
    global _discovered
    _ANALYZERS.clear()
    _SOURCES.clear()
    _discovered = False


# ------------------------------------------------------------------ доступ
def get_analyzers() -> Mapping[str, type[FeatureAnalyzer]]:
    return dict(_ANALYZERS)


def get_analyzer(key: str) -> type[FeatureAnalyzer]:
    try:
        return _ANALYZERS[key]
    except KeyError:
        raise DependencyResolutionError(
            f"Анализатор {key!r} не зарегистрирован. Доступны: {sorted(_ANALYZERS)}",
            key=key,
        ) from None


def get_source_providers() -> Mapping[str, type[SourceProvider]]:
    return dict(_SOURCES)


def resolve_source(uri: str) -> type[SourceProvider]:
    """Подобрать провайдер под URI.

    Спрашиваем сами провайдеры через ``supports()``, а не парсим схему здесь:
    иначе ядро пришлось бы править при добавлении каждого нового источника.
    """
    for provider_cls in _SOURCES.values():
        if provider_cls.supports(uri):
            return provider_cls
    raise DependencyResolutionError(
        f"Нет источника, способного обработать URI: {uri!r}",
        uri=uri,
    )


# ------------------------------------------------------------ планирование
def build_plan(
    enabled: Collection[str] | None = None,
    *,
    available_artifacts: Collection[str] = (),
    disabled: Collection[str] = (),
) -> list[list[type[FeatureAnalyzer]]]:
    """Разложить анализаторы по слоям исполнения.

    Внутри слоя анализаторы независимы — их можно запускать параллельно
    (в Celery это ``group()``). Слои идут строго последовательно.

    :param enabled: какие анализаторы брать; ``None`` — все зарегистрированные.
    :param available_artifacts: сырьё, доступное на старте
        (``artifact.video``, ``artifact.audio``, ``artifact.transcript``).
    :param disabled: что исключить (из ``ANALYZERS_DISABLED``).
    :raises DependencyResolutionError: если остались анализаторы, чьи
        зависимости невозможно удовлетворить (цикл или отсутствующее сырьё).
    """
    keys = set(enabled) if enabled is not None else set(_ANALYZERS)
    keys -= set(disabled)

    pending: dict[str, type[FeatureAnalyzer]] = {key: get_analyzer(key) for key in keys}
    satisfied: set[str] = set(available_artifacts)
    layers: list[list[type[FeatureAnalyzer]]] = []

    while pending:
        ready = [cls for key, cls in pending.items() if set(cls.requires).issubset(satisfied)]
        if not ready:
            raise DependencyResolutionError(
                "Невозможно построить план анализа: зависимости неразрешимы",
                unresolved=sorted(pending),
                missing=sorted(_missing_requirements(pending.values(), satisfied)),
                satisfied=sorted(satisfied),
            )

        # Дорогие анализаторы — в начало слоя: пока они считаются, дешёвые
        # успевают отработать, и слой не растягивается «хвостом».
        ready.sort(key=lambda cls: (-cls.cost, cls.key))
        layers.append(ready)

        for cls in ready:
            del pending[cls.key]
            satisfied |= set(cls.produces)

    return layers


def _missing_requirements(
    analyzers: Iterable[type[FeatureAnalyzer]],
    satisfied: Collection[str],
) -> set[str]:
    missing: set[str] = set()
    for cls in analyzers:
        missing |= set(cls.requires) - set(satisfied)
    return missing
