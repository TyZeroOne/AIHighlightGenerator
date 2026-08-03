"""Тест, охраняющий правило зависимостей.

Архитектура, которая держится на дисциплине, разваливается на третьем месяце.
Здесь она держится на тесте: любой импорт, нарушающий слоистость, роняет CI.

Проверка идёт по AST, а не по фактическим импортам в рантайме — так
нарушение ловится, даже если ветка кода никогда не исполняется.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"

#: Пакеты-адаптеры: реализуют порты, но не знают друг о друге.
ADAPTER_PACKAGES = frozenset(
    {
        "video_loader",
        "audio",
        "speech",
        "vision",
        "highlight_detector",
        "llm",
        "video_editor",
        "subtitles",
        "export",
    }
)

#: Тяжёлые/инфраструктурные библиотеки, запрещённые в чистых слоях.
FORBIDDEN_IN_PURE_DOMAIN = frozenset(
    {
        "sqlalchemy",
        "fastapi",
        "celery",
        "redis",
        "torch",
        "cv2",
        "numpy",
        "pydantic",
        "structlog",
        "anthropic",
        "httpx",
        "starlette",
    }
)


def _iter_modules(package: str) -> list[Path]:
    root = SRC / package.replace(".", "/")
    return sorted(root.rglob("*.py")) if root.exists() else []


def _imported_roots(path: Path) -> set[str]:
    """Корневые имена всех импортов модуля, включая внутри ``if TYPE_CHECKING``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _imported_modules(path: Path) -> set[str]:
    """Полные имена импортируемых модулей."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
    return modules


@pytest.mark.unit
@pytest.mark.parametrize("package", ["core.domain", "core.ports"])
def test_pure_layers_have_no_heavy_dependencies(package: str) -> None:
    """``core.domain`` и ``core.ports`` — только стандартная библиотека."""
    violations: list[str] = []
    for module in _iter_modules(package):
        forbidden = _imported_roots(module) & FORBIDDEN_IN_PURE_DOMAIN
        if forbidden:
            violations.append(f"{module.relative_to(SRC)}: {sorted(forbidden)}")

    assert not violations, (
        f"Слой {package} должен зависеть только от стандартной библиотеки.\n"
        + "\n".join(violations)
    )


@pytest.mark.unit
def test_domain_does_not_import_other_project_packages() -> None:
    """Домен не знает даже об остальном ``core`` — он в самом низу."""
    allowed_prefixes = ("core.domain",)
    violations: list[str] = []

    for module in _iter_modules("core.domain"):
        for imported in _imported_modules(module):
            if imported.startswith("core.") and not imported.startswith(allowed_prefixes):
                violations.append(f"{module.relative_to(SRC)} → {imported}")

    assert not violations, "core.domain импортирует внешние слои:\n" + "\n".join(violations)


@pytest.mark.unit
def test_ports_depend_only_on_domain() -> None:
    """Порты описывают контракты в терминах домена и больше ничего не знают."""
    allowed_prefixes = ("core.domain", "core.ports")
    violations: list[str] = []

    for module in _iter_modules("core.ports"):
        for imported in _imported_modules(module):
            if imported.startswith("core.") and not imported.startswith(allowed_prefixes):
                violations.append(f"{module.relative_to(SRC)} → {imported}")

    assert not violations, "core.ports импортирует лишнее:\n" + "\n".join(violations)


@pytest.mark.unit
def test_usecases_depend_only_on_domain_and_ports() -> None:
    """Сценарии знают, ЧТО делать, но не знают, ЧЕМ.

    Правило добавлено на Этапе 2 после реального нарушения: сценарий загрузки
    импортировал ``video_loader.validation`` и конструировал ``LocalFileProvider``
    напрямую. Работало — но означало, что заменить провайдер без правки
    сценария уже нельзя, то есть вся конструкция начала протекать.
    """
    allowed_core = ("core.domain", "core.ports", "core.usecases", "core.logging")
    violations: list[str] = []

    for module in _iter_modules("core.usecases"):
        for imported in _imported_modules(module):
            root = imported.split(".")[0]
            is_adapter = root in ADAPTER_PACKAGES or root in {"api", "workers"}
            is_forbidden_core = imported.startswith("core.") and not imported.startswith(
                allowed_core
            )
            if is_adapter or is_forbidden_core:
                violations.append(f"{module.relative_to(SRC)} → {imported}")

    assert not violations, (
        "core.usecases должен зависеть только от домена, портов и логирования:\n"
        + "\n".join(violations)
    )


@pytest.mark.unit
def test_usecases_do_not_import_infrastructure() -> None:
    """Ни SQLAlchemy, ни boto3, ни FastAPI в сценариях."""
    forbidden = {"sqlalchemy", "fastapi", "boto3", "celery", "redis", "starlette"}
    violations: list[str] = []

    for module in _iter_modules("core.usecases"):
        found = _imported_roots(module) & forbidden
        if found:
            violations.append(f"{module.relative_to(SRC)}: {sorted(found)}")

    assert not violations, "Инфраструктура в сценариях:\n" + "\n".join(violations)


@pytest.mark.unit
@pytest.mark.parametrize("package", sorted(ADAPTER_PACKAGES))
def test_adapters_do_not_import_each_other(package: str) -> None:
    """Адаптеры общаются только через ядро.

    Без этого правила ``vision`` начнёт импортировать ``speech``, тот —
    ``audio``, и через полгода вынуть один модуль станет невозможно.
    """
    others = ADAPTER_PACKAGES - {package}
    violations: list[str] = []

    for module in _iter_modules(package):
        for imported in _imported_modules(module):
            if imported.split(".")[0] in others:
                violations.append(f"{module.relative_to(SRC)} → {imported}")

    assert not violations, f"Пакет {package} импортирует другой адаптер напрямую:\n" + "\n".join(
        violations
    )


@pytest.mark.unit
def test_adapters_do_not_import_api_or_workers() -> None:
    """Зависимость всегда направлена внутрь: адаптер не знает о доставке."""
    violations: list[str] = []
    for package in sorted(ADAPTER_PACKAGES):
        for module in _iter_modules(package):
            for imported in _imported_modules(module):
                if imported.split(".")[0] in {"api", "workers"}:
                    violations.append(f"{module.relative_to(SRC)} → {imported}")

    assert not violations, "Адаптер импортирует слой доставки:\n" + "\n".join(violations)


#: Единственный файл ядра, которому разрешено знать конкретные реализации.
#: Композиционный корень для того и существует: где-то абстракция обязана
#: встретиться с конкретикой, и лучше, чтобы это было одно предсказуемое
#: место, а не тридцать разбросанных импортов.
COMPOSITION_ROOT = "core/container.py"


@pytest.mark.unit
def test_core_does_not_import_adapters_outside_composition_root() -> None:
    """Ядро не знает о реализациях — кроме композиционного корня.

    ``core.registry`` тоже подключает адаптеры, но **динамически**
    (``importlib.import_module`` по строке), поэтому статический анализ
    его не видит — и это тоже намеренно: реестр не должен знать имена
    классов, только группы entry points.
    """
    violations: list[str] = []
    for module in _iter_modules("core"):
        relative = module.relative_to(SRC).as_posix()
        if relative == COMPOSITION_ROOT:
            continue
        for imported in _imported_modules(module):
            if imported.split(".")[0] in ADAPTER_PACKAGES:
                violations.append(f"{relative} → {imported}")

    assert not violations, (
        "Ядро импортирует адаптер напрямую. Если это осознанное связывание — "
        f"его место в {COMPOSITION_ROOT}:\n" + "\n".join(violations)
    )


@pytest.mark.unit
def test_composition_root_is_the_only_place_wiring_adapters() -> None:
    """Обратная проверка: корень действительно связывает реализации.

    Тест выше можно было бы «пройти», перестав связывать что-либо вовсе.
    Этот следит, чтобы исключение оставалось осмысленным: если импорты
    адаптеров из корня исчезли, значит, они переехали куда-то ещё —
    и надо разобраться куда.
    """
    root = SRC / COMPOSITION_ROOT
    imported = _imported_modules(root)
    wired = {name for name in imported if name.split(".")[0] in ADAPTER_PACKAGES}

    assert wired, (
        f"{COMPOSITION_ROOT} не связывает ни одной реализации — "
        "проверьте, куда переехало связывание"
    )
