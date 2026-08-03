"""Тесты реестра плагинов и планировщика.

Реестр — механизм расширения проекта. Если он сломается, новые анализаторы
начнут молча не выполняться, и заметить это будет крайне трудно. Поэтому
покрываем плотно, включая ошибочные ситуации.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import ClassVar

import pytest

from core import registry
from core.domain.analysis import (
    ARTIFACT_AUDIO,
    ARTIFACT_TRANSCRIPT,
    ARTIFACT_VIDEO,
    AnalysisContext,
)
from core.domain.errors import DependencyResolutionError, DuplicateRegistrationError
from core.domain.values import FeatureValue, TimeRange

pytestmark = pytest.mark.unit


def make_analyzer(
    key: str,
    *,
    requires: set[str] | None = None,
    produces: set[str] | None = None,
    cost: int = 1,
) -> type:
    """Фабрика фейковых анализаторов.

    Обратите внимание: класс НЕ наследует ничего из ``core`` — соответствие
    протоколу структурное. Ровно так же сможет выглядеть сторонний плагин.
    """

    class _Analyzer:
        key: ClassVar[str] = ""
        requires: ClassVar[frozenset[str]] = frozenset()
        produces: ClassVar[frozenset[str]] = frozenset()
        cost: ClassVar[int] = 1

        async def analyze(
            self, context: AnalysisContext, scenes: Sequence[TimeRange]
        ) -> Mapping[int, Sequence[FeatureValue]]:
            return {}

    _Analyzer.key = key
    _Analyzer.requires = frozenset(requires or set())
    _Analyzer.produces = frozenset(produces or {f"{key}.value"})
    _Analyzer.cost = cost
    _Analyzer.__name__ = f"Analyzer_{key.replace('.', '_')}"
    return _Analyzer


class TestRegistration:
    def test_register_and_get(self) -> None:
        cls = registry.register_analyzer(make_analyzer("audio.loudness"))
        assert registry.get_analyzer("audio.loudness") is cls
        assert "audio.loudness" in registry.get_analyzers()

    def test_decorator_returns_class_unchanged(self) -> None:
        """Декоратор нужен ради побочного эффекта — сам класс не трогаем."""
        original = make_analyzer("x")
        assert registry.register_analyzer(original) is original

    def test_duplicate_key_is_rejected(self) -> None:
        registry.register_analyzer(make_analyzer("audio.loudness"))
        with pytest.raises(DuplicateRegistrationError, match=re.escape("audio.loudness")):
            registry.register_analyzer(make_analyzer("audio.loudness"))

    def test_reregistering_same_class_is_idempotent(self) -> None:
        """Повторный импорт модуля не должен ронять приложение."""
        cls = make_analyzer("audio.loudness")
        registry.register_analyzer(cls)
        registry.register_analyzer(cls)  # без исключения

    def test_analyzer_without_key_is_rejected(self) -> None:
        class NoKey:
            requires: ClassVar[frozenset[str]] = frozenset()
            produces: ClassVar[frozenset[str]] = frozenset()
            cost: ClassVar[int] = 1

        with pytest.raises(DependencyResolutionError, match="key"):
            registry.register_analyzer(NoKey)  # type: ignore[arg-type]

    def test_unknown_analyzer_error_lists_available(self) -> None:
        registry.register_analyzer(make_analyzer("audio.loudness"))
        with pytest.raises(DependencyResolutionError, match=re.escape("audio.loudness")):
            registry.get_analyzer("vision.motion")


class TestSourceRegistration:
    def test_register_and_resolve(self) -> None:
        class FileProvider:
            scheme: ClassVar[str] = "file"

            @classmethod
            def supports(cls, uri: str) -> bool:
                return uri.startswith("file://")

        class YouTubeProvider:
            scheme: ClassVar[str] = "youtube"

            @classmethod
            def supports(cls, uri: str) -> bool:
                return "youtube.com" in uri

        registry.register_source(FileProvider)  # type: ignore[arg-type]
        registry.register_source(YouTubeProvider)  # type: ignore[arg-type]

        assert registry.resolve_source("file:///data/movie.mkv") is FileProvider
        assert registry.resolve_source("https://youtube.com/watch?v=x") is YouTubeProvider

    def test_unknown_uri_raises(self) -> None:
        with pytest.raises(DependencyResolutionError, match="rtmp"):
            registry.resolve_source("rtmp://example.com/live")


class TestBuildPlan:
    def test_independent_analyzers_land_in_one_layer(self) -> None:
        """Независимые анализаторы должны идти параллельно, а не по очереди."""
        for key in ("audio.loudness", "audio.pauses", "vision.motion"):
            registry.register_analyzer(make_analyzer(key, requires={ARTIFACT_VIDEO}))

        plan = registry.build_plan(available_artifacts={ARTIFACT_VIDEO})

        assert len(plan) == 1
        assert {cls.key for cls in plan[0]} == {"audio.loudness", "audio.pauses", "vision.motion"}

    def test_dependency_creates_second_layer(self) -> None:
        registry.register_analyzer(
            make_analyzer(
                "speech.rate",
                requires={ARTIFACT_TRANSCRIPT},
                produces={"speech.rate.wpm"},
            )
        )
        registry.register_analyzer(
            make_analyzer("speech.tempo_delta", requires={"speech.rate.wpm"})
        )

        plan = registry.build_plan(available_artifacts={ARTIFACT_TRANSCRIPT})

        assert [c.key for c in plan[0]] == ["speech.rate"]
        assert [c.key for c in plan[1]] == ["speech.tempo_delta"]

    def test_expensive_analyzers_go_first_within_layer(self) -> None:
        """Дорогое — в начало слоя, иначе оно окажется «хвостом» и растянет этап."""
        registry.register_analyzer(make_analyzer("cheap", requires={ARTIFACT_VIDEO}, cost=1))
        registry.register_analyzer(make_analyzer("expensive", requires={ARTIFACT_VIDEO}, cost=10))

        plan = registry.build_plan(available_artifacts={ARTIFACT_VIDEO})

        assert [c.key for c in plan[0]] == ["expensive", "cheap"]

    def test_missing_artifact_raises_with_diagnostics(self) -> None:
        """Видео без аудио: анализатор громкости не может быть запланирован."""
        registry.register_analyzer(make_analyzer("audio.loudness", requires={ARTIFACT_AUDIO}))

        with pytest.raises(DependencyResolutionError) as exc_info:
            registry.build_plan(available_artifacts={ARTIFACT_VIDEO})

        assert ARTIFACT_AUDIO in exc_info.value.context["missing"]
        assert "audio.loudness" in exc_info.value.context["unresolved"]

    def test_cycle_is_detected(self) -> None:
        registry.register_analyzer(make_analyzer("a", requires={"b.value"}, produces={"a.value"}))
        registry.register_analyzer(make_analyzer("b", requires={"a.value"}, produces={"b.value"}))

        with pytest.raises(DependencyResolutionError, match="неразрешимы"):
            registry.build_plan(available_artifacts={ARTIFACT_VIDEO})

    def test_disabled_analyzers_are_excluded(self) -> None:
        registry.register_analyzer(make_analyzer("audio.loudness", requires={ARTIFACT_VIDEO}))
        registry.register_analyzer(make_analyzer("vision.faces", requires={ARTIFACT_VIDEO}))

        plan = registry.build_plan(available_artifacts={ARTIFACT_VIDEO}, disabled={"vision.faces"})

        assert [c.key for c in plan[0]] == ["audio.loudness"]

    def test_disabling_a_dependency_breaks_the_plan_loudly(self) -> None:
        """Отключив нижний анализатор, мы должны получить внятную ошибку,
        а не молча потерять зависящий от него признак."""
        registry.register_analyzer(
            make_analyzer(
                "speech.rate",
                requires={ARTIFACT_TRANSCRIPT},
                produces={"speech.rate.wpm"},
            )
        )
        registry.register_analyzer(make_analyzer("speech.delta", requires={"speech.rate.wpm"}))

        with pytest.raises(DependencyResolutionError):
            registry.build_plan(available_artifacts={ARTIFACT_TRANSCRIPT}, disabled={"speech.rate"})

    def test_empty_registry_gives_empty_plan(self) -> None:
        assert registry.build_plan(available_artifacts={ARTIFACT_VIDEO}) == []

    def test_explicit_subset(self) -> None:
        registry.register_analyzer(make_analyzer("a", requires={ARTIFACT_VIDEO}))
        registry.register_analyzer(make_analyzer("b", requires={ARTIFACT_VIDEO}))

        plan = registry.build_plan(["a"], available_artifacts={ARTIFACT_VIDEO})

        assert [c.key for c in plan[0]] == ["a"]


class TestExtensibility:
    def test_adding_analyzer_requires_no_core_changes(self) -> None:
        """Смысловой тест: новый плагин подключается одной регистрацией.

        Он ничего не наследует, живёт в чужом модуле и всё равно попадает
        в план исполнения. Это и есть заявленная расширяемость.
        """
        third_party = make_analyzer(
            "thirdparty.vibe", requires={ARTIFACT_TRANSCRIPT}, produces={"thirdparty.vibe.score"}
        )
        registry.register_analyzer(third_party)

        plan = registry.build_plan(available_artifacts={ARTIFACT_TRANSCRIPT})

        assert any(cls is third_party for layer in plan for cls in layer)
