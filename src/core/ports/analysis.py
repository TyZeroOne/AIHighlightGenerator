"""Порты анализа: плагин-анализатор признаков и стратегия скоринга."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar, Protocol

from core.domain.analysis import AnalysisContext, ScoringProfile
from core.domain.entities import Scene
from core.domain.values import FeatureValue, Score, TimeRange

__all__ = ["FeatureAnalyzer", "ScoringStrategy"]


class FeatureAnalyzer(Protocol):
    """Плагин, вычисляющий группу признаков.

    Четыре атрибута класса — это не метаданные «для красоты», а вход
    планировщика (``core.registry.build_plan``):

    * ``key``      — уникальное имя, по нему анализатор включают/отключают;
    * ``requires`` — какое сырьё нужно; определяет место в порядке исполнения;
    * ``produces`` — какие ключи признаков появятся; позволяет другим
                     анализаторам зависеть от результата этого;
    * ``cost``     — относительная стоимость; планировщик ставит дорогие
                     анализаторы в начало слоя, чтобы они не были последними
                     и не растягивали общее время.
    """

    key: ClassVar[str]
    requires: ClassVar[frozenset[str]]
    produces: ClassVar[frozenset[str]]
    cost: ClassVar[int]

    async def analyze(
        self,
        context: AnalysisContext,
        scenes: Sequence[TimeRange],
    ) -> Mapping[int, Sequence[FeatureValue]]:
        """Признаки по каждой сцене: ключ словаря — индекс в ``scenes``.

        Контракт: анализатор вправе вернуть признаки не для всех сцен
        (например, там, где нет речи). Отсутствующий признак — штатная
        ситуация, скоринг это учитывает.
        """
        ...


class ScoringStrategy(Protocol):
    async def score(
        self,
        scenes: Sequence[Scene],
        profile: ScoringProfile,
    ) -> Sequence[Score]:
        """Оценить сцены. Порядок результата совпадает с порядком входа."""
        ...
