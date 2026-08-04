"""Порты скоринга и отбора моментов.

Два порта, потому что этапа два и они разделены LLM-этапом: сначала
считается эвристика (её видит модель), потом — уже с вердиктами модели —
идёт отбор.

Сценарии знают только эти два интерфейса. Как именно нормализуются
признаки, из чего складываются сигналы, каким алгоритмом подавляются
пересечения — детали реализации в ``highlight_detector``, и меняются они
по другим поводам, чем место этапа в конвейере.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from core.domain.entities import Clip, Highlight, Scene
from core.domain.values import TimeRange

__all__ = ["HighlightSelector", "SceneScorer"]


class SceneScorer(Protocol):
    def score(self, scenes: Sequence[Scene], *, profile: str) -> Sequence[Scene]:
        """Посчитать сигналы и эвристический скор.

        Возвращает **обновлённые копии** сцен: с дописанными признаками
        ``signal.*`` и ``struct.*`` и заполненным ``heuristic_score``.
        Существующие признаки при этом сохраняются — этапы пишут в одну
        колонку по очереди.
        """
        ...


class HighlightSelector(Protocol):
    def select(
        self,
        scenes: Sequence[Scene],
        *,
        profile: str,
        pauses: Sequence[TimeRange] = (),
        total_duration_sec: float | None = None,
    ) -> Sequence[Highlight]:
        """Отобрать финальные моменты из оценённых сцен."""
        ...

    def fuse_scores(self, scenes: Sequence[Scene]) -> Sequence[Scene]:
        """Свести эвристику с оценкой модели в final_score.

        Отдельный шаг, потому что результат сохраняется: итоговый скор
        показывается в API и объясняет порядок клипов.
        """
        ...

    def to_clips(
        self,
        highlights: Sequence[Highlight],
        *,
        job_id: object,
        video_id: object,
    ) -> Sequence[Clip]:
        """Моменты → записи клипов (без файлов; файлы появятся на Этапе 8)."""
        ...
