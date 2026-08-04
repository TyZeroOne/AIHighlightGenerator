"""Монтажные склейки внутри сцены.

Казалось бы, противоречие: сцена по определению — это кусок между склейками,
откуда внутри неё склейки? Оттуда, что детектор сцен (Этап 3) склеивает
короткие куски: пачка из пяти планов по полсекунды становится одной сценой
длиной 2.5 с. Информация о том, что внутри был быстрый монтаж, при этом
теряется — а это как раз маркер экшена.

Здесь она восстанавливается: считается, сколько раз внутри сцены картинка
сменилась резко. ``cuts.rate`` — склеек в секунду — один из самых прямых
признаков «драки» и «погони» (док 05, §5.2).

Детектор отдельный и простой, а не PySceneDetect: тому нужен свой проход
по файлу, а здесь кадры уже декодированы соседним анализатором и лежат
в общем источнике.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Final

from core.domain.analysis import ARTIFACT_VIDEO, AnalysisContext
from core.domain.values import FeatureValue, TimeRange
from core.logging import get_logger
from core.registry import register_analyzer
from vision.context import source_for
from vision.dynamics import histogram_delta
from vision.frames import FrameSource

__all__ = ["CutsAnalyzer", "count_cuts"]

log = get_logger(__name__)

#: Порог различия гистограмм, выше которого считаем, что произошла склейка.
#: Подобран по смыслу метрики: 0.4 означает, что распределение яркостей
#: изменилось почти наполовину — плавным движением такого не добиться.
DEFAULT_CUT_THRESHOLD: Final = 0.4


def count_cuts(frames: Sequence[Any], threshold: float = DEFAULT_CUT_THRESHOLD) -> int:
    """Число резких смен картинки в последовательности кадров.

    Гистограммы, а не попиксельная разность: последняя срабатывает
    на панораме, где содержание кадра не меняется вовсе.
    """
    if len(frames) < 2:
        return 0

    return sum(
        1
        for index in range(1, len(frames))
        if histogram_delta(frames[index - 1], frames[index]) >= threshold
    )


@register_analyzer
class CutsAnalyzer:
    """Реализация порта ``FeatureAnalyzer`` для склеек внутри сцены."""

    key: ClassVar[str] = "vision.cuts"
    requires: ClassVar[frozenset[str]] = frozenset({ARTIFACT_VIDEO})
    produces: ClassVar[frozenset[str]] = frozenset({"vision.cuts.count", "vision.cuts.rate"})
    cost: ClassVar[int] = 2

    def __init__(
        self,
        *,
        source: FrameSource | None = None,
        threshold: float = DEFAULT_CUT_THRESHOLD,
    ) -> None:
        self._source = source
        self._threshold = threshold

    async def analyze(
        self,
        context: AnalysisContext,
        scenes: Sequence[TimeRange],
    ) -> Mapping[int, Sequence[FeatureValue]]:
        source = self._source or source_for(context)
        if source is None or not scenes:
            return {}

        features: dict[int, list[FeatureValue]] = {}
        for index, scene in enumerate(scenes):
            batch = await source.scene_frames(scene)
            if len(batch) < 2:
                continue

            cuts = count_cuts(batch.frames, self._threshold)
            features[index] = [
                FeatureValue("vision.cuts.count", float(cuts)),
                FeatureValue("vision.cuts.rate", round(cuts / scene.duration, 4)),
            ]

        log.debug("cuts.done", scenes=len(features))
        return features
