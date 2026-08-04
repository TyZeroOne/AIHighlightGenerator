"""Динамика кадра: насколько картинка меняется и насколько она насыщена.

Три признака, и каждый ловит свой тип «интересного»:

* **hist_delta** — изменение распределения яркостей кадр к кадру. Растёт
  и на смене плана, и на движении, и на вспышке; это общий индикатор
  «в кадре что-то происходит»;
* **entropy** — энтропия распределения яркостей. Высокая у детализованной
  сцены (толпа, город), низкая у пустой (небо, стена, титры). Отличает
  насыщенный кадр от плоского;
* **brightness_delta** — резкие скачки средней яркости. Это взрывы, вспышки,
  выстрелы, включённый свет — события, которые почти всегда заметны зрителю.

Всё считается по уменьшенным кадрам в оттенках серого: цвет здесь не нужен,
а вчетверо меньше пикселей — вчетверо быстрее.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from core.domain.analysis import ARTIFACT_VIDEO, AnalysisContext
from core.domain.values import FeatureValue, TimeRange
from core.logging import get_logger
from core.registry import register_analyzer
from vision.context import source_for
from vision.frames import FrameSource

__all__ = ["DynamicsAnalyzer", "brightness_series", "entropy_of", "histogram_delta"]

log = get_logger(__name__)


def entropy_of(frame: Any) -> float:
    """Энтропия распределения яркостей кадра, биты.

    Ноль — все пиксели одинаковы (чёрный экран, титры на однотонном фоне).
    Восемь — равномерное распределение по всем 256 уровням, то есть предельно
    насыщенная картинка.
    """
    import numpy as np

    counts = np.bincount(np.asarray(frame, dtype=np.uint8).ravel(), minlength=256)
    total = counts.sum()
    if total == 0:
        return 0.0

    probabilities = counts[counts > 0] / total
    return round(float(-(probabilities * np.log2(probabilities)).sum()), 4)


def histogram_delta(previous: Any, current: Any) -> float:
    """Насколько распределение яркостей отличается от предыдущего кадра, 0..1.

    Метрика — половина суммы модулей разностей нормированных гистограмм.
    Ноль означает совпадение, единица — полное расхождение (нет ни одного
    общего уровня яркости).

    Гистограмма, а не попиксельная разность: последняя реагирует на любое
    смещение камеры, даже когда содержание кадра то же самое. Нам же нужна
    смена **содержания**, а движение меряет отдельный анализатор.
    """
    import numpy as np

    first = np.bincount(np.asarray(previous, dtype=np.uint8).ravel(), minlength=256)
    second = np.bincount(np.asarray(current, dtype=np.uint8).ravel(), minlength=256)
    first_total, second_total = first.sum(), second.sum()
    if first_total == 0 or second_total == 0:
        return 0.0

    difference = np.abs(first / first_total - second / second_total).sum() / 2
    return round(float(difference), 4)


def brightness_series(frames: Sequence[Any]) -> list[float]:
    """Средняя яркость каждого кадра, 0..1."""
    import numpy as np

    return [round(float(np.asarray(frame, dtype=np.float32).mean() / 255.0), 4) for frame in frames]


@register_analyzer
class DynamicsAnalyzer:
    """Реализация порта ``FeatureAnalyzer`` для динамики кадра."""

    key: ClassVar[str] = "vision.dynamics"
    requires: ClassVar[frozenset[str]] = frozenset({ARTIFACT_VIDEO})
    produces: ClassVar[frozenset[str]] = frozenset(
        {
            "vision.dynamics.hist_delta",
            "vision.dynamics.entropy",
            "vision.dynamics.brightness_delta",
        }
    )
    #: Дешёвый на фоне соседей: только арифметика по уже декодированным кадрам.
    cost: ClassVar[int] = 2

    def __init__(self, *, source: FrameSource | None = None) -> None:
        self._source = source

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
            if batch.is_empty:
                continue

            values = _scene_features(batch.frames)
            if values:
                features[index] = values

        log.debug("dynamics.done", scenes=len(features))
        return features


def _scene_features(frames: Sequence[Any]) -> list[FeatureValue]:
    import numpy as np

    entropies = [entropy_of(frame) for frame in frames]
    brightness = brightness_series(frames)

    values = [
        FeatureValue("vision.dynamics.entropy", round(float(np.mean(entropies)), 4)),
    ]

    if len(frames) < 2:
        # Одного кадра хватает на энтропию, но не на изменения: у «разности»
        # с самим собой нет смысла, и ноль здесь соврал бы про статичность.
        return values

    deltas = [histogram_delta(frames[i - 1], frames[i]) for i in range(1, len(frames))]
    jumps = [abs(brightness[i] - brightness[i - 1]) for i in range(1, len(brightness))]

    values.append(FeatureValue("vision.dynamics.hist_delta", round(float(np.mean(deltas)), 4)))
    # Максимум, а не среднее: вспышка — это одно событие в сцене, и среднее
    # по всем кадрам растворило бы её в спокойном фоне.
    values.append(FeatureValue("vision.dynamics.brightness_delta", round(max(jumps), 4)))
    return values
