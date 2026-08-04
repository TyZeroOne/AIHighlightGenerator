"""Структурные признаки: то, что видно только из соседства сцен.

Все признаки Этапа 5 считались по одной сцене независимо. Эти четыре —
про место сцены в видео, и в отдельно взятой сцене их измерить нельзя.

| Признак | Что означает |
|---|---|
| ``struct.position`` | где сцена в видео: 0 — начало, 1 — конец |
| ``struct.duration`` | длительность относительно самой длинной сцены |
| ``struct.novelty`` | насколько сцена не похожа на соседей |
| ``struct.build_up`` | росла ли «энергия» на подходе к сцене |

Два последних несут основную нагрузку.

**Новизна** — главный маркер поворота. Резкая смена того, как выглядит
и звучит сцена, по сравнению с предыдущими — это либо новая линия,
либо неожиданность; и то и другое интересно.

**Нарастание** ловит кульминацию. Момент, к которому вело несколько
сцен подряд с растущей динамикой, ценнее такого же по абсолютным
признакам, но случившегося на ровном месте.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Final

__all__ = ["STRUCT_KEYS", "compute_structure"]

STRUCT_KEYS: Final = (
    "struct.position",
    "struct.duration",
    "struct.novelty",
    "struct.build_up",
)

#: Сколько предыдущих сцен смотреть при оценке новизны и нарастания.
#: Три — примерно полминуты материала на типичной нарезке: достаточно,
#: чтобы увидеть тенденцию, и мало, чтобы она не размылась.
_WINDOW: Final = 3

#: Признаки, по которым сравниваются сцены. Намеренно узкий набор: сюда
#: входит то, что описывает «характер» сцены, а не её содержание.
_PROFILE_KEYS: Final = (
    "audio.loudness.mean",
    "audio.pause.ratio",
    "speech.density",
    "speech.rate.wpm",
    "vision.motion.mean",
    "vision.cuts.rate",
    "vision.faces.count_mean",
    "vision.dynamics.entropy",
)

#: Признаки, из которых складывается «энергия» сцены для оценки нарастания.
_ENERGY_KEYS: Final = (
    "audio.loudness.mean",
    "speech.emotion.arousal",
    "vision.motion.mean",
    "vision.cuts.rate",
)


def compute_structure(
    normalized: Sequence[Mapping[str, float]],
    durations: Sequence[float],
) -> list[dict[str, float]]:
    """Структурные признаки для всех сцен видео.

    На вход — уже нормализованные признаки: новизна считается как расстояние
    между векторами сцен, и без общей шкалы её значение определялось бы
    единицами измерения, а не различием сцен.
    """
    count = len(normalized)
    if count == 0:
        return []

    longest = max(durations) if durations else 1.0
    result: list[dict[str, float]] = []

    for index in range(count):
        values: dict[str, float] = {}

        # Позиция: на одной сцене делить не на что — считаем её началом.
        values["struct.position"] = index / (count - 1) if count > 1 else 0.0
        values["struct.duration"] = (
            min(durations[index] / longest, 1.0) if longest > 0 and index < len(durations) else 0.0
        )
        values["struct.novelty"] = _novelty(normalized, index)
        values["struct.build_up"] = _build_up(normalized, index)

        result.append({key: round(value, 4) for key, value in values.items()})

    return result


def _novelty(scenes: Sequence[Mapping[str, float]], index: int) -> float:
    """Насколько сцена не похожа на предыдущие.

    Косинусное расстояние до среднего вектора окна. Косинус, а не евклид:
    важна не «сила» сцены, а направление её характера — тихий диалог после
    громкого диалога это не новизна, а тихий диалог после погони — да.
    """
    if index == 0:
        # Первой сцене не с чем сравниваться. Ноль, а не единица: начало
        # видео не является поворотом по построению.
        return 0.0

    current = _vector(scenes[index])
    window = [_vector(scenes[i]) for i in range(max(0, index - _WINDOW), index)]
    if not window or not any(current):
        return 0.0

    reference = [sum(column) / len(window) for column in zip(*window, strict=True)]
    similarity = _cosine(current, reference)
    return max(0.0, min(1.0, 1.0 - similarity))


def _build_up(scenes: Sequence[Mapping[str, float]], index: int) -> float:
    """Росла ли энергия на подходе к сцене.

    Сравнивается энергия самой сцены со средней энергией окна перед ней.
    Падение даёт ноль, а не отрицательное значение: спад — это не «анти-
    кульминация», а просто отсутствие нарастания.
    """
    if index == 0:
        return 0.0

    window = [_energy(scenes[i]) for i in range(max(0, index - _WINDOW), index)]
    if not window:
        return 0.0

    before = sum(window) / len(window)
    growth = _energy(scenes[index]) - before
    # Нормируем на 0.5: рост энергии на половину шкалы — это уже
    # выраженная кульминация, дальше насыщение.
    return max(0.0, min(1.0, growth / 0.5))


def _vector(features: Mapping[str, float]) -> list[float]:
    return [float(features.get(key, 0.0)) for key in _PROFILE_KEYS]


def _energy(features: Mapping[str, float]) -> float:
    present = [features[key] for key in _ENERGY_KEYS if key in features]
    return sum(present) / len(present) if present else 0.0


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left <= 0 or norm_right <= 0:
        return 1.0
    return dot / (norm_left * norm_right)
