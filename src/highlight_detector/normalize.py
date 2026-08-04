"""Приведение признаков к сопоставимой шкале.

Признаки Этапа 5 живут в несовместимых единицах: громкость около −23 LUFS,
оптический поток около 0.01, лиц — штуки, темп речи — сотни слов в минуту.
Складывать их в один скор без нормализации бессмысленно: слагаемое в сотнях
раздавит слагаемое в сотых, и «важность» признака определится случайно —
единицами измерения, а не смыслом.

## Внутри видео, а не глобально

Ключевое решение этого модуля. «Громко» в фильме ужасов и в лекции —
разные вещи; сцена на −18 LUFS в тихом подкасте это крик, а в боевике
рядовой диалог. Поэтому шкала строится по самому видео: важно не абсолютное
значение, а насколько сцена выделяется **среди своих**.

## Робастность

Медиана и межквартильный размах вместо среднего и стандартного отклонения.
Причина прикладная: во всяком видео есть выброс — одна вспышка, один взрыв,
один щелчок микрофона. Среднее и σ он сдвигает так, что все остальные сцены
сжимаются в узкую полосу около нуля, и различать их становится нечем.
Медиана и IQR его игнорируют.

## Короткое видео

На восьми сценах медиана и IQR — это статистика по восьми числам,
то есть почти шум. Для таких видео нормализация переключается на
преднастроенные глобальные шкалы: хуже, чем по своему видео, но честнее,
чем доверять выборке из восьми точек.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Final

__all__ = [
    "GLOBAL_SCALES",
    "MIN_SCENES_FOR_STATS",
    "modified_zscore",
    "normalize_feature",
    "normalize_scenes",
    "robust_normalize",
]

#: Меньше этого числа сцен — своя статистика ненадёжна (док 05, §5.3).
MIN_SCENES_FOR_STATS: Final = 8

#: На сколько IQR обрезать выбросы перед масштабированием. Три — компромисс:
#: полтора отсекли бы настоящие всплески (а они и есть хайлайты), пять
#: не защищают ни от чего.
_CLIP_IQR: Final = 3.0

#: Константа перевода MAD в оценку стандартного отклонения для нормального
#: распределения. Не магическое число: 1 / Φ⁻¹(0.75).
_MAD_SCALE: Final = 1.4826

#: Шкалы для коротких видео: (нижняя граница, верхняя граница) в сырых
#: единицах признака. Значения — из замеров Этапа 5 на реальном материале.
GLOBAL_SCALES: Final[Mapping[str, tuple[float, float]]] = {
    "audio.loudness.mean": (-40.0, -10.0),
    "audio.loudness.delta": (-6.0, 6.0),
    "audio.loudness.peak": (-20.0, 0.0),
    "audio.loudness.dynamic_range": (0.0, 8.0),
    "audio.pause.total": (0.0, 5.0),
    "audio.pause.max": (0.0, 3.0),
    "audio.pause.ratio": (0.0, 0.5),
    "speech.rate.wpm": (60.0, 200.0),
    "speech.rate.delta": (-60.0, 60.0),
    "speech.density": (0.0, 1.0),
    "speech.dialogue_turns": (0.0, 5.0),
    "vision.motion.mean": (0.0, 0.08),
    "vision.motion.peak": (0.0, 0.20),
    "vision.cuts.rate": (0.0, 1.0),
    "vision.camera.pan": (0.0, 0.15),
    "vision.camera.zoom": (0.0, 0.10),
    "vision.camera.shake": (0.0, 0.40),
    "vision.faces.count_max": (0.0, 5.0),
    "vision.faces.count_mean": (0.0, 3.0),
    "vision.faces.area_ratio": (0.0, 0.5),
    "vision.people.count": (0.0, 6.0),
    "vision.dynamics.hist_delta": (0.0, 0.3),
    "vision.dynamics.entropy": (4.0, 8.0),
    "vision.dynamics.brightness_delta": (0.0, 0.5),
}

#: Признаки, уже живущие в 0..1: нормализовать их по видео вредно.
#: Вероятность смеха 0.9 означает «смех есть» независимо от того, смеются
#: ли в остальном видео; растянув шкалу, мы превратили бы 0.02 в единицу
#: на видео, где никто не смеётся.
_ALREADY_UNIT: Final = frozenset(
    {
        "audio.event.laughter",
        "audio.event.scream",
        "audio.event.applause",
        "audio.event.music",
        "audio.event.impact",
        "speech.emotion.arousal",
        "speech.emotion.neu",
        "speech.emotion.hap",
        "speech.emotion.ang",
        "speech.emotion.sad",
        "vision.faces.closeup",
    }
)


def robust_normalize(values: Sequence[float]) -> list[float]:
    """Значения → 0..1 по медиане и межквартильному размаху.

    Порядок операций: центрируем по медиане, обрезаем по ±3 IQR, затем
    растягиваем в [0, 1]. Обрезка стоит **до** масштабирования — иначе
    единственный выброс определит верхнюю границу, и все настоящие данные
    окажутся в нижней десятой части шкалы.
    """
    if not values:
        return []
    if len(values) == 1:
        # Одна сцена: сравнивать не с чем, любое значение одинаково «среднее».
        return [0.5]

    median = statistics.median(values)
    spread = _iqr(values)

    if spread <= 0:
        # Все значения одинаковы — различать нечего. Половина, а не ноль:
        # ноль означал бы «признак отсутствует», а он есть и одинаков.
        return [0.5] * len(values)

    clipped = [
        max(min(value, median + _CLIP_IQR * spread), median - _CLIP_IQR * spread)
        for value in values
    ]
    low, high = min(clipped), max(clipped)
    if high - low <= 0:  # pragma: no cover — защита от вырожденного случая
        return [0.5] * len(values)
    return [(value - low) / (high - low) for value in clipped]


def modified_zscore(values: Sequence[float]) -> list[float]:
    """Насколько значение аномально — через медиану абсолютных отклонений.

    Нужен там, где важна не величина, а необычность: скачок громкости
    относительно медианы видео, срыв темпа речи. Обычный z-score здесь
    не годится по той же причине, по какой не годится среднее, —
    один выброс делает все остальные значения «нормальными».
    """
    if not values:
        return []
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    mad = statistics.median(deviations)
    if mad <= 0:
        return [0.0] * len(values)
    return [_MAD_SCALE * (value - median) / mad for value in values]


def normalize_feature(
    key: str,
    values: Sequence[float],
    *,
    use_global: bool,
) -> list[float]:
    """Нормализация одного признака по всем сценам видео."""
    if key in _ALREADY_UNIT:
        return [max(0.0, min(1.0, value)) for value in values]

    if use_global:
        scale = GLOBAL_SCALES.get(key)
        if scale is not None:
            low, high = scale
            span = high - low
            return [max(0.0, min(1.0, (value - low) / span)) for value in values]
        # Незнакомый признак на коротком видео: развернуть его не по чему,
        # и выдумывать шкалу опаснее, чем признать неопределённость.
        return [0.5] * len(values)

    return robust_normalize(values)


def normalize_scenes(
    features_per_scene: Sequence[Mapping[str, float]],
    *,
    scene_count: int | None = None,
) -> list[dict[str, float]]:
    """Признаки всех сцен видео → нормализованные признаки.

    Работает по колонкам: каждый признак нормализуется отдельно по всем
    сценам, где он есть. Сцены без признака его и не получают — «нет
    данных» и «ноль» это разные вещи, и подставлять ноль значило бы
    штрафовать сцену за то, что анализатор был отключён.
    """
    count = scene_count if scene_count is not None else len(features_per_scene)
    use_global = count < MIN_SCENES_FOR_STATS

    keys = {key for features in features_per_scene for key in features}
    normalized: list[dict[str, float]] = [{} for _ in features_per_scene]

    for key in sorted(keys):
        present = [
            (index, features[key])
            for index, features in enumerate(features_per_scene)
            if key in features
        ]
        scaled = normalize_feature(key, [value for _, value in present], use_global=use_global)
        for (index, _), value in zip(present, scaled, strict=True):
            normalized[index][key] = value

    return normalized


def _iqr(values: Sequence[float]) -> float:
    """Межквартильный размах.

    Считается вручную, а не через ``statistics.quantiles``: на выборках
    меньше четырёх точек тот бросает исключение, а у нас видео с тремя
    сценами — штатная ситуация.
    """
    ordered = sorted(values)
    size = len(ordered)
    if size < 4:
        return ordered[-1] - ordered[0]

    def percentile(fraction: float) -> float:
        position = fraction * (size - 1)
        low = int(position)
        high = min(low + 1, size - 1)
        weight = position - low
        return ordered[low] * (1 - weight) + ordered[high] * weight

    return percentile(0.75) - percentile(0.25)
