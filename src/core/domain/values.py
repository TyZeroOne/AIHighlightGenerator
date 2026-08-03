"""Объекты-значения (value objects) домена.

Все — неизменяемые (``frozen=True``) и без внешних зависимостей.
Неизменяемость здесь не догма: отрезок времени, который кто-то может
поменять «на месте», расползётся по коду непредсказуемо, а так его можно
свободно передавать, класть в множества и использовать как ключ словаря.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

from core.domain.timecode import format_timecode

__all__ = ["CropBox", "FeatureValue", "MediaWindow", "Score", "TimeRange"]

# Допуск для сравнения секунд: 1 мкс. Кадр даже при 240 fps длиннее 4 мс,
# так что этот допуск заведомо меньше любой значимой величины.
EPSILON: Final = 1e-6


@dataclass(frozen=True, slots=True, order=True)
class TimeRange:
    """Полуинтервал ``[start, end)`` в секундах от начала медиа."""

    start: float
    end: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.start) or not math.isfinite(self.end):
            raise ValueError(f"Границы должны быть конечными числами: [{self.start}, {self.end}]")
        if self.start < 0:
            raise ValueError(f"Начало не может быть отрицательным: {self.start}")
        if self.end - self.start <= EPSILON:
            raise ValueError(f"Конец должен быть больше начала: [{self.start}, {self.end}]")

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2

    def contains(self, moment: float) -> bool:
        return self.start - EPSILON <= moment < self.end + EPSILON

    def overlaps(self, other: TimeRange) -> bool:
        return self.start < other.end - EPSILON and other.start < self.end - EPSILON

    def intersection(self, other: TimeRange) -> TimeRange | None:
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        return TimeRange(start, end) if end - start > EPSILON else None

    def union(self, other: TimeRange) -> TimeRange:
        """Объединяющий отрезок. Для непересекающихся включает промежуток между ними."""
        return TimeRange(min(self.start, other.start), max(self.end, other.end))

    def iou(self, other: TimeRange) -> float:
        """Intersection over Union — метрика похожести отрезков, 0..1.

        Используется в NMS (Этап 7) для отсева дублирующихся кандидатов.
        """
        overlap = self.intersection(other)
        if overlap is None:
            return 0.0
        union_duration = self.duration + other.duration - overlap.duration
        return overlap.duration / union_duration

    def shifted(self, delta: float) -> TimeRange:
        """Сдвиг на ``delta`` секунд. Нужен при переводе в координаты клипа."""
        return TimeRange(self.start + delta, self.end + delta)

    def expanded(self, before: float, after: float, *, limit: TimeRange | None = None) -> TimeRange:
        """Расширение границ с необязательным ограничением рамками ``limit``."""
        start = self.start - before
        end = self.end + after
        if limit is not None:
            start = max(start, limit.start)
            end = min(end, limit.end)
        return TimeRange(max(start, 0.0), end)

    def __str__(self) -> str:
        return f"{format_timecode(self.start)}–{format_timecode(self.end)}"


@dataclass(frozen=True, slots=True)
class MediaWindow:
    """Окно медиа, над которым работает конвейер.

    Ключевая абстракция всей системы. Для файла — одно окно на всю длительность
    с ``is_final=True``. Для стрима — бесконечная последовательность окон,
    у которых ``is_final=False``. Ни один анализатор не знает, что перед ним:
    файл или живой эфир, — поэтому поддержка YouTube/Twitch/Kick не потребует
    правок в конвейере (док 02, §2.6).
    """

    source_id: str
    range: TimeRange
    index: int = 0
    is_final: bool = True

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id не может быть пустым")
        if self.index < 0:
            raise ValueError(f"index не может быть отрицательным: {self.index}")


@dataclass(frozen=True, slots=True)
class FeatureValue:
    """Одно измеренное значение признака.

    ``confidence`` нужен, потому что признаки приходят из моделей разной
    надёжности: громкость измеряется точно (1.0), а «вероятность смеха» —
    это выход классификатора, которому можно верить лишь отчасти.
    """

    key: str
    value: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("key не может быть пустым")
        if not math.isfinite(self.value):
            raise ValueError(f"Значение признака {self.key!r} должно быть конечным: {self.value}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence должен быть в [0, 1], получено {self.confidence}")


@dataclass(frozen=True, slots=True)
class Score:
    """Оценка 0..100 вместе с разложением по группам признаков.

    ``breakdown`` — не украшение: без него невозможно ответить на вопрос
    «почему эта сцена набрала 96?», а значит невозможно отлаживать скоринг.
    """

    value: float
    breakdown: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 100.0:
            raise ValueError(f"Оценка должна быть в [0, 100], получено {self.value}")
        # Замораживаем словарь: иначе «неизменяемый» dataclass даёт изменяемое поле.
        object.__setattr__(self, "breakdown", MappingProxyType(dict(self.breakdown)))


@dataclass(frozen=True, slots=True)
class CropBox:
    """Прямоугольник кадрирования в пикселях. Нужен для вертикали 1080×1920 (Этап 8)."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError(f"Координаты не могут быть отрицательными: ({self.x}, {self.y})")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Размеры должны быть положительными: {self.width}×{self.height}")

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2
