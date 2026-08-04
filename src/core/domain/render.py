"""Типы этапа нарезки: чем описывается рендер и что он возвращает.

Отдельный модуль, а не продолжение ``analysis.py``. Там живут типы,
которыми обмениваются анализаторы и скоринг — то есть всё, что относится
к **решениям**. Здесь начинается работа с **файлами**, и смешивать это
в одном модуле означало бы, что сцену от готового mp4 отделяет только
соседняя строчка в ``__all__``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final
from uuid import UUID

from core.domain.values import CropBox

__all__ = [
    "VERTICAL_HEIGHT",
    "VERTICAL_WIDTH",
    "CropKeyframe",
    "RenderSpec",
    "RenderedClip",
]

#: Формат вертикального клипа. Константа, а не настройка: 1080×1920 —
#: это то, что принимают Shorts, Reels и TikTok, и «настраиваемая вертикаль»
#: означала бы возможность собрать клип, который никуда не загрузится.
VERTICAL_WIDTH: Final = 1080
VERTICAL_HEIGHT: Final = 1920


@dataclass(frozen=True, slots=True)
class RenderSpec:
    """Параметры рендера одного файла.

    Один объект на все три вида вывода (горизонталь, вертикаль, превью):
    кодек, качество и громкость у них общие, а различия — в размере кадра,
    и его задаёт вызывающий.
    """

    width: int | None = None
    height: int | None = None
    crf: int = 20
    preset: str = "medium"
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    #: Целевая громкость по EBU R128, LUFS. ``None`` — не трогать звук вовсе.
    #: −14 LUFS — то, к чему приводят звук YouTube и Spotify: клип, собранный
    #: тише, площадка не поднимет, а собранный громче — прижмёт лимитером.
    loudness_lufs: float | None = -14.0
    #: Потоки ffmpeg. 0 — пусть решает сам: он знает про доступные ядра
    #: больше, чем мы на момент написания конфига.
    threads: int = 0
    #: Готовить файл к потоковому воспроизведению (moov в начало). Стоит
    #: одного дополнительного прохода по файлу, но без этого браузер
    #: не начнёт играть, пока не скачает клип целиком.
    faststart: bool = True
    burn_subtitles_path: Path | None = None


@dataclass(frozen=True, slots=True)
class CropKeyframe:
    """Положение окна кадрирования в конкретный момент клипа.

    Траектория задаётся ключевыми точками, а не значением на каждый кадр:
    после сглаживания движение медленное, и хранить 25 одинаковых позиций
    в секунду незачем — между точками рендер интерполирует сам.
    """

    time_sec: float
    box: CropBox

    def __post_init__(self) -> None:
        if self.time_sec < 0:
            raise ValueError(f"Время ключевой точки отрицательное: {self.time_sec}")


@dataclass(frozen=True, slots=True)
class RenderedClip:
    """Что получилось из одного клипа.

    Ошибка — поле, а не исключение, потому что этап переживает падение
    отдельного клипа: девять готовых файлов ценнее, чем ноль из-за одного
    битого отрезка (док 04, §4.5).
    """

    clip_id: UUID
    horizontal: Path | None = None
    vertical: Path | None = None
    thumbnail: Path | None = None
    duration_sec: float = 0.0
    #: Как получена вертикаль: ``track`` — по траектории лиц, ``static`` —
    #: неподвижный кроп, ``blurpad`` — фоллбек с размытым фоном.
    #: Пишется в метаданные ассета: без этого «почему тут кадр не двигался»
    #: пришлось бы выяснять пересмотром файла.
    reframe_mode: str | None = None
    #: Файлы субтитров по форматам: ``{"srt": ..., "ass": ...}``.
    #: Словарём, а не списком: дальше по конвейеру их спрашивают
    #: по формату, а не перебирают.
    subtitles: Mapping[str, Path] = field(default_factory=dict)
    error: str | None = None
    warnings: Sequence[str] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.error is None and self.horizontal is not None
