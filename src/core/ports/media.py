"""Порты работы с медиафайлами: сцены, аудио, рендер, субтитры."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from core.domain.analysis import RenderSpec
from core.domain.enums import SubtitleFormat
from core.domain.media import AudioTrack, LocalMedia, Transcript
from core.domain.values import CropBox, MediaWindow, TimeRange

__all__ = [
    "AudioExtractor",
    "ClipRenderer",
    "KeyframeExtractor",
    "SceneDetector",
    "SubtitleWriter",
]


class SceneDetector(Protocol):
    """Разбиение видео на сцены.

    Отдельный порт, а не функция, потому что реализаций планируется минимум две
    (PySceneDetect и FFmpeg ``scdet``), и выбор делается конфигом.
    """

    async def detect(self, media: LocalMedia, window: MediaWindow) -> Sequence[TimeRange]:
        """Границы сцен. Контракт: покрывают окно целиком, без дыр и пересечений."""
        ...


class KeyframeExtractor(Protocol):
    """Извлечение отдельных кадров по временным меткам.

    Метод принимает **список** моментов, а не один: кадров бывает столько же,
    сколько сцен (на двухчасовом фильме — за тысячу), и реализация обязана
    иметь возможность тянуть их параллельно. Порт с одиночным кадром такой
    возможности не оставляет — вызывающему пришлось бы городить пул самому.
    """

    async def extract(
        self,
        media: LocalMedia,
        moments: Sequence[float],
        destination: Path,
    ) -> Sequence[Path | None]:
        """Кадры для каждого момента, по индексам входа.

        ``None`` на позиции означает «кадр получить не удалось». Это не ошибка
        этапа: превью — вспомогательный артефакт, и один битый кадр в конце
        файла не повод терять результат разбиения.
        """
        ...


class AudioExtractor(Protocol):
    async def extract(self, media: LocalMedia, window: MediaWindow) -> AudioTrack:
        """Извлечь аудио в формате, пригодном для распознавания (WAV 16 kHz mono)."""
        ...


class ClipRenderer(Protocol):
    async def cut(self, media: LocalMedia, span: TimeRange, spec: RenderSpec) -> Path:
        """Вырезать фрагмент. Контракт: длительность результата = span.duration ± 50 мс."""
        ...

    async def reframe_vertical(self, source: Path, track: Sequence[CropBox]) -> Path:
        """Переформатировать в 1080×1920 по траектории кадрирования."""
        ...


class SubtitleWriter(Protocol):
    def write(
        self,
        transcript: Transcript,
        span: TimeRange,
        fmt: SubtitleFormat,
        destination: Path,
    ) -> Path:
        """Записать субтитры для фрагмента.

        Тайминги сдвигаются в систему координат клипа: слово на 921.5 с
        исходника в клипе, начинающемся с 921.0, получает время 0.5 с.
        """
        ...
