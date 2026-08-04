"""Порты работы с медиафайлами: сцены, аудио, рендер, субтитры."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from core.domain.enums import SubtitleFormat
from core.domain.media import AudioTrack, LocalMedia, Transcript
from core.domain.render import CropKeyframe, RenderSpec
from core.domain.values import MediaWindow, TimeRange

__all__ = [
    "AudioExtractor",
    "ClipRenderer",
    "CropPlanner",
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
    async def cut(
        self,
        media: LocalMedia,
        span: TimeRange,
        spec: RenderSpec,
        destination: Path,
    ) -> Path:
        """Вырезать фрагмент. Контракт: длительность результата = span.duration ± 50 мс."""
        ...

    async def reframe_vertical(
        self,
        source: Path,
        track: Sequence[CropKeyframe],
        spec: RenderSpec,
        destination: Path,
    ) -> tuple[Path, str]:
        """Переформатировать в 1080×1920 по траектории кадрирования.

        Второй элемент ответа — как именно получился кадр: по траектории,
        неподвижным кропом или размытым фоном. Пустая траектория не ошибка:
        клип без единого лица — штатная ситуация, и рендер обязан отдать
        файл, а не исключение.
        """
        ...

    async def thumbnail(self, source: Path, at_sec: float, destination: Path) -> Path:
        """Кадр-превью клипа."""
        ...


class CropPlanner(Protocol):
    """Планировщик траектории кадрирования.

    Отдельный порт от рендера намеренно: считать, **куда** смотреть, —
    это компьютерное зрение (лица, сэлиенси), а применять решение —
    работа ffmpeg. Разные зависимости, разная стоимость, разные причины
    меняться; за одним интерфейсом они мешали бы друг другу.
    """

    async def plan(
        self,
        source: Path,
        *,
        target_width: int,
        target_height: int,
    ) -> Sequence[CropKeyframe]:
        """Траектория окна кадрирования по видеофайлу.

        Пустая последовательность означает «опоры для кадрирования нет» —
        вызывающий переходит на фоллбек, а не считает это сбоем.
        """
        ...


class SubtitleWriter(Protocol):
    def write(
        self,
        transcript: Transcript,
        span: TimeRange,
        fmt: SubtitleFormat,
        destination: Path,
        *,
        width: int | None = None,
        height: int | None = None,
    ) -> Path:
        """Записать субтитры для фрагмента.

        Тайминги сдвигаются в систему координат клипа: слово на 921.5 с
        исходника в клипе, начинающемся с 921.0, получает время 0.5 с.

        Размер кадра нужен только ASS: libass рисует в объявленной
        системе координат и масштабирует результат к реальному кадру,
        поэтому файл, собранный под горизонталь, в вертикали растянет
        текст поперёк. Для текстовых форматов размер игнорируется.
        """
        ...
