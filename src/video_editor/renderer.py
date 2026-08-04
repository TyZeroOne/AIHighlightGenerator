"""Сборка порта ``ClipRenderer`` из трёх независимых кусков.

Резка, вертикаль и превью реализованы отдельными классами: у них разные
входы (исходник против готового клипа), разная стоимость и разные поводы
меняться. Порт же по ту сторону нужен один — сценарию этапа незачем знать,
что внутри их три.

Это не «класс ради класса»: тонкая склейка здесь именно тот случай, когда
объект оправдан — он держит общие настройки (пути к бинарникам, размер
вертикали) и избавляет контейнер от сборки трёх зависимостей на каждый
вызов.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from core.domain.media import LocalMedia
from core.domain.render import VERTICAL_HEIGHT, VERTICAL_WIDTH, CropKeyframe, RenderSpec
from core.domain.values import TimeRange
from video_editor.cutter import CutResult, FFmpegClipCutter
from video_editor.encoder import thumbnail_args
from video_editor.process import run_ffmpeg
from video_editor.vertical import FFmpegVerticalRenderer

__all__ = ["THUMBNAIL_TIMEOUT_SEC", "FFmpegClipRenderer"]

#: Один кадр из короткого файла — работа на доли секунды. Минуты хватает
#: с запасом на любой разумный клип, а зависший процесс не встанет намертво.
THUMBNAIL_TIMEOUT_SEC: Final = 60.0


class FFmpegClipRenderer:
    """Реализация порта ``ClipRenderer`` целиком."""

    def __init__(
        self,
        *,
        binary: str = "ffmpeg",
        probe_binary: str = "ffprobe",
        thumbnail_width: int = 640,
        allow_stream_copy: bool = False,
        vertical_width: int = VERTICAL_WIDTH,
        vertical_height: int = VERTICAL_HEIGHT,
    ) -> None:
        self._binary = binary
        self._thumbnail_width = thumbnail_width
        self._cutter = FFmpegClipCutter(
            binary=binary,
            probe_binary=probe_binary,
            allow_stream_copy=allow_stream_copy,
        )
        self._vertical = FFmpegVerticalRenderer(
            binary=binary,
            probe_binary=probe_binary,
            width=vertical_width,
            height=vertical_height,
        )

    async def cut(
        self,
        media: LocalMedia,
        span: TimeRange,
        spec: RenderSpec,
        destination: Path,
    ) -> Path:
        return await self._cutter.cut(media, span, spec, destination)

    async def cut_detailed(
        self,
        media: LocalMedia,
        span: TimeRange,
        spec: RenderSpec,
        destination: Path,
    ) -> CutResult:
        return await self._cutter.cut_detailed(media, span, spec, destination)

    async def reframe_vertical(
        self,
        source: Path,
        track: Sequence[CropKeyframe],
        spec: RenderSpec,
        destination: Path,
    ) -> tuple[Path, str]:
        return await self._vertical.reframe_vertical(source, track, spec, destination)

    async def thumbnail(self, source: Path, at_sec: float, destination: Path) -> Path:
        """Кадр-превью клипа.

        Момент выбирает вызывающий. Разумное значение — треть клипа:
        начало часто занято остатком предыдущей склейки, а самый конец
        клипа — уходом в затемнение.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        return await run_ffmpeg(
            self._binary,
            thumbnail_args(source, at_sec, self._thumbnail_width, destination),
            destination=destination,
            action="превью клипа",
            timeout_sec=THUMBNAIL_TIMEOUT_SEC,
        )
