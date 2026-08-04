"""Реализация порта ``SubtitleWriter``: транскрипт + границы клипа → файл.

Тонкая склейка: разбиение на реплики делает ``cues``, сериализацию —
``srt`` и ``ass``. Здесь только выбор формата и запись на диск.

Отдельный класс, а не функция, потому что настройки разбиения
(длина строки, скорость чтения, стиль) приходят из конфига один раз
на процесс, а вызовов — по одному на клип и формат.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from core.domain.enums import SubtitleFormat
from core.domain.errors import ValidationError
from core.domain.media import Transcript
from core.domain.values import TimeRange
from core.logging import get_logger
from subtitles.ass import AssStyle, chars_that_fit, render_ass
from subtitles.cues import Cue, CueSettings, build_cues
from subtitles.srt import render_srt, render_vtt

__all__ = ["DEFAULT_HEIGHT", "DEFAULT_WIDTH", "ClipSubtitleWriter"]

log = get_logger(__name__)

#: Размер кадра, под который собирается ASS, если вызывающий не сказал
#: иначе. Горизонталь 1080p — то, чем клип является по умолчанию.
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080


class ClipSubtitleWriter:
    """Субтитры одного клипа в одном формате."""

    def __init__(
        self,
        *,
        settings: CueSettings | None = None,
        style: AssStyle | None = None,
        karaoke: bool = True,
    ) -> None:
        self._settings = settings or CueSettings()
        self._style = style or AssStyle()
        self._karaoke = karaoke

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
        destination.parent.mkdir(parents=True, exist_ok=True)

        if fmt is SubtitleFormat.SRT:
            body = render_srt(build_cues(transcript, span, self._settings))
        elif fmt is SubtitleFormat.VTT:
            body = render_vtt(build_cues(transcript, span, self._settings))
        elif fmt is SubtitleFormat.ASS:
            frame_width = width or DEFAULT_WIDTH
            frame_height = height or DEFAULT_HEIGHT
            body = render_ass(
                self._cues_for_frame(transcript, span, frame_width, frame_height),
                width=frame_width,
                height=frame_height,
                style=self._style,
                karaoke=self._karaoke,
            )
        else:  # pragma: no cover — enum закрыт, ветка на случай расширения
            raise ValidationError(f"Неизвестный формат субтитров: {fmt}", format=str(fmt))

        # UTF-8 без BOM и с переводами строк LF. Windows-плееры с этим
        # справляются, а вот обратное — CRLF в ASS — libass местами
        # разбирает как часть текста.
        destination.write_text(body, encoding="utf-8", newline="\n")
        log.debug("subtitles.written", format=fmt.value, file=destination.name)
        return destination

    def _cues_for_frame(
        self,
        transcript: Transcript,
        span: TimeRange,
        width: int,
        height: int,
    ) -> list[Cue]:
        """Реплики под конкретный кадр.

        Длина строки берётся как минимум из двух: стандарта вещания
        (42 символа) и того, что физически влезает в кадр при выбранном
        кегле. Второе ограничение начинает действовать на вертикали,
        где шрифт крупнее, а места вдвое меньше.
        """
        fits = chars_that_fit(width, height, self._style)
        settings = self._settings
        if fits < settings.max_chars:
            settings = replace(settings, max_chars=fits)
        return build_cues(transcript, span, settings)
