"""Текстовые представления транскрипта: SRT, WebVTT, простой текст.

Здесь только сериализация того, что уже распознано: сегмент → строка с
тайм-кодами. Ни разбиения на строки по 42 символа, ни караоке-подсветки,
ни сдвига в координаты клипа — всё это Этап 9, и делается оно уже для клипа,
а не для целого видео.

Разделение не формальное. Транскрипт целого фильма нужен человеку и машине
как расшифровка: открыть в редакторе, прогнать поиском, скормить в LLM.
Субтитры клипа нужны зрителю на экране, и требования к ним другие —
длина строки, время показа, положение. Смешивать эти задачи в одном коде
означало бы получить формат, неудобный обеим.
"""

from __future__ import annotations

from collections.abc import Iterable

from core.domain.media import Transcript, TranscriptSegment

__all__ = ["to_srt", "to_text", "to_vtt"]


def to_srt(transcript: Transcript) -> str:
    """SubRip. Нумерация с единицы, запятая в дробной части — так требует формат."""
    blocks = [
        f"{index}\n{_stamp(segment.range.start, ',')} --> {_stamp(segment.range.end, ',')}\n"
        f"{segment.text.strip()}\n"
        for index, segment in enumerate(_speaking(transcript.segments), start=1)
    ]
    return "\n".join(blocks)


def to_vtt(transcript: Transcript) -> str:
    """WebVTT — тот же SRT, но с заголовком, точкой в дробной части и без нумерации.

    Именно он понимается браузером напрямую через ``<track>``, поэтому нужен
    отдельно от SRT, хотя разница между ними в три символа.
    """
    blocks = [
        f"{_stamp(segment.range.start, '.')} --> {_stamp(segment.range.end, '.')}\n"
        f"{segment.text.strip()}\n"
        for segment in _speaking(transcript.segments)
    ]
    return "WEBVTT\n\n" + "\n".join(blocks)


def to_text(transcript: Transcript) -> str:
    """Расшифровка без тайм-кодов — для чтения и для передачи в LLM."""
    return "\n".join(segment.text.strip() for segment in _speaking(transcript.segments))


def _speaking(segments: Iterable[TranscriptSegment]) -> list[TranscriptSegment]:
    return [segment for segment in segments if segment.text.strip()]


def _stamp(seconds: float, separator: str) -> str:
    """``921.48`` → ``00:15:21,480``.

    Свой форматтер, а не ``format_timecode`` из домена: там разделитель
    дробной части — точка, а SRT требует запятую, и плееры на этом строги.
    """
    total_millis = max(round(seconds * 1000), 0)
    millis = total_millis % 1000
    total_seconds = total_millis // 1000
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"
