"""SubRip для клипа: реплики → файл, который понимают все площадки.

Формат тривиален, и почти вся его сложность — в мелочах, на которых
плееры строги: запятая в дробной части, нумерация с единицы, пустая
строка между блоками, перевод строки в конце файла.

Отдельно от ``formats.py``: там сериализуется транскрипт целого видео
в исходных координатах, здесь — готовые реплики клипа, уже разбитые
по строкам и сдвинутые к нулю.
"""

from __future__ import annotations

from collections.abc import Sequence

from subtitles.cues import Cue

__all__ = ["render_srt", "render_vtt", "srt_timestamp"]


def render_srt(cues: Sequence[Cue]) -> str:
    """Реплики → текст файла ``.srt``.

    Пустой список даёт пустую строку, а не заголовок без блоков: SRT
    без единого блока — валидный пустой файл, и плеер на нём не спотыкается.
    """
    blocks = [
        f"{index}\n"
        f"{srt_timestamp(cue.range.start)} --> {srt_timestamp(cue.range.end)}\n"
        f"{cue.text}\n"
        for index, cue in enumerate(cues, start=1)
    ]
    return "\n".join(blocks)


def render_vtt(cues: Sequence[Cue]) -> str:
    """Тот же файл для браузера.

    WebVTT — единственный формат, который ``<video><track>`` понимает
    без библиотек. Отличий от SRT три: заголовок, точка вместо запятой
    и отсутствие нумерации; ради них держать отдельный модуль не стоит.
    """
    blocks = [
        f"{srt_timestamp(cue.range.start).replace(',', '.')} --> "
        f"{srt_timestamp(cue.range.end).replace(',', '.')}\n"
        f"{cue.text}\n"
        for cue in cues
    ]
    return "WEBVTT\n\n" + "\n".join(blocks)


def srt_timestamp(seconds: float) -> str:
    """``4.29`` → ``00:00:04,290``.

    Запятая, а не точка: с точкой файл открывается не везде, и это как раз
    та мелочь, из-за которой субтитры «не работают» без единого сообщения
    об ошибке.
    """
    total_millis = max(round(seconds * 1000), 0)
    millis = total_millis % 1000
    total_seconds = total_millis // 1000
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
