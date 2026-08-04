"""Реплики субтитров: из транскрипта видео — в текст на экране клипа.

Между «распознанной речью» и «субтитрами» лежит работа, которую легко
недооценить. Транскрипт — это поток слов с таймингами. Субтитры — это
кадры текста, каждый из которых человек должен успеть прочитать, не теряя
из виду картинку.

## Четыре ограничения, и все они про читаемость

**Длина строки.** 42 символа — стандарт вещания (BBC, Netflix, EBU-TT).
Он взят не из головы: строка длиннее заставляет глаз двигаться поперёк
кадра, и зритель перестаёт видеть, что происходит на экране.

**Число строк.** Две. Три строки закрывают треть вертикального кадра
и почти всегда означают, что реплику надо было разбить по времени.

**Время показа.** Снизу — чтобы успеть прочитать: считаем по скорости
чтения (символов в секунду) и не опускаемся ниже секунды даже для
одного слова. Сверху — чтобы текст не висел в тишине после того, как
фраза кончилась.

**Границы фраз.** Разрыв между репликами ставится там, где кончается
предложение или возникает пауза, а не там, где кончились 42 символа.
Реплика, разорванная посреди придаточного, читается вдвое дольше.

## Почему координаты клипа, а не видео

Слово на 921.5 секунде исходника в клипе, начинающемся с 921.0, должно
показаться на 0.5 секунде. Плеер ничего не знает про исходник — для него
файл начинается с нуля. Сдвиг делается здесь, один раз, и дальше форматы
работают уже с готовыми числами.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

from core.domain.media import Transcript, TranscriptWord
from core.domain.values import TimeRange

__all__ = [
    "Cue",
    "CueSettings",
    "build_cues",
    "split_into_cues",
    "wrap_lines",
]

#: Стандарт вещания: столько символов помещается в строку, не заставляя
#: глаз уходить от центра кадра.
DEFAULT_MAX_CHARS: Final = 42

#: Две строки. Третья закрывает треть вертикального кадра.
DEFAULT_MAX_LINES: Final = 2

#: Скорость чтения, символов в секунду. 17 — принятая для субтитров
#: величина: медленнее скучно, быстрее не успевает заметная часть зрителей.
DEFAULT_READING_SPEED: Final = 17.0

#: Реплика короче не показывается никогда, даже если это одно слово:
#: вспышка текста на полсекунды читается хуже, чем не читается вовсе.
DEFAULT_MIN_DURATION: Final = 1.0

#: И не дольше: текст, висящий после конца фразы, сбивает с толку.
DEFAULT_MAX_DURATION: Final = 6.0

#: Пауза между словами, после которой начинается новая реплика.
DEFAULT_GAP: Final = 0.7

#: Знаки конца предложения — естественные точки разрыва.
_SENTENCE_END: Final = re.compile(r"[.!?…]+[\"'»)]*$")


@dataclass(frozen=True, slots=True)
class CueSettings:
    max_chars: int = DEFAULT_MAX_CHARS
    max_lines: int = DEFAULT_MAX_LINES
    reading_speed: float = DEFAULT_READING_SPEED
    min_duration_sec: float = DEFAULT_MIN_DURATION
    max_duration_sec: float = DEFAULT_MAX_DURATION
    gap_sec: float = DEFAULT_GAP

    @property
    def max_length(self) -> int:
        """Сколько символов вообще помещается на экран одновременно."""
        return self.max_chars * self.max_lines


@dataclass(frozen=True, slots=True)
class Cue:
    """Один кадр субтитров в координатах клипа."""

    range: TimeRange
    lines: tuple[str, ...]
    #: Слова с таймингами — для караоке-подсветки в ASS. Для SRT не нужны,
    #: но хранятся здесь, а не пересчитываются вторым проходом: разбиение
    #: на реплики уже сделано, и повторять его ради другого формата значило
    #: бы получить два разных разбиения одного текста.
    words: tuple[TranscriptWord, ...] = field(default_factory=tuple)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def build_cues(
    transcript: Transcript,
    span: TimeRange,
    settings: CueSettings | None = None,
) -> list[Cue]:
    """Транскрипт видео + границы клипа → реплики в координатах клипа.

    Чистая функция: ни файлов, ни времени, ни случайности. Всё поведение
    субтитров проверяется на выдуманном транскрипте.
    """
    settings = settings or CueSettings()
    words = _words_within(transcript, span)

    if words:
        groups = split_into_cues(words, settings)
    else:
        # Пословных таймингов нет — работаем сегментами. Это не выдуманный
        # случай: словами делится не всякая модель, а внешние субтитры,
        # которые когда-нибудь приедут вместо распознавания, — тем более.
        groups = [
            [
                TranscriptWord(
                    text=segment.text.strip(), start=segment.range.start, end=segment.range.end
                )
            ]
            for segment in transcript.within(span)
            if segment.text.strip()
        ]

    cues: list[Cue] = []
    for group in groups:
        cue = _to_cue(group, span, settings)
        if cue is not None:
            cues.append(cue)
    return _resolve_overlaps(cues)


def split_into_cues(
    words: Sequence[TranscriptWord],
    settings: CueSettings,
) -> list[list[TranscriptWord]]:
    """Разбить поток слов на реплики.

    Три причины начать новую реплику, в порядке убывания приоритета:
    конец предложения, пауза в речи, переполнение экрана. Первые две —
    смысловые, третья — вынужденная, и именно поэтому она последняя:
    рвать по переполнению приходится только там, где говорят без пауз
    и без точек.
    """
    groups: list[list[TranscriptWord]] = []
    current: list[TranscriptWord] = []

    for word in words:
        if current:
            gap = word.start - current[-1].end
            length = len(_join(current)) + 1 + len(word.text.strip())
            ends_sentence = bool(_SENTENCE_END.search(current[-1].text.strip()))
            too_long = length > settings.max_length
            too_slow = word.end - current[0].start > settings.max_duration_sec

            if ends_sentence or gap >= settings.gap_sec or too_long or too_slow:
                groups.append(current)
                current = []

        current.append(word)

    if current:
        groups.append(current)
    return groups


def wrap_lines(text: str, max_chars: int, max_lines: int) -> tuple[str, ...]:
    """Разложить текст по строкам, разрывая только между словами.

    Строки выравниваются по длине, а не набиваются жадно. Жадная набивка
    даёт «сорок символов и одно слово во второй строке» — читается хуже
    и выглядит как ошибка вёрстки.

    Если текст не помещается в отведённое число строк, последняя строка
    остаётся длиннее нормы. Резать слово посередине хуже: это ошибка,
    которую видно, а длинная строка — та, которую можно пережить.
    """
    words = text.split()
    if not words:
        return ()
    if max_lines <= 1 or len(text) <= max_chars:
        return (text,)

    # Целевая длина строки — при равномерном распределении по строкам.
    lines_needed = min(max_lines, max(1, -(-len(text) // max_chars)))
    target = -(-len(text) // lines_needed)

    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if current and len(candidate) > max(target, len(word)) and len(lines) + 1 < max_lines:
            lines.append(" ".join(current))
            current = [word]
        else:
            current = [*current, word]
    if current:
        lines.append(" ".join(current))
    return tuple(lines)


def _to_cue(
    words: Sequence[TranscriptWord],
    span: TimeRange,
    settings: CueSettings,
) -> Cue | None:
    text = _join(words)
    if not text:
        return None

    start = max(words[0].start - span.start, 0.0)
    end = min(words[-1].end - span.start, span.duration)

    # Минимальное время показа: чем длиннее текст, тем дольше он висит.
    # Реплика вправе выехать за конец клипа не может — там уже нет кадра.
    needed = max(settings.min_duration_sec, len(text) / max(settings.reading_speed, 1.0))
    if end - start < needed:
        end = min(start + needed, span.duration)
    if end - start > settings.max_duration_sec:
        end = start + settings.max_duration_sec
    if end <= start:
        return None

    shifted = tuple(
        TranscriptWord(
            text=word.text,
            start=max(word.start - span.start, 0.0),
            end=max(word.end - span.start, 0.0),
            probability=word.probability,
        )
        for word in words
    )
    return Cue(
        range=TimeRange(start, end),
        lines=wrap_lines(text, settings.max_chars, settings.max_lines),
        words=shifted,
    )


def _resolve_overlaps(cues: Sequence[Cue]) -> list[Cue]:
    """Убрать наложения, возникшие после растягивания коротких реплик.

    Две реплики на экране одновременно — это не «две строки», а каша:
    плеер покажет их друг поверх друга. Растянутая предыдущая реплика
    подрезается, а если после подрезки от неё ничего не остаётся —
    выбрасывается целиком: лучше потерять вспышку текста, чем показать
    её поверх следующей.
    """
    result: list[Cue] = []
    for cue in cues:
        if result and cue.range.start < result[-1].range.end:
            previous = result[-1]
            trimmed_end = cue.range.start
            if trimmed_end - previous.range.start <= 0.2:
                result.pop()
            else:
                result[-1] = Cue(
                    range=TimeRange(previous.range.start, trimmed_end),
                    lines=previous.lines,
                    words=previous.words,
                )
        result.append(cue)
    return result


def _words_within(transcript: Transcript, span: TimeRange) -> list[TranscriptWord]:
    """Слова, попадающие в клип.

    По началу слова, а не по пересечению: слово, начавшееся до клипа
    и заехавшее в него на пару кадров, зрителю не покажут — оно уже
    прозвучало.
    """
    return [
        word
        for segment in transcript.segments
        for word in segment.words
        if span.contains(word.start) and word.text.strip()
    ]


def _join(words: Sequence[TranscriptWord]) -> str:
    return " ".join(word.text.strip() for word in words if word.text.strip())
