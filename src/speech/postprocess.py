"""Обработка сырого результата распознавания.

Модель отдаёт то, что услышала, и это не всегда то, что было сказано.
Три проблемы решаются здесь, и все три — чистыми функциями над числами
и строками, без модели и без файлов:

1. **Склейка кусков.** Длинный файл распознаётся частями с перекрытием,
   иначе на стыке теряются слова. Перекрытие означает, что одна и та же
   фраза приходит дважды — её надо выбрать ровно один раз.
2. **Мусор вместо речи.** На музыке и шуме Whisper уверенно выдумывает текст:
   «Продолжение следует…», «Субтитры сделал…», зацикленные повторы одной
   строки. В скоринг такое попадать не должно.
3. **Тайминги.** Пословные тайминги обязаны идти по возрастанию: на них
   держатся караоке-субтитры (Этап 9) и снап границ клипа к паузе (Этап 7).
   Модель изредка выдаёт слово, которое кончается раньше, чем началось.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Final

from core.domain.media import Transcript, TranscriptSegment, TranscriptWord
from core.domain.values import EPSILON, TimeRange

__all__ = [
    "Chunk",
    "chunk_windows",
    "drop_non_speech",
    "merge_chunks",
    "normalize_words",
    "shift",
]

#: Сколько раз подряд одна и та же строка означает зацикливание декодера.
#: Два повтора — законная речь («да, да»), три и больше в отдельных
#: сегментах — почти всегда петля на музыке или тишине.
_LOOP_REPEATS: Final = 3

#: Ниже этой средней логарифмической вероятности текст считается угаданным,
#: а не расслышанным. Значение из референсной реализации Whisper: оно
#: подобрано на их наборе, и своих замеров, чтобы его двигать, у нас нет.
_LOGPROB_VETO: Final = -1.0

_PUNCTUATION: Final = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES: Final = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class Chunk:
    """Кусок файла и то, что в нём распознано (в абсолютном времени)."""

    range: TimeRange
    transcript: Transcript


def chunk_windows(
    total_sec: float,
    *,
    chunk_sec: float,
    overlap_sec: float,
) -> list[TimeRange]:
    """Разбить файл на куски с перекрытием.

    ``chunk_sec <= 0`` означает «не резать»: модель пройдёт файл целиком.
    Резать нужно не ради качества — faster-whisper и так идёт по файлу
    окнами, — а ради двух вещей: дробного прогресса на этапе, который
    занимает четверть всего времени джоба, и предсказуемого потолка
    по памяти на файлах в несколько часов.
    """
    if total_sec <= 0:
        return []
    if chunk_sec <= 0 or total_sec <= chunk_sec:
        return [TimeRange(0.0, total_sec)]

    step = max(chunk_sec - overlap_sec, chunk_sec / 2)
    windows: list[TimeRange] = []
    start = 0.0
    while start < total_sec - EPSILON:
        end = min(start + chunk_sec, total_sec)
        windows.append(TimeRange(start, end))
        if end >= total_sec - EPSILON:
            break
        start += step
    return windows


def shift(transcript: Transcript, delta: float) -> Transcript:
    """Сдвинуть все тайминги на ``delta`` секунд.

    Модель, получив кусок файла, отсчитывает время от его начала. Конвейеру
    нужно время от начала видео — иначе субтитры уедут на десять минут.
    """
    if not delta:
        return transcript

    return Transcript(
        segments=tuple(
            replace(
                segment,
                range=segment.range.shifted(delta),
                words=tuple(
                    replace(word, start=word.start + delta, end=word.end + delta)
                    for word in segment.words
                ),
            )
            for segment in transcript.segments
        ),
        language=transcript.language,
    )


def merge_chunks(chunks: Sequence[Chunk]) -> Transcript:
    """Склеить куски, разрезав каждое перекрытие ровно посередине.

    Альтернативы хуже. «Брать из первого куска» теряет фразу, которая
    в первом обрывается на полуслове. «Сравнивать тексты и выбрасывать
    дубликаты» зависит от того, одинаково ли модель расслышала фразу дважды —
    а она регулярно слышит по-разному, и в результат попадают оба варианта.

    Разрез по середине перекрытия даёт то, что нужно: каждый момент времени
    принадлежит ровно одному куску, дубликатов нет по построению, а фраза
    на стыке достаётся тому куску, в котором у неё больше контекста.
    """
    if not chunks:
        return Transcript()
    if len(chunks) == 1:
        return chunks[0].transcript

    ordered = sorted(chunks, key=lambda chunk: chunk.range.start)
    bounds = _authoritative_bounds([chunk.range for chunk in ordered])

    segments: list[TranscriptSegment] = []
    for chunk, (low, high) in zip(ordered, bounds, strict=True):
        segments.extend(
            segment
            for segment in chunk.transcript.segments
            if low - EPSILON <= segment.range.midpoint < high - EPSILON
        )

    segments.sort(key=lambda segment: segment.range.start)
    language = next(
        (chunk.transcript.language for chunk in ordered if chunk.transcript.language), None
    )
    return Transcript(segments=tuple(segments), language=language)


def _authoritative_bounds(ranges: Sequence[TimeRange]) -> list[tuple[float, float]]:
    """Зона ответственности каждого куска: середины соседних перекрытий."""
    cuts = [(current.start + previous.end) / 2 for previous, current in itertools.pairwise(ranges)]
    lows = [ranges[0].start, *cuts]
    highs = [*cuts, ranges[-1].end]
    return list(zip(lows, highs, strict=True))


def drop_non_speech(
    transcript: Transcript,
    *,
    no_speech_threshold: float = 0.6,
    logprob_threshold: float = _LOGPROB_VETO,
) -> Transcript:
    """Выбросить сегменты, в которых речи не было.

    Три признака, и первый из них требует осторожности:

    * **высокий ``no_speech_prob`` при низкой уверенности в тексте.** Оба
      условия обязательны — почему, см. ниже;
    * пустой текст после нормализации — сегмент из одних знаков препинания;
    * зацикленный повтор одной строки — известный режим отказа декодера
      на музыке и длинной тишине.

    > ``no_speech_prob`` — оценка **тридцатисекундного окна декодирования**,
    > а не отдельного сегмента. Все сегменты внутри окна наследуют одно
    > и то же значение. Достаточно музыкальной отбивки в начале окна, чтобы
    > оценка ушла к 0.74 — и наивный порог выбросит вместе с ней всю речь,
    > которая идёт следом.
    >
    > Поэтому вторым условием стоит ``avg_logprob``: он относится уже
    > к конкретному сегменту и говорит, насколько модель уверена в **тексте**.
    > Уверенный текст в «шумном» окне — это речь, и трогать его нельзя.
    > Так же устроен и референсный код Whisper.
    >
    > Найдено замером: на 59-секундном клипе с русской речью фильтр без
    > второго условия удалял тридцать секунд подряд — шесть сегментов
    > с ``no_speech=0.74`` и ``logprob=-0.17``, то есть текст, в котором
    > модель была уверена почти полностью.
    """
    kept: list[TranscriptSegment] = []
    repeats = 0
    previous_key = ""

    for segment in transcript.segments:
        if _is_silence(segment, no_speech_threshold, logprob_threshold):
            continue

        key = _normalized(segment.text)
        if not key:
            continue

        repeats = repeats + 1 if key == previous_key else 0
        previous_key = key
        if repeats >= _LOOP_REPEATS - 1:
            # Первые два повтора уже приняты — они могли быть речью.
            # Начиная с третьего это петля: удаляем и её начало тоже.
            del kept[max(len(kept) - (_LOOP_REPEATS - 1), 0) :]
            continue

        kept.append(segment)

    return Transcript(segments=tuple(kept), language=transcript.language)


def _is_silence(
    segment: TranscriptSegment,
    no_speech_threshold: float,
    logprob_threshold: float,
) -> bool:
    """Речи в сегменте не было: и окно «молчит», и текст неуверенный."""
    if segment.no_speech_prob is None or segment.no_speech_prob <= no_speech_threshold:
        return False
    # Уверенность неизвестна — доверяем оценке окна: другого признака нет.
    if segment.avg_logprob is None:
        return True
    return segment.avg_logprob < logprob_threshold


def normalize_words(
    words: Sequence[TranscriptWord],
    *,
    bounds: TimeRange | None = None,
) -> tuple[TranscriptWord, ...]:
    """Привести пословные тайминги в монотонный вид.

    Что чинится: слово, которое кончается раньше начала; слово, начинающееся
    раньше предыдущего; выход за границы сегмента. Всё это модель выдаёт
    редко, но выдаёт — а субтитры с отрицательной длительностью не покажет
    ни один плеер.

    Слова не выбрасываются: потерянное слово хуже слова с подправленным
    таймингом на десять миллисекунд.
    """
    result: list[TranscriptWord] = []
    cursor = bounds.start if bounds else None

    for word in sorted(words, key=lambda item: (item.start, item.end)):
        start = word.start
        end = max(word.end, start)
        if cursor is not None:
            start = max(start, cursor)
            end = max(end, start)
        if bounds is not None:
            start = min(max(start, bounds.start), bounds.end)
            end = min(max(end, start), bounds.end)

        result.append(replace(word, start=start, end=end))
        cursor = end

    return tuple(result)


def _normalized(text: str) -> str:
    """Текст без пунктуации и регистра — для сравнения повторов."""
    return _SPACES.sub(" ", _PUNCTUATION.sub(" ", text.lower())).strip()
