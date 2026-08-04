"""Темп и плотность речи.

Самый дешёвый анализатор в проекте: ни модели, ни файла, ни процесса —
только арифметика над транскриптом, который уже посчитан на Этапе 4.
Тем не менее признаки отсюда — одни из самых полезных.

Что они означают на практике:

* **wpm** — быстрая речь выдаёт возбуждение, спор, экшен; медленная —
  раздумье или драматизм;
* **delta** — отклонение от собственного темпа этого диктора. Сто восемьдесят
  слов в минуту у скорого рассказчика норма, а у лектора — событие;
* **density** — доля сцены, занятая речью. Отличает диалог от паузы
  в разговоре и от чистого видеоряда;
* **dialogue_turns** — оценка числа смен реплик: перепалка звучит иначе,
  чем монолог той же длины и того же темпа.

Все функции чистые, поэтому проверяются на выдуманных транскриптах
без единого обращения к диску.
"""

from __future__ import annotations

import itertools
import statistics
from collections.abc import Mapping, Sequence
from typing import ClassVar, Final

from core.domain.analysis import ARTIFACT_TRANSCRIPT, AnalysisContext
from core.domain.media import Transcript, TranscriptSegment
from core.domain.values import FeatureValue, TimeRange
from core.logging import get_logger
from core.registry import register_analyzer

__all__ = ["SpeechRateAnalyzer", "density_in", "turns_in", "words_per_minute"]

log = get_logger(__name__)

#: Пауза, после которой считаем, что слово взял другой человек. Значение
#: эмпирическое: паузы внутри фразы редко превышают полсекунды, а смена
#: говорящего почти всегда заметнее.
_TURN_GAP: Final = 0.6

#: Минимум речи в сцене, при котором темп вообще имеет смысл. На двух словах
#: «слов в минуту» — это шум, умноженный на шестьдесят.
_MIN_WORDS: Final = 3


def words_per_minute(segments: Sequence[TranscriptSegment], scene: TimeRange) -> float | None:
    """Темп речи в сцене.

    Считается по **времени звучания речи**, а не по длине сцены: иначе
    короткая реплика посреди долгой паузы дала бы смехотворно низкий темп,
    хотя произнесена была скороговоркой.
    """
    spoken = 0.0
    words = 0
    for segment in segments:
        overlap = segment.range.intersection(scene)
        if overlap is None:
            continue
        spoken += overlap.duration
        words += _words_within(segment, overlap)

    if words < _MIN_WORDS or spoken <= 0:
        return None
    return round(words / spoken * 60.0, 2)


def density_in(segments: Sequence[TranscriptSegment], scene: TimeRange) -> float:
    """Доля сцены, занятая речью, 0..1.

    Пересечения сегментов схлопываются: у речи с двумя говорящими они
    накладываются, и наивная сумма дала бы плотность больше единицы.
    """
    spans = sorted(
        (span for segment in segments if (span := segment.range.intersection(scene)) is not None),
        key=lambda span: span.start,
    )
    if not spans:
        return 0.0

    total = 0.0
    current_start, current_end = spans[0].start, spans[0].end
    for span in spans[1:]:
        if span.start > current_end:
            total += current_end - current_start
            current_start, current_end = span.start, span.end
        else:
            current_end = max(current_end, span.end)
    total += current_end - current_start

    return round(min(total / scene.duration, 1.0), 4)


def turns_in(segments: Sequence[TranscriptSegment], scene: TimeRange) -> int:
    """Оценка числа смен реплик.

    Именно оценка: без диаризации (задел под неё — поле ``speaker``
    в ``transcript_segments``) точно сказать, кто говорит, нельзя. Считаем
    заметные паузы между сегментами — по ним диалог отличается от монолога
    достаточно надёжно, чтобы признак был полезен скорингу.
    """
    inside = [segment for segment in segments if segment.range.intersection(scene) is not None]
    if len(inside) < 2:
        return 0

    return sum(
        1
        for previous, current in itertools.pairwise(inside)
        if current.range.start - previous.range.end >= _TURN_GAP
    )


@register_analyzer
class SpeechRateAnalyzer:
    """Реализация порта ``FeatureAnalyzer`` для темпа речи."""

    key: ClassVar[str] = "speech.rate"
    requires: ClassVar[frozenset[str]] = frozenset({ARTIFACT_TRANSCRIPT})
    produces: ClassVar[frozenset[str]] = frozenset(
        {
            "speech.rate.wpm",
            "speech.rate.delta",
            "speech.density",
            "speech.dialogue_turns",
        }
    )
    #: Дешевле некуда: чистая арифметика над готовым транскриптом.
    cost: ClassVar[int] = 0

    async def analyze(
        self,
        context: AnalysisContext,
        scenes: Sequence[TimeRange],
    ) -> Mapping[int, Sequence[FeatureValue]]:
        transcript = context.transcript
        if transcript is None or transcript.is_empty or not scenes:
            return {}

        return _aggregate(transcript, scenes)


def _aggregate(
    transcript: Transcript,
    scenes: Sequence[TimeRange],
) -> dict[int, list[FeatureValue]]:
    segments = list(transcript.segments)
    rates = {
        index: rate
        for index, scene in enumerate(scenes)
        if (rate := words_per_minute(segments, scene)) is not None
    }
    # Базовый темп — медиана по сценам, где речь вообще была. Среднее здесь
    # хуже: одна скороговорка на всю лекцию сдвинула бы базу для всех.
    baseline = statistics.median(rates.values()) if rates else None

    features: dict[int, list[FeatureValue]] = {}
    for index, scene in enumerate(scenes):
        values = [
            FeatureValue("speech.density", density_in(segments, scene)),
            FeatureValue("speech.dialogue_turns", float(turns_in(segments, scene))),
        ]
        rate = rates.get(index)
        if rate is not None:
            values.append(FeatureValue("speech.rate.wpm", rate))
            if baseline:
                values.append(FeatureValue("speech.rate.delta", round(rate - baseline, 2)))
        features[index] = values

    return features


def _words_within(segment: TranscriptSegment, window: TimeRange) -> int:
    """Слов сегмента, попавших в окно.

    При наличии пословных таймингов считаем по ним — это точно. Без них
    (реализация ``Transcriber`` их не гарантирует для чужих плагинов)
    оцениваем по доле пересечения: грубо, но лучше, чем ноль.
    """
    if segment.words:
        return sum(1 for word in segment.words if window.contains(word.start))

    total = len(segment.text.split())
    if total == 0 or segment.range.duration <= 0:
        return 0
    return round(total * window.duration / segment.range.duration)
