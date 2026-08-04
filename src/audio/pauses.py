"""Паузы в речи.

Тишина — это содержание, а не отсутствие содержания. Драматическая пауза
перед репликой, повисшее молчание после вопроса, затишье перед взрывом —
всё это маркеры момента, ради которого зритель и остаётся. Поэтому паузы
считаются отдельным признаком, а не выводятся из «нет речи».

Детектор — Silero VAD, тот же, что уже отсекает тишину при распознавании
(Этап 4). Он приезжает вместе с faster-whisper, работает на CPU за секунды
и не требует ни новой зависимости, ни весов.

> **Почему не по транскрипту**
>
> Соблазн велик: сегменты речи уже есть, разница между ними — пауза.
> Но транскрипт знает только про **распознанную** речь. Реплика на фоне
> взрыва в него не попадёт, и «пауза» окажется там, где на самом деле крик.
> VAD слышит голос независимо от того, разобрал ли его Whisper.
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from core.domain.analysis import ARTIFACT_AUDIO, AnalysisContext
from core.domain.errors import ProcessingError
from core.domain.values import EPSILON, FeatureValue, TimeRange
from core.logging import get_logger
from core.registry import register_analyzer

__all__ = ["PauseAnalyzer", "pauses_in", "silence_between"]

log = get_logger(__name__)


def silence_between(speech: Sequence[TimeRange], window: TimeRange) -> list[TimeRange]:
    """Промежутки без речи внутри окна.

    Отдельная чистая функция, потому что здесь легко ошибиться на границах:
    речь может начинаться до окна и кончаться после, интервалы могут
    пересекаться, а окно — не содержать речи вовсе.
    """
    inside = sorted(
        (span for interval in speech if (span := interval.intersection(window)) is not None),
        key=lambda span: span.start,
    )
    if not inside:
        return [window]

    gaps: list[TimeRange] = []
    if inside[0].start - window.start > EPSILON:
        gaps.append(TimeRange(window.start, inside[0].start))
    for previous, current in itertools.pairwise(inside):
        if current.start - previous.end > EPSILON:
            gaps.append(TimeRange(previous.end, current.start))
    if window.end - inside[-1].end > EPSILON:
        gaps.append(TimeRange(inside[-1].end, window.end))
    return gaps


def pauses_in(speech: Sequence[TimeRange], scene: TimeRange) -> list[FeatureValue]:
    """Признаки пауз для одной сцены.

    ``max`` важнее, чем ``total``: десять пауз по полсекунды — это обычный
    ритм разговора, а одна пауза на четыре секунды — событие.
    """
    gaps = silence_between(speech, scene)
    total = sum(gap.duration for gap in gaps)
    longest = max((gap.duration for gap in gaps), default=0.0)
    return [
        FeatureValue("audio.pause.total", round(total, 3)),
        FeatureValue("audio.pause.max", round(longest, 3)),
        FeatureValue("audio.pause.ratio", round(min(total / scene.duration, 1.0), 4)),
    ]


@register_analyzer
class PauseAnalyzer:
    """Реализация порта ``FeatureAnalyzer`` для пауз."""

    key: ClassVar[str] = "audio.pauses"
    requires: ClassVar[frozenset[str]] = frozenset({ARTIFACT_AUDIO})
    produces: ClassVar[frozenset[str]] = frozenset(
        {"audio.pause.total", "audio.pause.max", "audio.pause.ratio"}
    )
    cost: ClassVar[int] = 2

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        min_silence_ms: int = 200,
        min_speech_ms: int = 150,
        speech_pad_ms: int = 0,
    ) -> None:
        self._threshold = threshold
        # Пороги мягче, чем при распознавании: там VAD решает, что показывать
        # модели, и полусекундная пауза внутри фразы только мешает. Здесь
        # такая пауза — измеряемая величина, и терять её нельзя.
        self._min_silence_ms = min_silence_ms
        self._min_speech_ms = min_speech_ms
        # Ноль вместо штатных 400 мс — и это главная настройка здесь.
        # Отступ расширяет каждый интервал речи в обе стороны, чтобы модель
        # не срезала начало слова. Для измерения пауз он губителен: съедает
        # по 800 мс из каждого промежутка, и всё короче этого исчезает вовсе.
        # На живом клипе с отступом по умолчанию пауз не находилось совсем.
        self._speech_pad_ms = speech_pad_ms

    async def analyze(
        self,
        context: AnalysisContext,
        scenes: Sequence[TimeRange],
    ) -> Mapping[int, Sequence[FeatureValue]]:
        if context.audio is None or not scenes:
            return {}

        speech = await asyncio.to_thread(
            self._detect_speech, context.audio.path, context.audio.sample_rate
        )
        log.debug("pauses.detected", intervals=len(speech), scenes=len(scenes))
        return {index: pauses_in(speech, scene) for index, scene in enumerate(scenes)}

    def _detect_speech(self, path: Path, sample_rate: int) -> list[TimeRange]:
        """Интервалы речи по всему файлу.

        Синхронная работа в отдельном потоке: модель считает секунды даже
        на часе аудио, но блокировать event loop нельзя и на секунду —
        в нём живут отмена задачи и heartbeat воркера.
        """
        try:
            from faster_whisper.audio import decode_audio
            from faster_whisper.vad import VadOptions, get_speech_timestamps
        except ImportError as error:  # pragma: no cover — зависит от окружения
            raise ProcessingError(
                "faster-whisper не установлен: VAD недоступен. Поставьте extra 'worker'.",
            ) from error

        try:
            samples: Any = decode_audio(str(path), sampling_rate=sample_rate)
            timestamps = get_speech_timestamps(
                samples,
                VadOptions(
                    threshold=self._threshold,
                    min_silence_duration_ms=self._min_silence_ms,
                    min_speech_duration_ms=self._min_speech_ms,
                    speech_pad_ms=self._speech_pad_ms,
                ),
                sampling_rate=sample_rate,
            )
        except Exception as error:
            raise ProcessingError(
                f"VAD не смог обработать аудио: {type(error).__name__}: {error}",
                path=path.name,
            ) from error

        return [
            TimeRange(item["start"] / sample_rate, item["end"] / sample_rate)
            for item in timestamps
            if item["end"] > item["start"]
        ]
