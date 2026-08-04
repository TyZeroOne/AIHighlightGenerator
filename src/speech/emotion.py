"""Эмоция в голосе: возбуждение, знак, класс.

Модель — wav2vec2, дообученный на распознавание эмоции (SUPERB ER).
Она отвечает четырьмя классами: `neu`, `hap`, `ang`, `sad`.

## Два несовпадения с док 05, и оба сознательные

**Первое: `dominant` пишется не меткой, а вероятностями.** Признак
в базе — число (`{"value": float, "confidence": float}`), строке там места
нет, и вводить его ради одного признака значит менять домен, схему БД
и скоринг разом. Вместо этого пишутся вероятности всех четырёх классов;
метка восстанавливается как argmax, а скоринг получает больше информации,
чем из одной строки: «0.51 злости против 0.49 радости» и «0.98 злости» —
это разные сцены.

**Второе: arousal и valence считаются проекцией, а не моделью.** Есть
модели, предсказывающие их напрямую (audeering и подобные), но это `large`
на 1.2 ГБ против 380 МБ здесь. Проекция четырёх классов на круг Рассела —
приближение, зато бесплатное:

| класс | arousal | valence |
|---|---|---|
| neu | 0.30 | 0.0 |
| hap | 0.80 | +0.8 |
| ang | 0.90 | −0.7 |
| sad | 0.25 | −0.6 |

Значения — взвешенная сумма по вероятностям, а не выбор строки таблицы:
смесь «полурадость-полугрусть» даёт средний arousal, а не скачок между
двумя точками.

## Почему не вся речь сцены

wav2vec2 обрабатывает звук примерно в пятнадцать раз быстрее реального
времени на процессоре. Два часа речи — это больше двух часов работы
только на этот признак. Поэтому на сцену берётся до `SER_MAX_SPEECH_SEC`
секунд **речи** (не тишины — интервалы известны из транскрипта), и этого
достаточно: эмоция говорящего не меняется каждую секунду.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Final, Protocol

from core.devices import resolve_torch_device
from core.domain.analysis import ARTIFACT_AUDIO, ARTIFACT_TRANSCRIPT, AnalysisContext
from core.domain.errors import ModelLoadError, ProcessingError
from core.domain.media import Transcript
from core.domain.values import FeatureValue, TimeRange
from core.infra.wav import read_window
from core.logging import get_logger
from core.registry import register_analyzer
from speech.sentiment import exclamation_density, text_sentiment

__all__ = [
    "AROUSAL",
    "VALENCE",
    "EmotionAnalyzer",
    "SpeechEmotionModel",
    "dimensions",
    "speech_spans",
]

log = get_logger(__name__)

#: Проекция классов на круг Рассела (см. заголовок модуля).
AROUSAL: Final[Mapping[str, float]] = {"neu": 0.30, "hap": 0.80, "ang": 0.90, "sad": 0.25}
VALENCE: Final[Mapping[str, float]] = {"neu": 0.0, "hap": 0.8, "ang": -0.7, "sad": -0.6}

#: Сколько секунд речи на сцену подавать модели.
DEFAULT_MAX_SPEECH_SEC: Final = 5.0

#: Короче этого куска классификатор ненадёжен: на полусекунде интонации
#: попросту нет, а модель всё равно выдаст уверенный ответ.
_MIN_SPEECH_SEC: Final = 0.6

#: Минимальный осмысленный остаток бюджета речи. Меньше — не берём кусок
#: вовсе: пятидесятая доля секунды ничего не добавит к оценке эмоции,
#: зато накопленная ошибка float делает такой отрезок вырожденным.
_MIN_SPAN_SEC: Final = 0.05


class EmotionScorer(Protocol):
    """Оценка вероятностей эмоциональных классов на куске речи."""

    def probabilities(self, samples: Any, sample_rate: int) -> Mapping[str, float]: ...


def speech_spans(
    transcript: Transcript,
    scene: TimeRange,
    *,
    limit_sec: float = DEFAULT_MAX_SPEECH_SEC,
) -> list[TimeRange]:
    """Куски речи внутри сцены общей длительностью не больше ``limit_sec``.

    Чистая функция: вся хитрость здесь в границах (сегмент может начинаться
    до сцены и кончаться после), и проверять это на выдуманном транскрипте
    несравнимо дешевле, чем на файле.
    """
    spans: list[TimeRange] = []
    remaining = limit_sec
    for segment in transcript.within(scene):
        # Сравнение с порогом, а не с нулём, и это не перестраховка.
        # ``remaining -= piece.duration`` накапливает ошибку float, и на
        # реальном транскрипте остаток обращался в 3e-15: больше нуля,
        # поэтому цикл продолжался, но обрезанный по нему кусок получал
        # нулевую длительность, и TimeRange падал на проверке границ.
        # Найдено живым прогоном на трейлере.
        if remaining < _MIN_SPAN_SEC:
            break
        piece = segment.range.intersection(scene)
        if piece is None or piece.duration <= 0:
            continue
        if piece.duration > remaining:
            piece = TimeRange(piece.start, piece.start + remaining)
        spans.append(piece)
        remaining -= piece.duration
    return spans


def dimensions(probabilities: Mapping[str, float]) -> tuple[float, float]:
    """Возбуждение (0..1) и знак эмоции (−1..1) из вероятностей классов."""
    total = sum(probabilities.values()) or 1.0
    arousal = sum(AROUSAL.get(label, 0.5) * value for label, value in probabilities.items()) / total
    valence = sum(VALENCE.get(label, 0.0) * value for label, value in probabilities.items()) / total
    return round(arousal, 4), round(valence, 4)


class SpeechEmotionModel:
    """SER-модель из `transformers` за ленивой загрузкой."""

    def __init__(
        self,
        *,
        model_name: str,
        device: str = "auto",
        cache_dir: Path | None = None,
    ) -> None:
        self._model_name = model_name
        self._requested_device = device
        self._cache_dir = cache_dir
        self._model: Any | None = None
        self._extractor: Any | None = None
        self._device: str | None = None
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        return self._device or "?"

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._extractor is not None:
            return self._model, self._extractor

        with self._lock:
            if self._model is not None and self._extractor is not None:
                return self._model, self._extractor
            try:
                import torch
                from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
            except ImportError as error:  # pragma: no cover — зависит от окружения
                raise ModelLoadError(
                    "transformers/torch не установлены. Поставьте extra 'worker'.",
                ) from error

            device = resolve_torch_device(self._requested_device)
            log.info("speech.emotion.loading", model=self._model_name, device=device)
            try:
                extractor = AutoFeatureExtractor.from_pretrained(
                    self._model_name, cache_dir=self._cache_dir
                )
                model = AutoModelForAudioClassification.from_pretrained(
                    self._model_name, cache_dir=self._cache_dir
                )
                model.eval()
                model.to(torch.device(device))
            except Exception as error:
                raise ModelLoadError(
                    f"Не удалось загрузить {self._model_name}: {type(error).__name__}: {error}",
                    model=self._model_name,
                    device=device,
                ) from error

            self._extractor = extractor
            self._model = model
            self._device = device
            return model, extractor

    def probabilities(self, samples: Any, sample_rate: int) -> Mapping[str, float]:
        """Вероятности эмоциональных классов.

        Softmax, а не сигмоида: в отличие от AudioSet, здесь классы
        взаимоисключающие — говорящий не бывает одновременно злым и грустным
        в одном и том же куске.
        """
        import torch

        model, extractor = self._load()
        inputs = extractor(samples, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits[0]
        values = torch.softmax(logits, dim=-1).cpu().numpy()
        labels = model.config.id2label
        return {labels[index]: float(value) for index, value in enumerate(values)}


@register_analyzer
class EmotionAnalyzer:
    """Реализация порта ``FeatureAnalyzer`` для эмоции в речи."""

    key: ClassVar[str] = "speech.emotion"
    #: Транскрипт нужен не ради текста (хотя тональность считается по нему),
    #: а ради интервалов речи: гонять SER по тишине — это платить за шум.
    requires: ClassVar[frozenset[str]] = frozenset({ARTIFACT_AUDIO, ARTIFACT_TRANSCRIPT})
    produces: ClassVar[frozenset[str]] = frozenset(
        {
            "speech.emotion.arousal",
            "speech.emotion.valence",
            "speech.emotion.neu",
            "speech.emotion.hap",
            "speech.emotion.ang",
            "speech.emotion.sad",
            "speech.text.sentiment",
            "speech.text.exclamations",
        }
    )
    cost: ClassVar[int] = 8

    def __init__(
        self,
        *,
        scorer: EmotionScorer | None = None,
        max_speech_sec: float | None = None,
    ) -> None:
        self._scorer = scorer
        if max_speech_sec is None:
            from core.config import get_settings

            max_speech_sec = getattr(get_settings(), "ser_max_speech_sec", DEFAULT_MAX_SPEECH_SEC)
        self._max_speech_sec = max_speech_sec

    async def analyze(
        self,
        context: AnalysisContext,
        scenes: Sequence[TimeRange],
    ) -> Mapping[int, Sequence[FeatureValue]]:
        transcript = context.transcript
        if context.audio is None or transcript is None or transcript.is_empty or not scenes:
            return {}

        scorer = self._scorer or _shared_model()
        return await asyncio.to_thread(
            self._run, scorer, context.audio.path, context.audio.sample_rate, transcript, scenes
        )

    # ------------------------------------------------------------ внутреннее
    def _run(
        self,
        scorer: EmotionScorer,
        path: Path,
        rate: int,
        transcript: Transcript,
        scenes: Sequence[TimeRange],
    ) -> Mapping[int, Sequence[FeatureValue]]:
        import numpy as np

        features: dict[int, list[FeatureValue]] = {}
        for index, scene in enumerate(scenes):
            values: list[FeatureValue] = []

            # Текстовая часть считается всегда, даже если речи слишком мало
            # для модели: буквы есть буквы.
            text = transcript.text_within(scene)
            if text:
                sentiment = text_sentiment(text)
                if sentiment is not None:
                    values.append(FeatureValue("speech.text.sentiment", sentiment))
                values.append(FeatureValue("speech.text.exclamations", exclamation_density(text)))

            spans = speech_spans(transcript, scene, limit_sec=self._max_speech_sec)
            speech_sec = sum(span.duration for span in spans)
            if spans and speech_sec >= _MIN_SPEECH_SEC:
                pieces = [read_window(path, span) for span in spans]
                pieces = [piece for piece in pieces if len(piece)]
                if pieces:
                    samples = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
                    try:
                        probabilities = scorer.probabilities(samples, rate)
                    except Exception as error:
                        raise ProcessingError(
                            f"SER не справился: {type(error).__name__}: {error}",
                            path=path.name,
                        ) from error

                    arousal, valence = dimensions(probabilities)
                    confidence = round(float(max(probabilities.values(), default=0.0)), 4)
                    values.append(FeatureValue("speech.emotion.arousal", arousal, confidence))
                    values.append(FeatureValue("speech.emotion.valence", valence, confidence))
                    values.extend(
                        FeatureValue(f"speech.emotion.{label}", round(float(value), 4))
                        for label, value in probabilities.items()
                    )

            if values:
                features[index] = values

        log.debug("speech.emotion.done", scenes=len(features))
        return features


#: Модель на процесс — см. те же соображения, что в ``audio.events``.
_MODEL: SpeechEmotionModel | None = None
_MODEL_LOCK = threading.Lock()


def _shared_model() -> SpeechEmotionModel:
    from core.config import get_settings

    global _MODEL
    model = _MODEL
    if model is not None:
        return model

    with _MODEL_LOCK:
        if _MODEL is None:
            settings = get_settings()
            _MODEL = SpeechEmotionModel(model_name=settings.ser_model, device=settings.device)
        return _MODEL


def reset_model() -> None:
    """Забыть загруженную модель. Нужно тестам."""
    global _MODEL
    _MODEL = None
