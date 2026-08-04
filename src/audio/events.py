"""Звуковые события: смех, крик, аплодисменты, музыка, удары.

Это первый анализатор проекта, которому нужна настоящая нейросеть общего
назначения, и главное решение здесь — **какая именно**.

> **Один AudioSet-классификатор вместо пяти детекторов**
>
> Специализированный детектор смеха точнее детектора-универсала на смехе —
> и бесполезен на всём остальном. Пять специализированных моделей это пять
> загрузок, пять наборов весов и пять проходов по одному и тому же звуку.
>
> AudioSet — размеченный корпус на 527 классов бытовых звуков, и модель,
> обученная на нём, за один проход отвечает про все пять интересующих нас
> событий сразу. Добавить шестое (звон стекла, сирена, лай) — это строка
> в таблице соответствий ниже, без новых весов и без нового кода.

Модель по умолчанию — AST (Audio Spectrogram Transformer), дообученный
на AudioSet. Роадмап называл PANNs; на практике AST в той же весовой
категории (~350 МБ) даёт заметно лучшее качество на AudioSet и, что важнее
для нас, приезжает через `transformers` — то есть тем же способом
и в тот же кэш `HF_HOME`, что и остальные модели проекта, вместо
скачивания чекпойнта с постороннего файлового хостинга.

**Классификация мультилейбловая.** В AudioSet у одного фрагмента может
быть несколько меток одновременно (музыка + аплодисменты + речь), поэтому
логиты превращаются в вероятности через сигмоиду, а не через softmax.
С softmax сумма по 527 классам была бы равна единице, и «уверенно слышу
музыку» превратилось бы в 0.02.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Final, Protocol

from core.devices import resolve_torch_device
from core.domain.analysis import ARTIFACT_AUDIO, AnalysisContext
from core.domain.errors import ModelLoadError, ProcessingError
from core.domain.values import FeatureValue, TimeRange
from core.infra.wav import read_window
from core.logging import get_logger
from core.registry import register_analyzer

__all__ = [
    "EVENT_LABELS",
    "AudioSetClassifier",
    "EventAnalyzer",
    "events_from_scores",
    "windows_for",
]

log = get_logger(__name__)

#: Наши события и метки AudioSet, из которых они собираются.
#:
#: Смысл группировки — устойчивость. «Laughter» и «Giggle» это одно и то же
#: событие для отбора моментов, но модель распределяет уверенность между
#: ними, и по одной метке смешная сцена набирает 0.3 вместо 0.6. Берём
#: максимум по группе: событие произошло, если сработала хоть одна её метка.
EVENT_LABELS: Final[Mapping[str, tuple[str, ...]]] = {
    "laughter": (
        "Laughter",
        "Baby laughter",
        "Giggle",
        "Snicker",
        "Belly laugh",
        "Chuckle, chortle",
    ),
    "scream": (
        "Screaming",
        "Shout",
        "Yell",
        "Bellow",
        "Whoop",
        "Children shouting",
        "Battle cry",
    ),
    "applause": (
        "Applause",
        "Clapping",
        "Cheering",
        "Crowd",
        "Hubbub, speech noise, speech babble",
    ),
    "music": (
        "Music",
        "Musical instrument",
        "Singing",
        "Song",
        "Background music",
        "Theme music",
    ),
    "impact": (
        "Explosion",
        "Gunshot, gunfire",
        "Smash, crash",
        "Bang",
        "Boom",
        "Thump, thud",
        "Slap, smack",
        "Breaking",
    ),
}

#: Длина окна, на которой обучался AST. Отклоняться можно — препроцессор
#: дополнит или обрежет, — но качество на сильно других длинах падает.
DEFAULT_WINDOW_SEC: Final = 10.0

#: Ниже этого значения событие не пишется в признаки вовсе. AudioSet выдаёт
#: 527 вероятностей, и почти все они — шум около нуля; складывать его
#: в базу значит хранить мегабайты нулей и мешать скорингу.
DEFAULT_THRESHOLD: Final = 0.1

#: Вето по тишине.
#:
#: Замер на реальных весах: десять секунд ровной тишины модель размечает
#: как ``Silence`` 0.61 — и одновременно ``Music`` 0.27. Причина понятна:
#: в AudioSet тихие фрагменты чаще всего вырезаны из музыкальных роликов,
#: и модель выучила эту связь. Для нас это чистое ложное срабатывание:
#: сцена, где не звучит вообще ничего, получала бы признак «музыка» выше
#: порога.
#:
#: Поэтому окно, про которое модель уверенно говорит «тишина», выбрасывается
#: целиком. Ровно тот же приём, что и вето по ``avg_logprob`` на Этапе 4:
#: одна уверенная метка отменяет остальные, если по смыслу они несовместимы.
_SILENCE_LABEL: Final = "Silence"
_SILENCE_VETO: Final = 0.5


class EventScorer(Protocol):
    """То, что умеет оценить вероятности звуковых меток на куске аудио.

    Порт объявлен здесь, а не в ``core/ports``: он нужен ровно одному
    анализатору и существует ради подмены модели в тестах.
    """

    def probabilities(self, samples: Any, sample_rate: int) -> Mapping[str, float]: ...


def windows_for(
    scene: TimeRange,
    *,
    window_sec: float = DEFAULT_WINDOW_SEC,
    max_windows: int = 3,
) -> list[TimeRange]:
    """Окна классификации внутри сцены.

    Короткая сцена — одно окно по центру: смещать его некуда, а дополнить
    тишиной препроцессор умеет сам. Длинная — до ``max_windows`` окон,
    равномерно по сцене.

    Ограничение сверху принципиально. Событие может случиться в любой момент,
    и честный ответ — сплошное окно с перекрытием; но на двухчасовом фильме
    это тысячи прогонов модели и часы работы. Три окна на сцену — компромисс,
    при котором смех и аплодисменты (они длятся секундами) находятся,
    а бюджет этапа остаётся конечным.
    """
    if scene.duration <= window_sec or max_windows <= 1:
        center = (scene.start + scene.end) / 2
        half = window_sec / 2
        return [TimeRange(max(center - half, 0.0), max(center - half, 0.0) + window_sec)]

    count = min(max_windows, max(int(scene.duration // window_sec), 1))
    if count <= 1:
        center = (scene.start + scene.end) / 2
        half = window_sec / 2
        return [TimeRange(center - half, center + half)]

    # Равномерно: начала окон раскладываются по сцене так, чтобы последнее
    # заканчивалось ровно на её конце.
    step = (scene.duration - window_sec) / (count - 1)
    return [
        TimeRange(scene.start + step * i, scene.start + step * i + window_sec) for i in range(count)
    ]


def events_from_scores(
    scores: Mapping[str, float],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[FeatureValue]:
    """Вероятности меток AudioSet → наши пять признаков.

    Чистая функция: таблица соответствий — самая изменчивая часть этого
    модуля, и проверять её на выдуманных вероятностях несравнимо дешевле,
    чем прогонять модель.
    """
    features: list[FeatureValue] = []
    for event, labels in EVENT_LABELS.items():
        value = max((scores.get(label, 0.0) for label in labels), default=0.0)
        if value >= threshold:
            features.append(FeatureValue(f"audio.event.{event}", round(float(value), 4)))
    return features


class AudioSetClassifier:
    """AudioSet-классификатор из `transformers` за ленивой загрузкой.

    Загрузка отложена до первого использования и защищена замком —
    ровно по тем же причинам, что у Whisper на Этапе 4: модель весит
    сотни мегабайт, а задачи в воркере могут пойти параллельно.
    """

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
        model = self._model
        if model is not None and self._extractor is not None:
            return model, self._extractor

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
            log.info("audio.events.loading", model=self._model_name, device=device)
            try:
                extractor = AutoFeatureExtractor.from_pretrained(
                    self._model_name, cache_dir=self._cache_dir
                )
                loaded = AutoModelForAudioClassification.from_pretrained(
                    self._model_name, cache_dir=self._cache_dir
                )
                loaded.eval()
                loaded.to(torch.device(device))
            except Exception as error:
                raise ModelLoadError(
                    f"Не удалось загрузить {self._model_name}: {type(error).__name__}: {error}",
                    model=self._model_name,
                    device=device,
                ) from error

            self._extractor = extractor
            self._model = loaded
            self._device = device
            return loaded, extractor

    def probabilities(self, samples: Any, sample_rate: int) -> Mapping[str, float]:
        """Вероятность каждой метки AudioSet на этом куске звука."""
        import torch

        model, extractor = self._load()
        inputs = extractor(samples, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits[0]
        # Сигмоида, а не softmax: разметка мультилейбловая (см. заголовок модуля).
        probabilities = torch.sigmoid(logits).cpu().numpy()
        labels = model.config.id2label
        return {labels[index]: float(value) for index, value in enumerate(probabilities)}


@register_analyzer
class EventAnalyzer:
    """Реализация порта ``FeatureAnalyzer`` для звуковых событий."""

    key: ClassVar[str] = "audio.events"
    requires: ClassVar[frozenset[str]] = frozenset({ARTIFACT_AUDIO})
    produces: ClassVar[frozenset[str]] = frozenset(
        {f"audio.event.{event}" for event in EVENT_LABELS}
    )
    #: Самый дорогой анализатор аудиогруппы: нейросеть на каждое окно.
    cost: ClassVar[int] = 9

    def __init__(
        self,
        *,
        scorer: EventScorer | None = None,
        window_sec: float | None = None,
        max_windows: int | None = None,
        threshold: float | None = None,
    ) -> None:
        settings = _settings()
        self._scorer = scorer
        self._window_sec = window_sec if window_sec is not None else settings["window_sec"]
        self._max_windows = max_windows if max_windows is not None else settings["max_windows"]
        self._threshold = threshold if threshold is not None else settings["threshold"]

    async def analyze(
        self,
        context: AnalysisContext,
        scenes: Sequence[TimeRange],
    ) -> Mapping[int, Sequence[FeatureValue]]:
        if context.audio is None or not scenes:
            return {}

        scorer = self._scorer or _shared_classifier()
        path = context.audio.path
        rate = context.audio.sample_rate

        return await asyncio.to_thread(self._run, scorer, path, rate, scenes)

    # ------------------------------------------------------------ внутреннее
    def _run(
        self,
        scorer: EventScorer,
        path: Path,
        rate: int,
        scenes: Sequence[TimeRange],
    ) -> Mapping[int, Sequence[FeatureValue]]:
        """Синхронный проход по сценам в отдельном потоке.

        Модель работает с GIL-освобождающим кодом, но сам цикл блокирующий,
        а держать event loop занятым на минуты нельзя: в нём живут отмена
        задачи и heartbeat воркера.
        """
        features: dict[int, list[FeatureValue]] = {}
        for index, scene in enumerate(scenes):
            best: dict[str, float] = {}
            for window in windows_for(
                scene, window_sec=self._window_sec, max_windows=self._max_windows
            ):
                samples = read_window(path, window)
                if len(samples) == 0:
                    continue
                try:
                    scores = scorer.probabilities(samples, rate)
                except Exception as error:
                    raise ProcessingError(
                        f"Классификатор звуковых событий не справился: "
                        f"{type(error).__name__}: {error}",
                        path=path.name,
                    ) from error

                if scores.get(_SILENCE_LABEL, 0.0) >= _SILENCE_VETO:
                    # Тишина отменяет всё остальное — см. комментарий
                    # к _SILENCE_VETO.
                    continue

                # Максимум по окнам: событие в сцене есть, если оно случилось
                # хоть в одном её окне. Среднее размывало бы секундный смех
                # в тридцатисекундной сцене до неразличимости.
                for label, value in scores.items():
                    if value > best.get(label, 0.0):
                        best[label] = value

            values = events_from_scores(best, threshold=self._threshold)
            if values:
                features[index] = values

        log.debug("audio.events.done", scenes=len(features))
        return features


def _settings() -> dict[str, Any]:
    """Настройки анализатора. Реестр создаёт плагины без аргументов,
    поэтому конфиг читается здесь — как и в ``vision.context``."""
    from core.config import get_settings

    settings = get_settings()
    return {
        "window_sec": getattr(settings, "audio_event_window_sec", DEFAULT_WINDOW_SEC),
        "max_windows": getattr(settings, "audio_event_max_windows", 3),
        "threshold": getattr(settings, "audio_event_threshold", DEFAULT_THRESHOLD),
    }


#: Модель на процесс, а не на анализатор: 350 МБ весов ради каждого джоба
#: заново — это минуты на загрузку и лишний гигабайт памяти при двух
#: одновременных задачах.
_CLASSIFIER: AudioSetClassifier | None = None
_CLASSIFIER_LOCK = threading.Lock()


def _shared_classifier() -> AudioSetClassifier:
    from core.config import get_settings

    global _CLASSIFIER
    classifier = _CLASSIFIER
    if classifier is not None:
        return classifier

    with _CLASSIFIER_LOCK:
        if _CLASSIFIER is None:
            settings = get_settings()
            _CLASSIFIER = AudioSetClassifier(
                model_name=settings.audio_event_model,
                device=settings.device,
            )
        return _CLASSIFIER


def reset_classifier() -> None:
    """Забыть загруженную модель. Нужно тестам."""
    global _CLASSIFIER
    _CLASSIFIER = None
