"""Лица и люди в кадре.

Крупный план — это почти всегда эмоция, а эмоция — это момент, ради
которого зритель остаётся. Поэтому считается не только «сколько лиц»,
но и «насколько крупно»: доля кадра, занятая самым большим лицом.

## Чем детектируем

**Лица — YuNet.** Это 232 килобайта весов и единицы миллисекунд на кадр,
при этом качество несопоставимо с каскадами Хаара, которые OpenCV носит
с собой с нулевых. Роадмап называл SCRFD/InsightFace; от него отказались
осознанно: `insightface` тянет собственный стек зависимостей и качает
модели с постороннего хостинга, а выигрыш в качестве на нашей задаче
(есть ли в кадре лицо и насколько оно крупное) не окупает этого.

Каскад Хаара остался как запасной вариант (`FACE_DETECTOR=haar`): он
не требует вообще ничего, кроме OpenCV, и это единственный режим,
работающий на машине без доступа в интернет.

**Люди — SSDLite/MobileNetV3 из torchvision.** Человек со спины, в толпе,
в маске лицом не считается, а для «сколько людей в кадре» это важно.
Модель выбрана самая дешёвая из детекторов COCO: 14 МБ весов и десятки
миллисекунд на кадр против секунд у DETR.

## Почему не по всем кадрам

Лица меняются медленнее, чем движение: между соседними кадрами при 3 fps
человек не появляется и не исчезает. Поэтому детекторы работают по
подвыборке (`VISION_DETECT_FRAMES` кадров на сцену), а не по всем
сэмплам сцены. На двухчасовом фильме это разница между четырьмя минутами
и получасом.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Final, Protocol

from core.config import get_settings
from core.devices import resolve_torch_device
from core.domain.analysis import ARTIFACT_VIDEO, AnalysisContext
from core.domain.errors import ModelLoadError
from core.domain.values import FeatureValue, TimeRange
from core.logging import get_logger
from core.registry import register_analyzer
from vision.context import source_for
from vision.frames import FrameBatch, FrameSource

__all__ = [
    "FaceAnalyzer",
    "HaarFaceDetector",
    "PersonDetector",
    "YuNetFaceDetector",
    "face_features",
    "pick_frames",
    "shared_face_detector",
]

log = get_logger(__name__)

#: Доля кадра, начиная с которой план считается крупным. 12 % — это лицо
#: примерно в треть высоты кадра: разговор в упор, а не человек в комнате.
CLOSEUP_RATIO: Final = 0.12

#: Сколько кадров сцены отдавать детекторам, если конфиг молчит.
DEFAULT_DETECT_FRAMES: Final = 5


class FaceDetector(Protocol):
    """Детектор лиц: цветной кадр → прямоугольники с уверенностью."""

    def detect(self, frame: Any) -> Sequence[tuple[float, float, float, float, float]]: ...


class PeopleDetector(Protocol):
    """Детектор людей: цветной кадр → число найденных людей."""

    def count(self, frame: Any) -> int: ...


def pick_frames(batch: FrameBatch, limit: int) -> list[Any]:
    """Равномерная подвыборка цветных кадров сцены.

    Равномерная, а не «первые N»: сцена может начинаться с перебивки,
    и первые кадры расскажут о ней меньше, чем кадры, размазанные
    по всей длительности.
    """
    frames = batch.color or batch.frames
    if not frames:
        return []
    if limit <= 0 or len(frames) <= limit:
        return list(frames)

    step = len(frames) / limit
    return [frames[min(int(step * index), len(frames) - 1)] for index in range(limit)]


def face_features(
    counts: Sequence[int],
    areas: Sequence[float],
    people: Sequence[int],
) -> list[FeatureValue]:
    """Признаки сцены из покадровых наблюдений.

    ``count_max`` и ``count_mean`` отвечают на разные вопросы: первый —
    «сколько человек участвует в сцене вообще», второй — «сцена про людей
    или люди мелькнули в одном кадре».
    """
    values: list[FeatureValue] = []
    if counts:
        values.append(FeatureValue("vision.faces.count_max", float(max(counts))))
        values.append(FeatureValue("vision.faces.count_mean", round(sum(counts) / len(counts), 3)))
    if areas:
        # Максимум, а не среднее: крупный план в середине сцены — это
        # событие, и усреднение с общими планами его бы стёрло.
        largest = max(areas)
        values.append(FeatureValue("vision.faces.area_ratio", round(largest, 4)))
        values.append(
            FeatureValue("vision.faces.closeup", 1.0 if largest >= CLOSEUP_RATIO else 0.0)
        )
    if people:
        values.append(FeatureValue("vision.people.count", float(max(people))))
    return values


class YuNetFaceDetector:
    """Детектор лиц YuNet поверх ``cv2.FaceDetectorYN``.

    **Детекция сериализована замком, и это не перестраховка.** Объект
    ``FaceDetectorYN`` держит внутри себя граф сети и буферы под текущий
    размер входа — то есть изменяемое состояние. Пока детектор звали
    из одного потока (анализаторы Этапа 5 идут по сценам по очереди),
    это не проявлялось. На Этапе 8 два клипа поехали кадрироваться
    параллельно, и OpenCV упал прямо в графе::

        (-215:Assertion failed) buf.u == m.u in function 'forwardGraph'

    Замок вместо детектора на поток: веса весят 232 КБ, но копия графа
    на каждый поток — это ещё и своя память под буферы, а сама детекция
    занимает малую часть времени рендера. Сериализовать дешевле.
    """

    def __init__(self, *, repo_id: str, filename: str, score_threshold: float = 0.7) -> None:
        self._repo_id = repo_id
        self._filename = filename
        self._score_threshold = score_threshold
        self._detector: Any | None = None
        self._size: tuple[int, int] | None = None
        self._lock = threading.Lock()
        self._detect_lock = threading.Lock()

    def _load(self) -> Any:
        detector = self._detector
        if detector is not None:
            return detector

        with self._lock:
            if self._detector is not None:
                return self._detector
            try:
                import cv2
                from huggingface_hub import hf_hub_download
            except ImportError as error:  # pragma: no cover — зависит от окружения
                raise ModelLoadError(
                    "OpenCV/huggingface_hub не установлены. Поставьте extra 'worker'.",
                ) from error

            try:
                # Веса едут через тот же кэш HF_HOME, что и Whisper с SER:
                # один том, одна политика, один способ прогреть образ.
                path = hf_hub_download(repo_id=self._repo_id, filename=self._filename)
                self._detector = cv2.FaceDetectorYN.create(
                    path, "", (320, 320), self._score_threshold
                )
            except Exception as error:
                raise ModelLoadError(
                    f"Не удалось загрузить YuNet ({self._repo_id}): "
                    f"{type(error).__name__}: {error}",
                    model=self._repo_id,
                ) from error
            log.info("vision.faces.loaded", detector="yunet", model=self._repo_id)
            return self._detector

    def detect(self, frame: Any) -> Sequence[tuple[float, float, float, float, float]]:
        detector = self._load()
        height, width = frame.shape[:2]
        with self._detect_lock:
            # Размер входа задаётся отдельно и обязан совпадать с кадром:
            # иначе координаты уезжают молча, без единой ошибки. Заодно
            # это вторая причина замка: между установкой размера и самой
            # детекцией чужой поток не должен переставить его обратно.
            if self._size != (width, height):
                detector.setInputSize((width, height))
                self._size = (width, height)

            _, faces = detector.detect(frame)
        if faces is None:
            return []
        return [
            (float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[-1]))
            for row in faces
        ]


class HaarFaceDetector:
    """Каскад Хаара из комплекта OpenCV — запасной вариант без загрузок."""

    def __init__(self, *, min_neighbors: int = 6, min_size: int = 24) -> None:
        self._min_neighbors = min_neighbors
        self._min_size = min_size
        self._cascade: Any | None = None
        # Тот же замок и по той же причине, что у YuNet: общий каскад
        # держит внутреннее состояние, а детектор один на процесс.
        self._detect_lock = threading.Lock()

    def _load(self) -> Any:
        if self._cascade is not None:
            return self._cascade
        import cv2

        path = f"{cv2.data.haarcascades}haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
        cascade = cv2.CascadeClassifier(path)  # type: ignore[attr-defined]
        if cascade.empty():  # pragma: no cover — зависит от сборки OpenCV
            raise ModelLoadError(f"Каскад не загрузился: {path}")
        self._cascade = cascade
        log.info("vision.faces.loaded", detector="haar")
        return cascade

    def detect(self, frame: Any) -> Sequence[tuple[float, float, float, float, float]]:
        import cv2

        cascade = self._load()
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        with self._detect_lock:
            found = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=self._min_neighbors,
                minSize=(self._min_size, self._min_size),
            )
        # Каскад не возвращает уверенность вовсе — ставим 1.0 и честно
        # помним, что это не то же самое, что уверенность YuNet.
        return [(float(x), float(y), float(w), float(h), 1.0) for x, y, w, h in found]


class PersonDetector:
    """Детектор людей на SSDLite/MobileNetV3 из torchvision."""

    #: Индекс класса «человек» в разметке COCO.
    PERSON_CLASS: Final = 1

    def __init__(self, *, device: str = "auto", score_threshold: float = 0.5) -> None:
        self._requested_device = device
        self._score_threshold = score_threshold
        self._model: Any | None = None
        self._device: str | None = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        model = self._model
        if model is not None:
            return model

        with self._lock:
            if self._model is not None:
                return self._model
            try:
                import torch
                from torchvision.models.detection import (
                    SSDLite320_MobileNet_V3_Large_Weights,
                    ssdlite320_mobilenet_v3_large,
                )
            except ImportError as error:  # pragma: no cover — зависит от окружения
                raise ModelLoadError(
                    "torchvision не установлен. Поставьте extra 'worker'.",
                ) from error

            device = resolve_torch_device(self._requested_device)
            try:
                weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
                loaded = ssdlite320_mobilenet_v3_large(weights=weights)
                loaded.eval()
                loaded.to(torch.device(device))
            except Exception as error:
                raise ModelLoadError(
                    f"Не удалось загрузить детектор людей: {type(error).__name__}: {error}",
                    device=device,
                ) from error

            self._model = loaded
            self._device = device
            log.info("vision.people.loaded", device=device)
            return loaded

    def count(self, frame: Any) -> int:
        import numpy as np
        import torch

        model = self._load()
        # BGR → RGB и в диапазон [0, 1]: ровно то, чего ждёт torchvision.
        rgb = np.ascontiguousarray(frame[:, :, ::-1]) if frame.ndim == 3 else frame
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
        tensor = tensor.to(next(model.parameters()).device)

        with torch.inference_mode():
            prediction = model([tensor])[0]

        labels = prediction["labels"].cpu().numpy()
        scores = prediction["scores"].cpu().numpy()
        return int(
            sum(
                1
                for label, score in zip(labels, scores, strict=True)
                if label == self.PERSON_CLASS and score >= self._score_threshold
            )
        )


@register_analyzer
class FaceAnalyzer:
    """Реализация порта ``FeatureAnalyzer`` для лиц и людей."""

    key: ClassVar[str] = "vision.faces"
    requires: ClassVar[frozenset[str]] = frozenset({ARTIFACT_VIDEO})
    produces: ClassVar[frozenset[str]] = frozenset(
        {
            "vision.faces.count_max",
            "vision.faces.count_mean",
            "vision.faces.area_ratio",
            "vision.faces.closeup",
            "vision.people.count",
        }
    )
    cost: ClassVar[int] = 7

    def __init__(
        self,
        *,
        source: FrameSource | None = None,
        faces: FaceDetector | None = None,
        people: PeopleDetector | None = None,
        frames_per_scene: int | None = None,
    ) -> None:
        self._source = source
        self._faces = faces
        self._people = people
        if frames_per_scene is None:
            frames_per_scene = getattr(
                get_settings(), "vision_detect_frames", DEFAULT_DETECT_FRAMES
            )
        self._frames_per_scene = frames_per_scene

    async def analyze(
        self,
        context: AnalysisContext,
        scenes: Sequence[TimeRange],
    ) -> Mapping[int, Sequence[FeatureValue]]:
        source = self._source or source_for(context)
        if source is None or not scenes:
            return {}

        faces = self._faces or shared_face_detector()
        people = self._people if self._people is not None else _shared_person_detector()

        features: dict[int, list[FeatureValue]] = {}
        for index, scene in enumerate(scenes):
            batch = await source.scene_frames(scene)
            frames = pick_frames(batch, self._frames_per_scene)
            if not frames:
                continue

            counts: list[int] = []
            areas: list[float] = []
            crowd: list[int] = []
            for frame in frames:
                height, width = frame.shape[:2]
                area = float(height * width) or 1.0
                boxes = faces.detect(frame)
                counts.append(len(boxes))
                if boxes:
                    areas.append(max(box[2] * box[3] for box in boxes) / area)
                if people is not None:
                    crowd.append(people.count(frame))

            values = face_features(counts, areas, crowd)
            if values:
                features[index] = values

        log.debug("vision.faces.done", scenes=len(features))
        return features


#: Детекторы на процесс: веса грузятся один раз, а не на каждый джоб.
_FACES: FaceDetector | None = None
_PEOPLE: PeopleDetector | None = None
_DETECTOR_LOCK = threading.Lock()


def shared_face_detector() -> FaceDetector:
    global _FACES
    detector = _FACES
    if detector is not None:
        return detector

    with _DETECTOR_LOCK:
        if _FACES is None:
            settings = get_settings()
            if getattr(settings, "face_detector", "yunet") == "haar":
                _FACES = HaarFaceDetector()
            else:
                _FACES = YuNetFaceDetector(
                    repo_id=settings.face_model_repo,
                    filename=settings.face_model_file,
                    score_threshold=settings.face_score_threshold,
                )
        return _FACES


def _shared_person_detector() -> PeopleDetector | None:
    """Детектор людей — или ``None``, если он выключен конфигом.

    Выключать его осмысленно: это самая дорогая часть видеогруппы,
    а признак «сколько людей» нужен не всякому профилю скоринга.
    """
    global _PEOPLE
    detector = _PEOPLE
    if detector is not None:
        return detector

    settings = get_settings()
    if not getattr(settings, "person_detector", ""):
        return None

    with _DETECTOR_LOCK:
        if _PEOPLE is None:
            _PEOPLE = PersonDetector(
                device=settings.device,
                score_threshold=settings.person_score_threshold,
            )
        return _PEOPLE


def reset_detectors() -> None:
    """Забыть загруженные детекторы. Нужно тестам."""
    global _FACES, _PEOPLE
    _FACES = None
    _PEOPLE = None
