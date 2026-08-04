"""Движение в кадре и движение самой камеры.

Различать эти две вещи принципиально. Оптический поток одинаково велик
и когда по неподвижному кадру бежит человек, и когда камера панорамирует
по пустой улице. Для отбора моментов это разные события: первое — действие,
второе — перебивка.

Поэтому считается и то и другое:

* **motion.mean / motion.peak** — плотный оптический поток Farnebäck.
  Сколько вообще движения в кадре;
* **camera.pan / camera.zoom / camera.shake** — глобальное преобразование
  между кадрами, оценённое по особым точкам. Панорама, наезд и дрожание
  ручной камеры — это свойства съёмки, а не содержания.

Поток считается на уменьшенных кадрах (док 05, §5.1 — «на 1/4 разрешении»):
магнитуда движения от масштаба почти не зависит, если нормировать её
на размер кадра, а времени экономится вчетверо.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Final

from core.domain.analysis import ARTIFACT_VIDEO, AnalysisContext
from core.domain.values import FeatureValue, TimeRange
from core.logging import get_logger
from core.registry import register_analyzer
from vision.context import source_for
from vision.frames import FrameSource

__all__ = ["CameraMotion", "MotionAnalyzer", "camera_motion", "flow_magnitude"]

log = get_logger(__name__)

#: Минимум совпавших точек, при котором оценке преобразования можно верить.
#: На однотонном кадре (небо, стена) точек не находится вовсе, и три пары —
#: это уже не оценка, а фантазия.
_MIN_MATCHES: Final = 12


class CameraMotion:
    """Оценка движения камеры между двумя кадрами."""

    __slots__ = ("pan", "shake", "zoom")

    def __init__(self, pan: float, zoom: float, shake: float) -> None:
        self.pan = pan
        self.zoom = zoom
        self.shake = shake


def flow_magnitude(previous: Any, current: Any) -> float:
    """Средняя магнитуда оптического потока, нормированная на размер кадра.

    Нормировка обязательна: на кадре 320 пикселей сдвиг в 10 пикселей —
    это треть экрана, а на 1920 — незаметное шевеление. Без деления
    на ширину признак зависел бы от разрешения исходника.
    """
    import cv2
    import numpy as np

    # Буфер под результат заводим сами: обёртка принимает и None, но тогда
    # тип возвращаемого значения перестаёт быть определённым для проверки.
    buffer = np.zeros((*previous.shape[:2], 2), dtype=np.float32)
    # Аргументы позиционные: у обёртки OpenCV нет именованных параметров.
    # Значения — рекомендованные документацией: пирамида в три уровня,
    # окно 15 пикселей, три итерации на уровень.
    flow = cv2.calcOpticalFlowFarneback(previous, current, buffer, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    width = max(int(previous.shape[1]), 1)
    return round(float(np.mean(magnitude)) / width, 6)


def camera_motion(previous: Any, current: Any) -> CameraMotion | None:
    """Панорамирование, наезд и дрожание между кадрами.

    Оценивается аффинное преобразование по особым точкам ORB. ``None``
    означает «не удалось»: на однотонном кадре точек нет, и придумывать
    за камеру мы не будем — отсутствующий признак честнее нуля, который
    скоринг примет за «камера неподвижна».
    """
    import cv2
    import numpy as np

    detector = cv2.ORB.create(nfeatures=400)
    first_points, first_desc = detector.detectAndCompute(previous, None)
    second_points, second_desc = detector.detectAndCompute(current, None)
    if first_desc is None or second_desc is None:
        return None
    if len(first_points) < _MIN_MATCHES or len(second_points) < _MIN_MATCHES:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(first_desc, second_desc), key=lambda item: item.distance)
    if len(matches) < _MIN_MATCHES:
        return None

    source = np.array([first_points[m.queryIdx].pt for m in matches], dtype=np.float32).reshape(
        -1, 1, 2
    )
    target = np.array([second_points[m.trainIdx].pt for m in matches], dtype=np.float32).reshape(
        -1, 1, 2
    )

    # Частичное аффинное: сдвиг, поворот и равномерный масштаб. Полного
    # здесь не нужно — скос и неравномерный масштаб камера не создаёт,
    # а лишние степени свободы делают оценку шумной.
    matrix, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC)
    if matrix is None or inliers is None or int(inliers.sum()) < _MIN_MATCHES:
        return None

    width = max(int(previous.shape[1]), 1)
    height = max(int(previous.shape[0]), 1)
    shift_x, shift_y = float(matrix[0, 2]), float(matrix[1, 2])
    scale = float(np.hypot(float(matrix[0, 0]), float(matrix[0, 1])))

    pan = round(float(np.hypot(shift_x / width, shift_y / height)), 6)
    zoom = round(abs(scale - 1.0), 6)
    # Дрожание — это движение, которое **не** объясняется общим сдвигом:
    # разброс невязок вокруг найденного преобразования.
    predicted = cv2.transform(source, matrix)
    residuals = np.linalg.norm(predicted - target, axis=2).ravel()
    shake = round(float(residuals.std()) / width, 6)

    return CameraMotion(pan=pan, zoom=zoom, shake=shake)


@register_analyzer
class MotionAnalyzer:
    """Реализация порта ``FeatureAnalyzer`` для движения."""

    key: ClassVar[str] = "vision.motion"
    requires: ClassVar[frozenset[str]] = frozenset({ARTIFACT_VIDEO})
    produces: ClassVar[frozenset[str]] = frozenset(
        {
            "vision.motion.mean",
            "vision.motion.peak",
            "vision.camera.pan",
            "vision.camera.zoom",
            "vision.camera.shake",
        }
    )
    #: Самый дорогой в группе: плотный поток и поиск особых точек
    #: на каждой паре кадров.
    cost: ClassVar[int] = 5

    def __init__(self, *, source: FrameSource | None = None) -> None:
        self._source = source

    async def analyze(
        self,
        context: AnalysisContext,
        scenes: Sequence[TimeRange],
    ) -> Mapping[int, Sequence[FeatureValue]]:
        source = self._source or source_for(context)
        if source is None or not scenes:
            return {}

        features: dict[int, list[FeatureValue]] = {}
        for index, scene in enumerate(scenes):
            batch = await source.scene_frames(scene)
            if len(batch) < 2:
                # На одном кадре движения не бывает по определению.
                continue

            values = _scene_features(batch.frames)
            if values:
                features[index] = values

        log.debug("motion.done", scenes=len(features))
        return features


def _scene_features(frames: Sequence[Any]) -> list[FeatureValue]:
    import numpy as np

    magnitudes = [flow_magnitude(frames[i - 1], frames[i]) for i in range(1, len(frames))]
    values = [
        FeatureValue("vision.motion.mean", round(float(np.mean(magnitudes)), 6)),
        # Пик важен отдельно: удар или прыжок — это одна пара кадров,
        # и в среднем по спокойной сцене такое событие исчезает.
        FeatureValue("vision.motion.peak", round(float(np.max(magnitudes)), 6)),
    ]

    camera = [
        motion
        for i in range(1, len(frames))
        if (motion := camera_motion(frames[i - 1], frames[i])) is not None
    ]
    if camera:
        values.append(
            FeatureValue("vision.camera.pan", round(float(np.mean([m.pan for m in camera])), 6))
        )
        values.append(
            FeatureValue("vision.camera.zoom", round(float(np.mean([m.zoom for m in camera])), 6))
        )
        values.append(
            FeatureValue("vision.camera.shake", round(float(np.mean([m.shake for m in camera])), 6))
        )

    return values
