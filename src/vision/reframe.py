"""Куда смотреть в вертикальном кадре: траектория кадрирования по лицам.

Из горизонтального кадра 16:9 в вертикальный 9:16 помещается меньше
трети ширины. Вопрос «какая именно треть» и есть вся задача: центр кадра
почти никогда не совпадает с тем, ради чего момент выбран.

## Почему это здесь, а не в ``video_editor``

Роадмап предполагал один модуль ``video_editor/vertical.py``. При
реализации выяснилось, что внутри него две разные работы: **найти**,
куда смотреть (декодирование, детектор лиц, сглаживание — компьютерное
зрение), и **применить** решение (фильтры ffmpeg). У них разные
зависимости, разная стоимость и разные причины меняться, а адаптеры
не импортируют друг друга (док 02, §2.2) — значит, детектор лиц из
``vision`` в ``video_editor`` не переиспользовать. Поэтому планирование
живёт здесь, рядом с детектором, а применение — там, рядом с ffmpeg.
Их связывает порт ``CropPlanner``.

## Три обработки траектории — и зачем каждая

Сырые наблюдения непригодны напрямую по трём независимым причинам,
и каждая лечится своим шагом.

1. **Дыры.** Актёр отвернулся, лицо не найдено, наблюдения нет.
   Прыгать в центр кадра на эти полсекунды нельзя — заполняем
   удержанием последней известной позиции.

2. **Дрожь.** Детектор даёт координаты с точностью в единицы процентов
   ширины, и кадр трясётся, даже когда человек стоит. Лечится
   скользящим средним.

3. **Рывки.** В диалоге камера должна переходить между собеседниками
   не мгновенно, иначе смотреть невозможно. Лечится ограничением
   скорости: за секунду кроп проезжает не больше заданной доли ширины.

Порядок важен: сглаживание до заполнения дыр размазало бы по соседям
случайное значение, а ограничение скорости до сглаживания оставило бы
дрожь, только медленную.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from core.config import get_settings
from core.domain.errors import ProcessingError
from core.domain.render import CropKeyframe
from core.domain.values import CropBox
from core.logging import get_logger
from vision.faces import FaceDetector, shared_face_detector

__all__ = [
    "FaceCropPlanner",
    "build_boxes",
    "crop_size",
    "fill_gaps",
    "limit_speed",
    "smooth",
    "split_on_gaps",
]

log = get_logger(__name__)

#: Насколько часто меряем положение лиц, если конфиг молчит.
DEFAULT_SAMPLE_FPS: Final = 4.0

#: Окно сглаживания по умолчанию, секунды.
DEFAULT_SMOOTH_SEC: Final = 1.2

#: Предельная скорость кропа в долях ширины кадра за секунду.
DEFAULT_MAX_SPEED: Final = 0.25

#: Ширина, до которой уменьшается кадр перед детектором. Больше не нужно:
#: YuNet обучен на 320 пикселях, а нам от него нужна не форма лица,
#: а положение по горизонтали.
DETECT_WIDTH: Final = 640

#: Лицо считается кандидатом на «главное», если оно не меньше этой доли
#: от самого большого в кадре. Иначе в диалоге кроп уезжал бы на прохожего
#: в глубине кадра ровно в тот момент, когда говорящий моргнул.
MAIN_FACE_RATIO: Final = 0.5

#: Сколько подряд наблюдений без лица считать разрывом. Два при частоте
#: 4 Гц — это полсекунды, за которые в клипе почти наверняка произошла
#: склейка: непрерывный план не теряет лицо так надолго.
CUT_GAP_SAMPLES: Final = 2


class FaceCropPlanner:
    """Реализация порта ``CropPlanner`` по детектору лиц."""

    def __init__(
        self,
        *,
        detector: FaceDetector | None = None,
        sample_fps: float | None = None,
        smooth_sec: float | None = None,
        max_speed: float | None = None,
    ) -> None:
        settings = get_settings()
        self._detector = detector
        self._sample_fps = float(
            sample_fps
            if sample_fps is not None
            else getattr(settings, "reframe_sample_fps", DEFAULT_SAMPLE_FPS)
        )
        self._smooth_sec = float(
            smooth_sec
            if smooth_sec is not None
            else getattr(settings, "reframe_smooth_sec", DEFAULT_SMOOTH_SEC)
        )
        self._max_speed = float(
            max_speed
            if max_speed is not None
            else getattr(settings, "reframe_max_speed", DEFAULT_MAX_SPEED)
        )

    async def plan(
        self,
        source: Path,
        *,
        target_width: int,
        target_height: int,
    ) -> Sequence[CropKeyframe]:
        import asyncio

        return await asyncio.to_thread(
            self._plan_sync,
            source,
            target_width,
            target_height,
        )

    # ------------------------------------------------------------ внутреннее
    def _plan_sync(
        self,
        source: Path,
        target_width: int,
        target_height: int,
    ) -> Sequence[CropKeyframe]:
        try:
            import cv2
        except ImportError as error:  # pragma: no cover — зависит от окружения
            raise ProcessingError("OpenCV не установлен. Поставьте extra 'worker'.") from error

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ProcessingError(f"Не удалось открыть клип: {source.name}", path=source.name)

        detector = self._detector or shared_face_detector()
        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
            step = max(round(fps / max(self._sample_fps, 0.1)), 1)

            times, centers = self._observe(cv2, capture, detector, fps=fps, step=step)
        finally:
            capture.release()

        if width <= 0 or height <= 0 or not times:
            return ()
        if all(center is None for center in centers):
            # Ни одного лица за весь клип — опоры для кадрирования нет.
            # Это не сбой: экшен без единого крупного плана встречается
            # постоянно, и решать, что делать дальше, будет рендер.
            log.info("reframe.no_faces", clip=source.name, samples=len(times))
            return ()

        crop_w, crop_h = crop_size(width, height, target_width, target_height)
        track = self._trajectory(centers, sample_step_sec=step / fps)
        boxes = build_boxes(track, times, frame_width=width, crop_width=crop_w, crop_height=crop_h)
        log.info(
            "reframe.planned",
            clip=source.name,
            samples=len(boxes),
            crop=f"{crop_w}x{crop_h}",
            frame=f"{width}x{height}",
            covered=round(sum(center is not None for center in centers) / len(centers), 3),
        )
        return boxes

    def _observe(
        self,
        cv2: Any,
        capture: Any,
        detector: FaceDetector,
        *,
        fps: float,
        step: int,
    ) -> tuple[list[float], list[float | None]]:
        """Наблюдения «время → центр главного лица» в долях ширины."""
        times: list[float] = []
        centers: list[float | None] = []
        index = 0
        previous: float | None = None

        while True:
            if index % step:
                if not capture.grab():
                    break
                index += 1
                continue

            ok, frame = capture.read()
            if not ok:
                break

            center = _main_face_center(cv2, detector, frame, previous)
            if center is not None:
                previous = center
            times.append(round(index / fps, 3))
            centers.append(center)
            index += 1

        return times, centers

    def _trajectory(
        self,
        centers: Sequence[float | None],
        *,
        sample_step_sec: float,
    ) -> list[float]:
        """Наблюдения → готовая траектория, кусок за куском.

        Кусками, а не целиком: длинный промежуток без лиц почти всегда
        означает склейку, а через склейку кроп должен **перепрыгивать**,
        а не переезжать. Плавный переезд между двумя разными планами —
        это движение, которого в кадре нет; зритель видит его как
        уползающую камеру.

        Замер на трейлере: доля лиц, попавших в окно, от разбиения
        почти не изменилась (93 % и 98 %) — на этом материале лица
        и так лежали близко к траектории. Изменилось **как** кроп
        доезжает: максимальный шаг вырос с 40 до 130–177 пикселей,
        и весь этот шаг теперь приходится на кадры, где лица нет.
        """
        window = max(round(self._smooth_sec / max(sample_step_sec, 1e-6)), 1)
        max_step = self._max_speed * max(sample_step_sec, 1e-6)

        track: list[float] = []
        for chunk in split_on_gaps(centers, CUT_GAP_SAMPLES):
            track.extend(limit_speed(smooth(fill_gaps(chunk), window), max_step))
        return track


def crop_size(
    width: int,
    height: int,
    target_width: int,
    target_height: int,
) -> tuple[int, int]:
    """Размер окна кадрирования под целевое соотношение сторон.

    Берём максимально возможное окно: чем оно больше, тем меньше
    масштабирования, а каждое масштабирование — потерянная резкость.
    Ширины не хватает почти всегда (16:9 → 9:16), но встречается и обратное:
    исходник уже вертикальный, и тогда ограничение приходит по высоте.
    """
    ratio = target_width / target_height
    crop_w = round(height * ratio)
    if crop_w <= width:
        return _even(crop_w), _even(height)
    crop_h = round(width / ratio)
    return _even(width), _even(min(crop_h, height))


def split_on_gaps(values: Sequence[float | None], min_gap: int) -> list[list[float | None]]:
    """Разбить наблюдения по длинным пропускам.

    Пропуск отходит **следующему** куску, а не предыдущему, и это важно:
    после заполнения он получит значение уже нового плана, то есть кроп
    переместится в тот момент, когда лица в кадре всё равно нет. Отдай мы
    пропуск предыдущему куску — прыжок пришёлся бы ровно на появление
    лица, то есть на самый заметный кадр.
    """
    if min_gap <= 0:
        return [list(values)]

    chunks: list[list[float | None]] = []
    current: list[float | None] = []
    gap = 0
    seen_value = False

    for value in values:
        if value is None:
            gap += 1
            current.append(value)
            continue

        # Разрыв закончился: если он был длинным и до него что-то было,
        # начинаем новый кусок с самого начала разрыва.
        if gap >= min_gap and seen_value:
            head = current[: len(current) - gap]
            if head:
                chunks.append(head)
            current = current[len(current) - gap :]
        gap = 0
        seen_value = True
        current.append(value)

    if current:
        chunks.append(current)
    return chunks


def fill_gaps(values: Sequence[float | None]) -> list[float]:
    """Заполнить пропуски удержанием соседнего известного значения.

    Назад — последним известным, в начале последовательности — первым
    будущим. Середина кадра как значение по умолчанию отвергнута: если
    лицо нашлось только на второй секунде, начинать всё равно надо
    оттуда, где оно окажется, а не из центра, откуда придётся ехать.
    """
    known = [value for value in values if value is not None]
    if not known:
        return [0.5] * len(values)

    result: list[float] = []
    last = known[0]
    for value in values:
        if value is not None:
            last = value
        result.append(last)
    return result


def smooth(values: Sequence[float], window: int) -> list[float]:
    """Скользящее среднее с центрированным окном.

    Центрированным, а не запаздывающим: у нас на руках весь клип целиком,
    и делать вид, что будущее неизвестно, незачем. Запаздывающее окно дало
    бы сдвиг траектории на полокна назад — кроп приезжал бы к говорящему
    уже после того, как тот замолчал.
    """
    if window <= 1 or len(values) <= 1:
        return list(values)

    half = window // 2
    result: list[float] = []
    for index in range(len(values)):
        left = max(index - half, 0)
        right = min(index + half + 1, len(values))
        chunk = values[left:right]
        result.append(sum(chunk) / len(chunk))
    return result


def limit_speed(values: Sequence[float], max_step: float) -> list[float]:
    """Ограничить приращение между соседними точками.

    После этого шага траектория гарантированно не дёргается — и это
    проверяемое числом свойство, а не впечатление: производная ограничена
    сверху по построению.
    """
    if not values or max_step <= 0:
        return list(values)

    result = [values[0]]
    for value in values[1:]:
        previous = result[-1]
        delta = max(-max_step, min(max_step, value - previous))
        result.append(previous + delta)
    return result


def build_boxes(
    centers: Sequence[float],
    times: Sequence[float],
    *,
    frame_width: int,
    crop_width: int,
    crop_height: int,
) -> list[CropKeyframe]:
    """Центры в долях ширины → окна кадрирования в пикселях.

    Окно всегда целиком внутри кадра: выехавший за край кроп ffmpeg
    не примет, а «немного за краем» на границе кадра — это ровно тот
    случай, когда говорящий стоит с краю.
    """
    limit = max(frame_width - crop_width, 0)
    boxes: list[CropKeyframe] = []
    for time_sec, center in zip(times, centers, strict=True):
        left = round(center * frame_width - crop_width / 2)
        left = max(0, min(limit, left))
        boxes.append(
            CropKeyframe(
                time_sec=max(time_sec, 0.0),
                box=CropBox(x=left, y=0, width=crop_width, height=crop_height),
            )
        )
    return boxes


def _main_face_center(
    cv2: Any,
    detector: FaceDetector,
    frame: Any,
    previous: float | None,
) -> float | None:
    """Центр главного лица кадра в долях ширины, или ``None``.

    «Главное» — самое крупное, но с оговоркой на непрерывность: среди
    сопоставимых по размеру выбирается ближайшее к предыдущему положению.
    Без этой оговорки кроп в диалоге двух лиц метался бы между ними
    по случайным колебаниям площади в пару процентов.
    """
    height, width = frame.shape[:2]
    if width <= 0:
        return None

    scale = 1.0
    if width > DETECT_WIDTH:
        scale = DETECT_WIDTH / width
        frame = cv2.resize(
            frame,
            (DETECT_WIDTH, max(round(height * scale), 1)),
            interpolation=cv2.INTER_AREA,
        )

    boxes = detector.detect(frame)
    if not boxes:
        return None

    detected_width = float(frame.shape[1])
    areas = [float(box[2] * box[3]) for box in boxes]
    largest = max(areas)
    candidates: list[tuple[float, float]] = [
        (float(box[0] + box[2] / 2) / detected_width, area)
        for box, area in zip(boxes, areas, strict=True)
        if area >= largest * MAIN_FACE_RATIO
    ]
    if previous is None:
        return max(candidates, key=lambda item: item[1])[0]
    anchor = previous
    return min(candidates, key=lambda item: abs(item[0] - anchor))[0]


def _even(value: int) -> int:
    """Ближайший чётный сверху вниз: нечётная сторона несовместима с yuv420p."""
    return max(value - (value % 2), 2)
