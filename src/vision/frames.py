"""Общий источник кадров для видеоанализаторов.

Главная проблема этой группы — не алгоритмы, а декодирование. Двухчасовой
фильм в 1080p — это 180 тысяч кадров; полный проход по нему занимает минуты,
и если каждый из трёх анализаторов декодирует файл сам, минуты превращаются
в десятки. Бюджет этапа (док 10: ≤ 20 мин на GPU) на это не рассчитан.

Отсюда два решения.

**Сэмплирование.** Признакам вроде «средняя магнитуда движения» не нужны
все кадры: два-четыре в секунду описывают сцену не хуже, а работы в десять
раз меньше. Частота настраивается (``VISION_SAMPLE_FPS``).

**Общий кэш.** Анализаторы одного слоя идут по сценам в одном порядке
и просят одинаковые кадры, поэтому декодирование происходит один раз,
а не по разу на анализатор.

Кэш держит несколько последних сцен, а не одну. Одной хватило бы, только
если бы анализаторы шли строго в ногу; на деле они работают с разной
скоростью (плотный оптический поток на порядок дороже гистограмм)
и разъезжаются на сцену-другую. Замер на тесте: с кэшем на одну сцену
файл декодировался двенадцать раз вместо четырёх.

Размер кэша ограничен: сцена при 320 пикселях и трёх кадрах в секунду —
это меньше мегабайта, но двухчасовое видео целиком не поместилось бы
ни в какой бюджет памяти.

Кадры отдаются **уменьшенными**, в двух видах сразу: в оттенках серого
(поток, гистограммы, энтропия) и цветными (детектор лиц). Полное разрешение
не нужно никому из них, а памяти и времени оно стоит вчетверо.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from core.domain.errors import ProcessingError
from core.domain.values import TimeRange
from core.logging import get_logger

__all__ = ["FrameBatch", "FrameSource", "reset_sources", "sample_times", "shared_source"]

log = get_logger(__name__)

#: Больше кадров в секунду почти не добавляет информации о сцене, а время
#: растёт линейно. Значение подтверждается замером на этапе.
DEFAULT_SAMPLE_FPS: Final = 3.0

#: Длинная сторона кадра после уменьшения. 320 пикселей хватает и потоку,
#: и гистограммам; лица потребуют больше и попросят своё разрешение сами.
DEFAULT_MAX_SIDE: Final = 320

#: Сколько сцен держать в кэше. Четыре — с запасом на расхождение
#: анализаторов по скорости и всё ещё единицы мегабайт памяти.
DEFAULT_CACHE_SIZE: Final = 4


@dataclass(frozen=True, slots=True)
class FrameBatch:
    """Кадры одной сцены вместе с их временными метками.

    Хранятся оба представления: серое (``frames``) и цветное (``color``).
    Потоку, гистограммам и энтропии цвет не нужен, а вот детектору лиц —
    нужен: YuNet обучен на трёхканальном входе.

    Держать два массива дороже по памяти ровно вчетверо, но это единицы
    мегабайт на сцену — и несравнимо дешевле, чем второй проход
    по видеофайлу ради тех же самых кадров в другом цветовом формате.
    """

    scene: TimeRange
    times: tuple[float, ...]
    frames: tuple[Any, ...]
    color: tuple[Any, ...] = ()

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def is_empty(self) -> bool:
        return not self.frames


class FrameSource:
    """Выдаёт кадры сцены, декодируя файл минимально возможное число раз."""

    def __init__(
        self,
        path: Path,
        *,
        sample_fps: float = DEFAULT_SAMPLE_FPS,
        max_side: int = DEFAULT_MAX_SIDE,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        self._path = path
        self._sample_fps = max(sample_fps, 0.1)
        self._max_side = max_side
        self._cache_size = max(cache_size, 1)
        self._cache: OrderedDict[TimeRange, FrameBatch] = OrderedDict()
        # Замок, а не просто кэш: анализаторы слоя работают параллельно,
        # и без него два из них декодировали бы одну сцену одновременно —
        # то есть ровно то, ради чего кэш и заводился.
        self._lock = asyncio.Lock()

    async def scene_frames(self, scene: TimeRange) -> FrameBatch:
        """Кадры сцены, уменьшенные и в оттенках серого."""
        async with self._lock:
            cached = self._cache.get(scene)
            if cached is not None:
                self._cache.move_to_end(scene)
                return cached

            batch = await asyncio.to_thread(self._decode, scene)
            self._cache[scene] = batch
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            return batch

    # ------------------------------------------------------------ внутреннее
    def _decode(self, scene: TimeRange) -> FrameBatch:
        """Синхронное декодирование в отдельном потоке.

        Перемотка на начало сцены, затем чтение до её конца с пропуском
        лишних кадров. ``grab()`` вместо ``read()`` для пропускаемых:
        он не тратит время на преобразование кадра, который мы всё равно
        выбросим.

        Позиция считается **в кадрах**, а не в миллисекундах, и это не вкус.
        ``CAP_PROP_POS_MSEC``, прочитанный до чтения кадра, отстаёт на кадр:
        замер показал сцену [0, 3), последний кадр которой помечен 2.96 с,
        а по содержанию принадлежит уже следующей — с ложной склейкой
        в придачу. Номер кадра такой двусмысленности не имеет.

        Вторая тонкость: перемотка попадает на ближайший **ключевой** кадр,
        а он лежит раньше запрошенного — у файла с GOP в секунду это до
        секунды чужого материала. Поэтому лишнее пролистывается явно.
        """
        try:
            import cv2
        except ImportError as error:  # pragma: no cover — зависит от окружения
            raise ProcessingError(
                "OpenCV не установлен. Поставьте extra 'worker'.",
            ) from error

        capture = cv2.VideoCapture(str(self._path))
        if not capture.isOpened():
            raise ProcessingError(
                f"Не удалось открыть видео: {self._path.name}", path=self._path.name
            )

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
            step = max(round(fps / self._sample_fps), 1)

            first = int(scene.start * fps)
            last = int(scene.end * fps)
            capture.set(cv2.CAP_PROP_POS_FRAMES, first)
            # Куда перемотка попала на самом деле: до ближайшего ключевого.
            current = int(capture.get(cv2.CAP_PROP_POS_FRAMES))

            times: list[float] = []
            frames: list[Any] = []
            colored: list[Any] = []

            while current < last:
                # Хвост предыдущей сцены, доставшийся от перемотки по ключевым,
                # и кадры между сэмплами пролистываем без декодирования.
                if current < first or (current - first) % step:
                    if not capture.grab():
                        break
                    current += 1
                    continue

                ok, frame = capture.read()
                if not ok:
                    break

                small = self._prepare(cv2, frame)
                colored.append(small)
                frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
                times.append(round(current / fps, 3))
                current += 1
        finally:
            capture.release()

        return FrameBatch(
            scene=scene,
            times=tuple(times),
            frames=tuple(frames),
            color=tuple(colored),
        )

    def _prepare(self, cv2: Any, frame: Any) -> Any:
        """Уменьшенный кадр, всё ещё цветной."""
        height, width = frame.shape[:2]
        longest = max(height, width)
        if longest > self._max_side:
            scale = self._max_side / longest
            frame = cv2.resize(
                frame,
                (max(int(width * scale), 1), max(int(height * scale), 1)),
                interpolation=cv2.INTER_AREA,
            )
        return frame


#: Источники по файлам. Общее состояние — и это осознанно.
#:
#: Анализаторы создаёт реестр вызовом ``cls()``: передать им общий объект
#: через конструктор некому, а класть его в ``AnalysisContext`` нельзя —
#: домен не должен знать про OpenCV. Значит, разделение ресурса остаётся
#: внутренним делом пакета ``vision``, и это ровно то место, где ему место.
#:
#: Безопасность: воркер работает пулом процессов (prefork), в каждом
#: одновременно идёт один джоб, так что гонок между джобами нет.
_SOURCES: dict[Path, FrameSource] = {}


def shared_source(
    path: Path,
    *,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    max_side: int = DEFAULT_MAX_SIDE,
) -> FrameSource:
    """Один источник кадров на файл — общий для всех видеоанализаторов.

    Без него каждый анализатор декодировал бы двухчасовой фильм заново:
    три полных прохода вместо одного и бюджет этапа в мусорку.
    """
    source = _SOURCES.get(path)
    if source is None:
        source = FrameSource(path, sample_fps=sample_fps, max_side=max_side)
        _SOURCES[path] = source
    return source


def reset_sources() -> None:
    """Забыть все источники. Нужно тестам и между джобами."""
    _SOURCES.clear()


def sample_times(scene: TimeRange, sample_fps: float) -> Sequence[float]:
    """Моменты, в которые предполагается взять кадры.

    Отдельная чистая функция: по ней тесты проверяют раскладку сэмплов,
    не открывая ни одного видеофайла.
    """
    step = 1.0 / max(sample_fps, 0.1)
    count = max(int(scene.duration / step), 1)
    return [scene.start + step * index for index in range(count)]
