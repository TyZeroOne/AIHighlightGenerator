"""Основной детектор сцен — PySceneDetect.

Два детектора одновременно, и это не «на всякий случай»:

* ``ContentDetector`` считает разницу соседних кадров в HSV и ловит **резкие
  склейки** — их в смонтированном видео большинство;
* ``AdaptiveDetector`` сравнивает кадр со скользящим средним и ловит то,
  на чём первый ошибается: плавные переходы, наплывы, а главное — не срабатывает
  на быстрой панораме, где яркость меняется скачком без всякой склейки.

``SceneManager`` умеет держать несколько детекторов и объединяет их точки
склеек. Ложные срабатывания второго детектора нам обходятся дешевле пропусков:
лишняя граница режет сцену пополам (обе половины остаются в анализе), а
пропущенная склеивает два разных плана в один — и скоринг усредняет их вместе.

Библиотека синхронная и считает долго, поэтому вся работа уходит
в ``asyncio.to_thread``: держать event loop заблокированным на минуты нельзя —
в этом же процессе живут отмена задачи и heartbeat воркера.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from core.domain.errors import ProcessingError
from core.domain.media import LocalMedia
from core.domain.values import MediaWindow, TimeRange
from core.logging import get_logger
from video_loader.scenes.postprocess import normalize_scenes

__all__ = ["PySceneDetectDetector"]

log = get_logger(__name__)


class PySceneDetectDetector:
    """Реализация порта ``SceneDetector`` поверх PySceneDetect."""

    name = "pyscenedetect"

    def __init__(
        self,
        *,
        threshold: float = 27.0,
        min_len_sec: float = 1.5,
        max_len_sec: float = 90.0,
        downscale: int = 0,
        use_adaptive: bool = True,
    ) -> None:
        self._threshold = threshold
        self._min_len_sec = min_len_sec
        self._max_len_sec = max_len_sec
        self._downscale = downscale
        self._use_adaptive = use_adaptive

    async def detect(self, media: LocalMedia, window: MediaWindow) -> Sequence[TimeRange]:
        cuts = await asyncio.to_thread(self._detect_cuts, media, window.range)
        scenes = normalize_scenes(
            cuts,
            window.range,
            min_len_sec=self._min_len_sec,
            max_len_sec=self._max_len_sec,
        )
        log.info(
            "scenes.detected",
            detector=self.name,
            cuts=len(cuts),
            scenes=len(scenes),
            window_sec=round(window.range.duration, 2),
        )
        return scenes

    # ------------------------------------------------------------ внутреннее
    def _detect_cuts(self, media: LocalMedia, window: TimeRange) -> list[float]:
        """Синхронная часть: открыть файл и прогнать детекторы.

        Исполняется в отдельном потоке. Импорты — внутри функции: OpenCV и NumPy
        грузятся секунду и занимают десятки мегабайт, а нужны они только тому
        процессу, который реально считает сцены.
        """
        try:
            from scenedetect import AdaptiveDetector, ContentDetector, SceneManager, open_video
        except ImportError as error:  # pragma: no cover — зависит от окружения
            raise ProcessingError(
                "PySceneDetect не установлен. Поставьте extra 'worker' "
                "или переключитесь на SCENE_DETECTOR=ffmpeg",
            ) from error

        try:
            video = open_video(str(media.path))
        except Exception as error:
            raise ProcessingError(
                f"Не удалось открыть видео для разбиения на сцены: {type(error).__name__}",
                path=media.path.name,
            ) from error

        fps = float(video.frame_rate) or media.info.fps or 25.0
        min_scene_len = max(int(self._min_len_sec * fps), 1)

        # Библиотека пишет «Detecting scenes...» и прогресс через стандартный
        # logging на уровне INFO. В структурных логах воркера это мусор:
        # своё сообщение о результате мы пишем сами, с job_id и trace_id.
        logging.getLogger("pyscenedetect").setLevel(logging.WARNING)

        manager = SceneManager()
        if self._downscale > 0:
            manager.auto_downscale = False
            manager.downscale = self._downscale

        manager.add_detector(
            ContentDetector(threshold=self._threshold, min_scene_len=min_scene_len)
        )
        if self._use_adaptive:
            # adaptive_threshold в шкале «во сколько раз кадр отличается от
            # скользящего среднего» — она независима от threshold контентного
            # детектора, и переносить туда наши 27 нельзя. Значение по умолчанию
            # (3.0) библиотека подобрала на своём наборе, менять его без
            # собственных замеров смысла нет.
            manager.add_detector(AdaptiveDetector(min_scene_len=min_scene_len))

        if window.start > 0:
            video.seek(window.start)

        try:
            manager.detect_scenes(
                video=video,
                end_time=window.end,
                show_progress=False,
            )
            scene_list: Sequence[tuple[Any, Any]] = manager.get_scene_list()
        except Exception as error:
            raise ProcessingError(
                f"Разбиение на сцены не удалось: {type(error).__name__}: {error}",
                path=media.path.name,
                detector=self.name,
            ) from error

        # Берём начала сцен, кроме первой: первое «начало» — это начало окна,
        # а не склейка. get_scene_list, а не get_cut_list: второй объявлен
        # устаревшим в 0.7 и печатает предупреждение на каждый вызов.
        return [_seconds(start) for start, _ in scene_list[1:]]


def _seconds(timecode: Any) -> float:
    """Секунды из ``FrameTimecode`` независимо от версии библиотеки.

    В 0.7 появилось свойство ``seconds``, а метод ``get_seconds()`` объявлен
    устаревшим. Поддерживаем оба: диапазон версий в ``pyproject`` открытый,
    и обновление минорной версии не должно ронять этап предупреждениями.
    """
    value = getattr(timecode, "seconds", None)
    if value is None:
        value = timecode.get_seconds()
    return float(value)
