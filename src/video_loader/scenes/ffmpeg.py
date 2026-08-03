"""Фоллбек-детектор сцен: фильтр FFmpeg ``scdet``.

Зачем вторая реализация, когда есть PySceneDetect:

1. **Отсутствие зависимостей.** Здесь не нужны ни OpenCV, ни NumPy — только
   ffmpeg, который в образе есть по любому другому поводу. Когда сборка
   с OpenCV ломается (а с ним это регулярно), конвейер продолжает работать.
2. **Скорость на длинных файлах.** Декодирование идёт внутри ffmpeg, без
   передачи каждого кадра в Python. На двухчасовом фильме разница заметна.
3. **Проверка контракта.** Две независимые реализации одного порта — лучший
   способ убедиться, что порт описывает поведение, а не одну библиотеку.
   Тесты гоняют обе через один и тот же набор проверок.

Расплата — качество: ``scdet`` считает разницу проще и на плавных переходах
ошибается чаще. Поэтому он фоллбек, а не основной вариант.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from typing import Final

from core.domain.errors import FFmpegError, TransientError
from core.domain.media import LocalMedia
from core.domain.values import MediaWindow, TimeRange
from core.logging import get_logger
from video_loader.scenes.postprocess import normalize_scenes

__all__ = ["FFmpegSceneDetector", "parse_scdet_output", "scdet_threshold"]

log = get_logger(__name__)

#: Максимальная шкала различия у PySceneDetect (разница по HSV, 0..255)
#: и у scdet (проценты, 0..100). Переводим одно в другое, чтобы общий
#: SCENE_THRESHOLD означал одну и ту же чувствительность у обеих реализаций.
_PYSCENEDETECT_SCALE: Final = 255.0
_SCDET_SCALE: Final = 100.0

#: Строки метаданных, которые печатает `metadata=print`:
#:     frame:42   pts:42042   pts_time:1.68
#:     lavfi.scd.time=1.68
#:     lavfi.scd.score=18.462
_TIME_PATTERN: Final = re.compile(r"lavfi\.scd\.time\s*[:=]\s*([0-9]+\.?[0-9]*)")
_PTS_PATTERN: Final = re.compile(r"\bpts_time[:=]\s*([0-9]+\.?[0-9]*)")


def scdet_threshold(threshold: float) -> float:
    """Порог PySceneDetect (0..255) → порог scdet (0..100)."""
    scaled = threshold * _SCDET_SCALE / _PYSCENEDETECT_SCALE
    return round(min(max(scaled, 0.0), _SCDET_SCALE), 3)


def parse_scdet_output(text: str) -> list[float]:
    """Вытащить моменты склеек из вывода ``metadata=print``.

    Разбор вынесен отдельной чистой функцией, чтобы его можно было проверять
    на записанном выводе — без ffmpeg и без видеофайлов. Предпочтение отдаётся
    ``lavfi.scd.time``: это момент, который назвал сам фильтр. ``pts_time``
    берётся запасным вариантом — у него время кадра, на котором сработал
    детектор, что на кадр-другой позже.
    """
    moments: list[float] = []
    pending_pts: float | None = None

    for line in text.splitlines():
        match = _TIME_PATTERN.search(line)
        if match:
            moments.append(float(match.group(1)))
            pending_pts = None
            continue

        pts = _PTS_PATTERN.search(line)
        if pts:
            # Заголовок нового кадра: предыдущий так и не назвал scd.time —
            # значит, его момент берём из pts.
            if pending_pts is not None:
                moments.append(pending_pts)
            pending_pts = float(pts.group(1))

    if pending_pts is not None:
        moments.append(pending_pts)

    return sorted(set(moments))


class FFmpegSceneDetector:
    """Реализация порта ``SceneDetector`` поверх фильтра ``scdet``."""

    name = "ffmpeg"

    def __init__(
        self,
        *,
        binary: str = "ffmpeg",
        threshold: float = 27.0,
        min_len_sec: float = 1.5,
        max_len_sec: float = 90.0,
    ) -> None:
        self._binary = binary
        self._threshold = threshold
        self._min_len_sec = min_len_sec
        self._max_len_sec = max_len_sec

    async def detect(self, media: LocalMedia, window: MediaWindow) -> Sequence[TimeRange]:
        raw = await self._run(media, window.range)
        # Моменты приходят от начала файла, а окно может начинаться не с нуля:
        # с `-ss` до `-i` ffmpeg обнуляет отсчёт, поэтому сдвигаем сами.
        cuts = [moment + window.range.start for moment in parse_scdet_output(raw)]

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

    async def _run(self, media: LocalMedia, window: TimeRange) -> str:
        # sc_pass=1 пропускает через фильтр только кадры со сменой сцены —
        # без него metadata=print печатал бы строку на каждый кадр фильма,
        # и разбор занялся бы гигабайтами текста.
        graph = (
            f"scdet=threshold={scdet_threshold(self._threshold)}:sc_pass=1,metadata=print:file=-"
        )
        args = [
            self._binary,
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-ss",
            f"{window.start:.3f}",
            "-t",
            f"{window.duration:.3f}",
            "-i",
            str(media.path),
            "-an",  # звук не декодируем: он тут не нужен и стоит времени
            "-sn",
            "-vf",
            graph,
            "-f",
            "null",
            "-",
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as error:
            raise TransientError(
                f"ffmpeg не найден: {self._binary}. Проверьте FFMPEG_BIN и образ.",
                binary=self._binary,
            ) from error

        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise FFmpegError(
                f"scdet завершился с кодом {process.returncode}: "
                f"{_first_line(stderr, str(media.path)) or 'без диагностики'}",
                returncode=process.returncode,
            )

        return stdout.decode("utf-8", errors="replace")


def _first_line(stderr: bytes, path: str) -> str:
    """Первая содержательная строка диагностики, без внутренних путей.

    Ошибка этапа доезжает до клиента через ``jobs.error``, а в пути лежит
    раскладка файловой системы сервера — ровно тот дефект, который был найден
    на Этапе 2 в сообщениях ffprobe.
    """
    text = stderr.decode("utf-8", errors="replace").replace(path, "<файл>")
    return next((line.strip() for line in text.splitlines() if line.strip()), "")[:200]
