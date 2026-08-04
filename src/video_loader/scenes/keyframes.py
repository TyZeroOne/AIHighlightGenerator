"""Извлечение превью-кадров через FFmpeg.

Кадр на сцену нужен трём потребителям: человеку (просмотреть разбиение
глазами), LLM (Этап 6 может показать модели кадр вместе с текстом) и UI
(лента сцен). Требования у всех одинаковые — небольшой JPEG, быстро.

Ключевое решение — **перемотка до `-i`**. `ffmpeg -ss 921 -i file.mp4` прыгает
по индексу к ближайшему ключевому кадру и декодирует доли секунды.
`ffmpeg -i file.mp4 -ss 921` декодирует все 921 секунду и только потом берёт
кадр. На тысяче сцен разница — между минутой и сутками.

Второе решение — параллельность. Каждый кадр это отдельный короткий процесс
ffmpeg; последовательный цикл на 1200 сценах упирается в накладные расходы
запуска. Семафор ограничивает число одновременных процессов, иначе тысяча
ffmpeg'ов выест всю память воркера.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

from core.domain.errors import TransientError
from core.domain.media import LocalMedia
from core.infra.proc import death_signal
from core.logging import get_logger

__all__ = ["FFmpegKeyframeExtractor"]

log = get_logger(__name__)


class FFmpegKeyframeExtractor:
    """Реализация порта ``KeyframeExtractor``."""

    def __init__(
        self,
        *,
        binary: str = "ffmpeg",
        width: int = 640,
        max_parallel: int = 8,
        timeout_sec: float = 30.0,
    ) -> None:
        self._binary = binary
        self._width = width
        self._max_parallel = max(max_parallel, 1)
        self._timeout = timeout_sec

    async def extract(
        self,
        media: LocalMedia,
        moments: Sequence[float],
        destination: Path,
    ) -> Sequence[Path | None]:
        if not moments:
            return []

        destination.mkdir(parents=True, exist_ok=True)
        semaphore = asyncio.Semaphore(self._max_parallel)

        async def grab(index: int, moment: float) -> Path | None:
            async with semaphore:
                return await self._grab_one(media, moment, destination / f"{index:06d}.jpg")

        frames = await asyncio.gather(
            *(grab(index, moment) for index, moment in enumerate(moments))
        )

        failed = sum(1 for frame in frames if frame is None)
        if failed:
            # Не ошибка этапа: превью — вспомогательный артефакт. Но молчать
            # тоже нельзя, иначе «почему у половины сцен нет кадра» останется
            # без ответа.
            log.warning("keyframes.partial", requested=len(moments), failed=failed)

        return frames

    async def _grab_one(self, media: LocalMedia, moment: float, target: Path) -> Path | None:
        args = [
            self._binary,
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-ss",
            f"{max(moment, 0.0):.3f}",
            "-i",
            str(media.path),
            "-frames:v",
            "1",
            "-an",
            "-vf",
            # -1 по высоте сохраняет пропорции и округляет до чётного:
            # нечётная высота у JPEG допустима, но ломает часть просмотрщиков.
            f"scale={self._width}:-2:flags=fast_bilinear",
            "-q:v",
            "4",  # шкала JPEG 2..31, 4 — визуально чисто при малом размере
            "-y",
            str(target),
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                # Дочерний ffmpeg не должен пережить воркера: при отмене джоба
                # осиротевший процесс доедает ядра, и никто о нём уже не знает.
                preexec_fn=death_signal(),
            )
        except FileNotFoundError as error:
            # Отсутствие бинарника — не «кадр не получился», а сломанное
            # окружение: это должно валить этап, а не тихо давать None.
            raise TransientError(
                f"ffmpeg не найден: {self._binary}. Проверьте FFMPEG_BIN и образ.",
                binary=self._binary,
            ) from error

        try:
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=self._timeout)
        except TimeoutError:
            process.kill()
            await process.wait()
            log.warning("keyframe.timeout", moment=round(moment, 2))
            return None

        if process.returncode != 0 or not target.exists() or target.stat().st_size == 0:
            log.debug(
                "keyframe.failed",
                moment=round(moment, 2),
                returncode=process.returncode,
                error=stderr.decode("utf-8", errors="replace")[:120],
            )
            target.unlink(missing_ok=True)
            return None

        return target
