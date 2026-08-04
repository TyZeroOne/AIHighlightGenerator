"""Запуск ffmpeg и ffprobe: одно место на весь пакет.

Каждый модуль нарезки запускает внешний процесс, и без общей обёртки
обработка ошибок расползлась бы по трём файлам в трёх вариантах — а она
здесь неочевидная. ffmpeg пишет диагностику в stderr, возвращает 0 при
частичном успехе и охотно печатает в лог полный путь к исходнику
пользователя, которому в наших логах и ошибках API делать нечего.

Своя обёртка, а не общая с ``audio`` или ``video_loader``: адаптеры
не импортируют друг друга (правило зависимостей, док 02, §2.2).
Дублирование тридцати строк дешевле связи, из-за которой нельзя вынуть
пакет нарезки, не потянув за собой распознавание речи.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from core.domain.errors import FFmpegError, TransientError
from core.logging import get_logger

__all__ = [
    "FFMPEG_BASE",
    "keyframe_times",
    "probe_duration",
    "probe_json",
    "run_ffmpeg",
]

log = get_logger(__name__)

#: Общее начало любой команды. ``-nostdin`` обязателен: без него ffmpeg,
#: запущенный из воркера, пытается читать управляющие символы с консоли
#: и виснет на первом же вопросе о перезаписи файла.
FFMPEG_BASE: tuple[str, ...] = ("-hide_banner", "-nostdin", "-v", "error")


async def run_ffmpeg(
    binary: str,
    args: Sequence[str],
    *,
    destination: Path,
    action: str,
    timeout_sec: float | None = None,
) -> Path:
    """Запустить ffmpeg и убедиться, что файл действительно получился.

    Кода возврата мало. ffmpeg завершается нулём, обработав ноль кадров,
    если фильтр отбросил всё, — и на выходе остаётся файл в 48 байт,
    который дальше по конвейеру притворяется валидным видео. Поэтому
    проверяется и существование, и размер.
    """
    command = [binary, *FFMPEG_BASE, *args]

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as error:
        raise TransientError(
            f"ffmpeg не найден: {binary}. Проверьте FFMPEG_BIN и образ.",
            binary=binary,
        ) from error

    try:
        _, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_sec)
    except TimeoutError:
        process.kill()
        await process.wait()
        destination.unlink(missing_ok=True)
        raise FFmpegError(
            f"{action}: ffmpeg не уложился в {timeout_sec:.0f} с и был остановлен",
            action=action,
        ) from None

    if process.returncode != 0:
        destination.unlink(missing_ok=True)
        raise FFmpegError(
            f"{action}: ffmpeg завершился с кодом {process.returncode}. "
            f"{_diagnostics(stderr) or 'Без диагностики.'}",
            action=action,
            returncode=process.returncode,
        )

    if not destination.exists() or destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        raise FFmpegError(f"{action}: ffmpeg отработал, но файл пуст", action=action)

    return destination


async def probe_json(binary: str, args: Sequence[str]) -> dict[str, object]:
    """Разобранный JSON-ответ ffprobe.

    Пустой словарь вместо исключения на любой сбой: ffprobe здесь нужен
    для необязательных уточнений (где ключевые кадры, какая длительность
    вышла), и ни одно из них не стоит того, чтобы уронить готовый клип.
    """
    command = [binary, "-hide_banner", "-v", "error", "-of", "json", *args]
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()
    except (FileNotFoundError, OSError) as error:
        log.warning("ffprobe.unavailable", binary=binary, error=str(error))
        return {}

    if process.returncode != 0 or not stdout:
        return {}

    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:  # pragma: no cover — ffprobe отдаёт валидный JSON
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def probe_duration(binary: str, path: Path) -> float | None:
    """Длительность готового файла по контейнеру.

    Именно по контейнеру, а не по числу кадров: проверка «клип вышел той
    длины, которую заказывали» должна опираться на то же число, которое
    увидит плеер и площадка.
    """
    payload = await probe_json(
        binary,
        ["-show_entries", "format=duration", str(path)],
    )
    fmt = payload.get("format")
    if not isinstance(fmt, dict):
        return None
    try:
        return float(fmt["duration"])
    except (KeyError, TypeError, ValueError):
        return None


async def keyframe_times(
    binary: str,
    path: Path,
    *,
    start: float,
    end: float,
) -> list[float]:
    """Моменты ключевых кадров в интервале.

    ``-skip_frame nokey`` заставляет декодер пропускать всё, кроме ключевых
    кадров, а ``-read_intervals`` ограничивает чтение нужным куском: без
    этого запрос к двухчасовому файлу занял бы минуты вместо миллисекунд.
    """
    payload = await probe_json(
        binary,
        [
            "-select_streams",
            "v:0",
            "-skip_frame",
            "nokey",
            "-show_entries",
            "frame=pts_time",
            "-read_intervals",
            f"{max(start - 1.0, 0.0):.3f}%{end + 1.0:.3f}",
            str(path),
        ],
    )
    frames = payload.get("frames")
    if not isinstance(frames, list):
        return []

    times: list[float] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        try:
            times.append(float(frame["pts_time"]))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(times)


def _diagnostics(stderr: bytes) -> str:
    """Последние строки stderr без путей файловой системы.

    Последние, а не первые: полезное ffmpeg говорит в конце, а начало
    занято предупреждениями о неподдерживаемых атомах контейнера.
    Пути вырезаются целиком — сообщение об ошибке уходит в API и в логи,
    и раскрывать в нём устройство файловой системы сервера незачем.
    """
    text = stderr.decode("utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " | ".join(_strip_paths(line) for line in lines[-3:])[:300]


def _strip_paths(line: str) -> str:
    return " ".join("<путь>" if _looks_like_path(token) else token for token in line.split(" "))


def _looks_like_path(token: str) -> bool:
    stripped = token.strip("'\"")
    return stripped.startswith(("/", "\\")) or (len(stripped) > 2 and stripped[1:3] == ":\\")
