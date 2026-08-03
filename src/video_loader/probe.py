"""Обёртка над ``ffprobe``: разбор технических параметров медиафайла.

Почему ffprobe, а не Python-библиотека вроде ``pymediainfo`` или ``av``:
ffprobe уже есть в образе (он часть ffmpeg, который нужен для нарезки),
понимает всё, что понимает ffmpeg, и не требует биндингов к C-библиотекам,
которые ломаются при обновлении системных пакетов.
"""

from __future__ import annotations

import asyncio
import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Final

from core.domain.errors import SourceError, TransientError, UnsupportedMediaError
from core.domain.media import MediaInfo
from core.logging import get_logger

__all__ = ["FFprobe", "parse_probe"]

log = get_logger(__name__)

PROBE_TIMEOUT_SEC: Final = 60.0


class FFprobe:
    """Асинхронный вызов ``ffprobe``."""

    def __init__(self, binary: str = "ffprobe", timeout_sec: float = PROBE_TIMEOUT_SEC) -> None:
        self._binary = binary
        self._timeout = timeout_sec

    async def probe(self, path: Path) -> MediaInfo:
        """Разобрать файл. Бросает ``SourceError``, если это не медиа."""
        if not path.exists():
            raise SourceError(f"Файл не найден: {path}", path=str(path))

        raw = await self._run(path)
        return parse_probe(raw, size_bytes=path.stat().st_size)

    async def _run(self, path: Path) -> dict[str, Any]:
        # Аргументы передаются списком, а не строкой: при строке имя файла
        # вида `movie.mp4; rm -rf /` было бы выполнено оболочкой.
        # Список аргументов исключает интерпретацию оболочкой в принципе.
        args = [
            self._binary,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as error:
            raise TransientError(
                f"ffprobe не найден: {self._binary}. Проверьте FFPROBE_BIN и образ.",
                binary=self._binary,
            ) from error

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self._timeout)
        except TimeoutError:
            # Процесс нужно убить явно: без этого он останется висеть,
            # держа файловый дескриптор и место на диске.
            process.kill()
            await process.wait()
            raise SourceError(
                f"ffprobe не уложился в {self._timeout} с — файл повреждён или недоступен",
                path=str(path),
            ) from None

        if process.returncode != 0:
            # Ненулевой код на существующем читаемом файле означает одно:
            # ffprobe не смог разобрать содержимое. С точки зрения HTTP это
            # 415 «неподдерживаемый тип», а не 400 «плохой запрос» —
            # запрос-то корректный, просто файл не является медиа.
            raise UnsupportedMediaError(
                f"Файл не удалось прочитать как медиа: {_safe_reason(stderr, path)}",
                returncode=process.returncode,
            )

        try:
            return json.loads(stdout)  # type: ignore[no-any-return]
        except json.JSONDecodeError as error:
            raise UnsupportedMediaError("ffprobe вернул некорректный ответ") from error


def _safe_reason(stderr: bytes, path: Path) -> str:
    """Короткая причина отказа без внутренних путей.

    Вывод ffprobe содержит полный путь к файлу — а он у нас во временном
    каталоге вида ``/tmp/highlight-generator/incoming/<uuid>.mp4``. Отдавать
    такое клиенту незачем: это раскрывает раскладку файловой системы сервера
    и имена внутренних каталогов. Клиенту нужна причина, а не наш путь.
    """
    text = stderr.decode("utf-8", errors="replace").strip()
    if not text:
        return "ffprobe завершился с ошибкой"

    # Убираем и полный путь, и просто имя временного файла.
    text = text.replace(str(path), "<файл>").replace(path.name, "<файл>")
    # Из многострочного вывода берём первую содержательную строку:
    # остальное — детали внутренней диагностики ffmpeg.
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return first[:200] or "ffprobe завершился с ошибкой"


def parse_probe(raw: dict[str, Any], *, size_bytes: int | None = None) -> MediaInfo:
    """Превратить вывод ffprobe в ``MediaInfo``.

    Вынесено отдельной чистой функцией: разбор можно тестировать на
    записанных JSON-ответах, не запуская ffprobe и не имея видеофайлов.
    """
    streams = raw.get("streams") or []
    fmt = raw.get("format") or {}

    video = _first_stream(streams, "video")
    audio = _first_stream(streams, "audio")

    if video is None and audio is None:
        raise SourceError("В файле нет ни видео-, ни аудиопотока — это не медиафайл")

    duration = _duration(fmt, video, audio)
    if duration <= 0:
        raise SourceError(f"Некорректная длительность: {duration}")

    return MediaInfo(
        duration_sec=duration,
        width=_int(video, "width"),
        height=_int(video, "height"),
        fps=_fps(video),
        video_codec=video.get("codec_name") if video else None,
        audio_codec=audio.get("codec_name") if audio else None,
        container=_container(fmt),
        size_bytes=size_bytes or _int(fmt, "size"),
        raw=raw,
    )


def _first_stream(streams: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    """Первый поток нужного типа.

    Именно первый, а не «единственный»: в mkv часто несколько аудиодорожек
    (языки) и дорожки субтитров. Дефолтная для воспроизведения — первая.
    """
    for stream in streams:
        if stream.get("codec_type") == kind:
            # Обложка альбома лежит как видеопоток из одного кадра —
            # это не видео, и принимать такой файл за фильм нельзя.
            if kind == "video" and stream.get("disposition", {}).get("attached_pic"):
                continue
            return stream
    return None


def _duration(
    fmt: dict[str, Any],
    video: dict[str, Any] | None,
    audio: dict[str, Any] | None,
) -> float:
    """Длительность из контейнера, а при её отсутствии — из потоков.

    У некоторых mkv в заголовке контейнера длительности нет, зато она есть
    у дорожек. Берём максимум: аудио- и видеодорожки могут кончаться в разное
    время, а нам нужна длительность файла целиком.
    """
    candidates = [_float(fmt, "duration")]
    candidates += [_float(stream, "duration") for stream in (video, audio) if stream]
    values = [value for value in candidates if value is not None and value > 0]
    return max(values) if values else 0.0


def _fps(video: dict[str, Any] | None) -> float | None:
    """Частота кадров из ``avg_frame_rate`` вида ``"24000/1001"``.

    ``avg_frame_rate``, а не ``r_frame_rate``: второе — это «базовая» частота
    таймкодов, которая у файлов с переменным fps сильно завышена
    (нередко 1000/1) и для расчёта числа кадров бесполезна.
    """
    if not video:
        return None
    raw = video.get("avg_frame_rate") or video.get("r_frame_rate")
    if not raw or raw in {"0/0", "N/A"}:
        return None
    try:
        value = float(Fraction(raw))
    except (ValueError, ZeroDivisionError):
        return None
    return value if value > 0 else None


def _container(fmt: dict[str, Any]) -> str | None:
    """Нормализованное имя контейнера.

    ffprobe отдаёт список совместимых форматов: ``"mov,mp4,m4a,3gp,3g2,mj2"``.
    Берём первый и приводим к нашим именам — mp4-файл ffprobe опознаёт
    как ``mov``, и без нормализации проверка по белому списку сломалась бы.
    """
    name = str(fmt.get("format_name") or "").split(",")[0].strip().lower()
    return _CONTAINER_ALIASES.get(name, name) or None


_CONTAINER_ALIASES: Final[dict[str, str]] = {
    "mov": "mp4",
    "m4a": "mp4",
    "matroska": "mkv",
    "webm": "mkv",
}


def _int(source: dict[str, Any] | None, key: str) -> int | None:
    if not source or source.get(key) in (None, "N/A"):
        return None
    try:
        return int(source[key])
    except (TypeError, ValueError):
        return None


def _float(source: dict[str, Any] | None, key: str) -> float | None:
    if not source or source.get(key) in (None, "N/A"):
        return None
    try:
        return float(source[key])
    except (TypeError, ValueError):
        return None
