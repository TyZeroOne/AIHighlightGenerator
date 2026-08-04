"""Вертикальная версия клипа: 1080×1920 из горизонтального исходника.

Применение готового решения, а не его поиск: куда смотреть — считает
планировщик (порт ``CropPlanner``, реализация — в пакете ``vision``),
здесь только ffmpeg.

## Три режима и почему их три

**`track`** — окно кадрирования едет по траектории. Основной режим,
когда в клипе есть лица.

**`static`** — окно неподвижно. Включается, когда траектория почти
не меняется: разброс в пару пикселей не стоит ни файла команд, ни риска
подёргивания на округлениях. Заодно это самый дешёвый путь.

**`blurpad`** — кадр целиком по центру, сверху и снизу он же, размытый
и растянутый на весь экран. Фоллбек, когда лиц нет вовсе. Альтернатива —
кроп по центру — отвергнута: в экшен-сцене без лиц центр кадра ничем
не лучше любого другого места, а обрезка гарантированно теряет две трети
изображения. Блюр-пад не теряет ничего, и зритель сразу видит, что это
горизонтальный кадр, а не неудачное кадрирование.

## Как траектория попадает в ffmpeg

Фильтр `crop` умеет менять параметры на ходу по командам фильтра
`sendcmd`, который читает их из файла: «на секунде 1.24 поставь x = 312».

Команды выписываются **на каждый кадр**, а не на каждую точку траектории.
Это не избыточность: `sendcmd` не интерполирует, он переставляет окно
ступенькой. Точки считаются четыре раза в секунду, и без уплотнения кроп
дёргался бы четыре раза в секунду ровно теми скачками, которые
ограничитель скорости старательно ограничивал. Между точками положение
интерполируется линейно — в этом и смысл того, что траектория гладкая.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Final

from core.domain.render import VERTICAL_HEIGHT, VERTICAL_WIDTH, CropKeyframe, RenderSpec
from core.logging import get_logger
from video_editor.encoder import video_args
from video_editor.process import probe_json, run_ffmpeg

__all__ = [
    "BLUR_SIGMA",
    "MODE_BLURPAD",
    "MODE_STATIC",
    "MODE_TRACK",
    "FFmpegVerticalRenderer",
    "blurpad_chain",
    "interpolate",
    "sendcmd_script",
    "static_chain",
]

log = get_logger(__name__)

MODE_TRACK: Final = "track"
MODE_STATIC: Final = "static"
MODE_BLURPAD: Final = "blurpad"

#: Разброс траектории в пикселях, ниже которого движение не имеет смысла.
STATIC_THRESHOLD: Final = 2

#: Радиус размытия фона в фоллбеке. 25 — фон гарантированно нечитаем,
#: но остаётся цветовым продолжением кадра, а не серой заливкой.
BLUR_SIGMA: Final = 25

#: Частота команд кропа. Не ниже 25 — иначе движение снова становится
#: ступенчатым на плавных участках.
MIN_COMMAND_FPS: Final = 25.0

VERTICAL_TIMEOUT_SEC: Final = 1800.0


class FFmpegVerticalRenderer:
    """Сборка вертикального клипа. Часть реализации порта ``ClipRenderer``."""

    def __init__(
        self,
        *,
        binary: str = "ffmpeg",
        probe_binary: str = "ffprobe",
        width: int = VERTICAL_WIDTH,
        height: int = VERTICAL_HEIGHT,
    ) -> None:
        self._binary = binary
        self._probe = probe_binary
        self._width = width
        self._height = height

    async def reframe_vertical(
        self,
        source: Path,
        track: Sequence[CropKeyframe],
        spec: RenderSpec,
        destination: Path,
    ) -> tuple[Path, str]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        mode = _choose_mode(track)

        if mode is MODE_BLURPAD:
            args = self._blurpad_args(source, spec, destination)
        elif mode is MODE_STATIC:
            args = self._static_args(source, track, spec, destination)
        else:
            args = await self._track_args(source, track, spec, destination)

        await run_ffmpeg(
            self._binary,
            args,
            destination=destination,
            action=f"вертикаль {self._width}×{self._height}",
            timeout_sec=VERTICAL_TIMEOUT_SEC,
        )
        log.info("clip.vertical", mode=mode, source=source.name)
        return destination, mode

    # ------------------------------------------------------------ режимы
    def _blurpad_args(self, source: Path, spec: RenderSpec, destination: Path) -> list[str]:
        chain = blurpad_chain(self._width, self._height)
        return [
            "-i",
            str(source),
            "-filter_complex",
            chain,
            "-map",
            "[v]",
            *_audio_passthrough(),
            *_video_only(spec),
            *_container(spec),
            "-y",
            str(destination),
        ]

    def _static_args(
        self,
        source: Path,
        track: Sequence[CropKeyframe],
        spec: RenderSpec,
        destination: Path,
    ) -> list[str]:
        box = track[len(track) // 2].box
        chain = static_chain(box.x, box.y, box.width, box.height, self._width, self._height)
        return [
            "-i",
            str(source),
            *_maps(),
            *_video_only(spec, filter_chain=chain),
            *_container(spec),
            "-y",
            str(destination),
        ]

    async def _track_args(
        self,
        source: Path,
        track: Sequence[CropKeyframe],
        spec: RenderSpec,
        destination: Path,
    ) -> list[str]:
        fps = max(await _source_fps(self._probe, source), MIN_COMMAND_FPS)
        dense = interpolate(track, fps)
        script = destination.with_suffix(".crop.cmd")
        script.write_text(sendcmd_script(dense), encoding="ascii")

        first = track[0].box
        chain = (
            f"sendcmd=f='{_escape(script)}',"
            f"crop=w={first.width}:h={first.height}:x={first.x}:y={first.y},"
            f"scale={self._width}:{self._height}:flags=bicubic,setsar=1"
        )
        return [
            "-i",
            str(source),
            *_maps(),
            *_video_only(spec, filter_chain=chain),
            *_container(spec),
            "-y",
            str(destination),
        ]


def _choose_mode(track: Sequence[CropKeyframe]) -> str:
    if not track:
        return MODE_BLURPAD
    xs = [keyframe.box.x for keyframe in track]
    return MODE_STATIC if max(xs) - min(xs) <= STATIC_THRESHOLD else MODE_TRACK


def interpolate(track: Sequence[CropKeyframe], fps: float) -> list[tuple[float, int]]:
    """Уплотнить траекторию до частоты кадров, линейно между точками.

    Возвращает пары «время, x»: вертикаль двигается только по горизонтали,
    а высота окна и так равна высоте кадра.
    """
    if not track:
        return []
    if len(track) == 1:
        return [(track[0].time_sec, track[0].box.x)]

    step = 1.0 / max(fps, 1.0)
    result: list[tuple[float, int]] = []
    index = 0
    time_sec = track[0].time_sec
    end = track[-1].time_sec

    while time_sec <= end + 1e-9:
        while index + 1 < len(track) - 1 and track[index + 1].time_sec <= time_sec:
            index += 1
        left, right = track[index], track[index + 1]
        span = right.time_sec - left.time_sec
        weight = 0.0 if span <= 0 else (time_sec - left.time_sec) / span
        weight = max(0.0, min(1.0, weight))
        value = left.box.x + (right.box.x - left.box.x) * weight
        result.append((round(time_sec, 3), round(value)))
        time_sec += step

    return result


def sendcmd_script(points: Sequence[tuple[float, int]]) -> str:
    """Файл команд для фильтра ``sendcmd``.

    Повторы отбрасываются: на неподвижных участках x не меняется десятками
    кадров подряд, и переставлять его в то же значение — только раздувать
    файл, который ffmpeg разбирает целиком при старте.
    """
    lines: list[str] = []
    previous: int | None = None
    for time_sec, value in points:
        if value == previous:
            continue
        lines.append(f"{max(time_sec, 0.0):.3f} crop x {value};")
        previous = value
    return "\n".join(lines) + "\n"


def static_chain(
    x: int,
    y: int,
    width: int,
    height: int,
    target_width: int,
    target_height: int,
) -> str:
    return (
        f"crop=w={width}:h={height}:x={x}:y={y},"
        f"scale={target_width}:{target_height}:flags=bicubic,setsar=1"
    )


def blurpad_chain(target_width: int, target_height: int) -> str:
    """Кадр по центру, размытая его же копия на весь экран сзади.

    ``force_original_aspect_ratio=increase`` для фона и ``decrease``
    для переднего плана — не симметричная пара опечаток: фон обязан
    покрыть экран целиком (лишнее срезается), а передний план — влезть
    целиком (лишнее остаётся фоном).
    """
    return (
        "[0:v]split=2[bg][fg];"
        f"[bg]scale={target_width}:{target_height}:force_original_aspect_ratio=increase,"
        f"crop={target_width}:{target_height},gblur=sigma={BLUR_SIGMA}[bgb];"
        f"[fg]scale={target_width}:{target_height}:force_original_aspect_ratio=decrease[fgs];"
        "[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1[v]"
    )


def _video_only(spec: RenderSpec, *, filter_chain: str | None = None) -> list[str]:
    """Кодек и фильтры без масштабирования из спецификации.

    Размер здесь задают фильтры, а не поля ``width``/``height``: иначе
    ffmpeg получил бы два противоречивых указания на один и тот же кадр.
    """
    return video_args(replace(spec, width=None, height=None), filter_chain=filter_chain)


def _maps() -> list[str]:
    """Что попадает в вертикальный файл.

    Видеодорожку приходится называть явно. Стоит появиться хотя бы одному
    ``-map``, как автоматический выбор потоков выключается целиком — и файл
    собирается из одного звука, молча и с нулевым кодом возврата. Ровно
    так и случилось на первом прогоне: ffmpeg отработал, ffprobe в выходном
    файле видеопотока не нашёл.
    """
    return ["-map", "0:v:0", *_audio_passthrough()]


def _audio_passthrough() -> list[str]:
    """Звук берём как есть.

    Источник вертикали — наш собственный горизонтальный клип: звук в нём
    уже приведён к целевой громкости и кодеку. Перекодировать второй раз
    значит потерять качество на ровном месте и повторно применить
    нормализацию к уже нормализованному.
    """
    return ["-map", "0:a?", "-c:a", "copy"]


def _container(spec: RenderSpec) -> list[str]:
    return ["-movflags", "+faststart"] if spec.faststart else []


def _escape(path: Path) -> str:
    """Экранирование пути внутри аргумента фильтра.

    В графе фильтров двоеточие разделяет параметры, запятая — фильтры,
    и путь вида ``C:\\tmp\\a,b.cmd`` разваливает разбор. Обратный слеш
    перед каждым таким символом — то, чего ждёт парсер ffmpeg.
    """
    text = str(path)
    for symbol in ("\\", ":", ",", "'", "[", "]", ";"):
        text = text.replace(symbol, "\\" + symbol)
    return text


async def _source_fps(probe_binary: str, source: Path) -> float:
    payload = await probe_json(
        probe_binary,
        [
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            str(source),
        ],
    )
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        return MIN_COMMAND_FPS
    raw = streams[0].get("avg_frame_rate") if isinstance(streams[0], dict) else None
    if not isinstance(raw, str) or "/" not in raw:
        return MIN_COMMAND_FPS
    numerator, _, denominator = raw.partition("/")
    try:
        value = float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        return MIN_COMMAND_FPS
    return value if value > 0 else MIN_COMMAND_FPS
