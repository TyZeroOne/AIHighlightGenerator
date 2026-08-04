"""Точная резка фрагмента из исходного видео.

## Где `-ss` — до `-i` или после

Разница между «быстро» и «точно» здесь давно не та, что в туториалах
десятилетней давности.

`-ss` **после** `-i` заставляет ffmpeg декодировать файл с нуля и выбросить
всё до нужной секунды. На отметке 1:23:40 двухчасового фильма это минуты
работы впустую — на каждый клип.

`-ss` **до** `-i` — это перемотка средствами демуксера: ffmpeg прыгает
к ближайшему ключевому кадру перед целью и начинает оттуда. Раньше это
означало неточность в секунды, но с версии 2.1 при перекодировании
он декодирует хвост от ключевого кадра до цели и выбрасывает лишние кадры
сам. Получается и быстро, и точно — поэтому здесь всегда `-ss` до `-i`.

Отдельная резка «хвостов» с последующим склеиванием (smart cut) не нужна
ровно потому, что мы всё равно перекодируем: приводим громкость к общему
уровню и качество к общему пресету. Экономить нечего.

## Почему копирование потока выключено по умолчанию

Копирование без перекодирования (`-c copy`) быстрее в разы, и роадмап
предполагал включать его, когда границы клипа совпали с ключевыми
кадрами. Замер показал, что этого условия мало.

Трейлер, начало клипа поставлено **точно** в ключевой кадр (18.750 с),
заказано 14.000 с:

| Режим | Факт | Отклонение | Время |
|---|---|---|---|
| `encode` | 14.000 с | 0.000 с | 0.6 с |
| `copy` | 14.123 с | **+0.123 с** | 0.1 с |

Причина в том, что копирование режет по границам **пакетов**, а не
кадров: последний GOP уезжает целиком, и длительность получается
такой, какой её сделал кодек исходника. Требование этапа — ±0.05 с,
и 0.123 в него не помещается.

Поэтому режим остаётся в коде, но включается только явно
(`RENDER_ALLOW_STREAM_COPY=true`) — например, когда важнее скорость,
а не точность до кадра. К тому же он всё равно несовместим
с нормализацией громкости и масштабированием, то есть с настройками
по умолчанию: :func:`can_stream_copy` отвергнет спецификацию раньше,
чем дело дойдёт до ключевых кадров.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from core.domain.errors import ValidationError
from core.domain.media import LocalMedia
from core.domain.render import RenderSpec
from core.domain.values import TimeRange
from core.logging import get_logger
from video_editor.encoder import audio_args, video_args
from video_editor.process import keyframe_times, probe_duration, run_ffmpeg

__all__ = ["CutResult", "FFmpegClipCutter", "can_stream_copy"]

log = get_logger(__name__)

#: Насколько близко начало клипа должно лежать к ключевому кадру, чтобы
#: копирование потока было безопасным. 40 мс — примерно один кадр при 25 fps.
KEYFRAME_TOLERANCE: Final = 0.04

#: Кодеки, при которых копирование вообще имеет смысл: результат должен
#: остаться тем, что играет везде без разговоров.
COPYABLE_VIDEO: Final = frozenset({"h264"})
COPYABLE_AUDIO: Final = frozenset({"aac"})

#: Запас времени на один клип. Полчаса — это в разы больше, чем занимает
#: сорокапятисекундный отрезок даже на медленном пресете; смысл лимита
#: в том, чтобы зависший ffmpeg не держал очередь до бесконечности.
CUT_TIMEOUT_SEC: Final = 1800.0


@dataclass(frozen=True, slots=True)
class CutResult:
    path: Path
    #: ``copy`` или ``encode`` — какой режим сработал. Нужно замеру
    #: и разбору: без этого «почему один клип готовился 3 секунды,
    #: а другой 40» остаётся загадкой.
    mode: str
    duration_sec: float | None = None


class FFmpegClipCutter:
    """Резка фрагмента. Часть реализации порта ``ClipRenderer``."""

    def __init__(
        self,
        *,
        binary: str = "ffmpeg",
        probe_binary: str = "ffprobe",
        allow_stream_copy: bool = False,
    ) -> None:
        self._binary = binary
        self._probe = probe_binary
        self._allow_copy = allow_stream_copy

    async def cut(
        self,
        media: LocalMedia,
        span: TimeRange,
        spec: RenderSpec,
        destination: Path,
    ) -> Path:
        result = await self.cut_detailed(media, span, spec, destination)
        return result.path

    async def cut_detailed(
        self,
        media: LocalMedia,
        span: TimeRange,
        spec: RenderSpec,
        destination: Path,
    ) -> CutResult:
        """Та же резка, но с отчётом о режиме и фактической длительности."""
        window = _clamp(span, media.info.duration_sec)
        destination.parent.mkdir(parents=True, exist_ok=True)

        mode = "encode"
        if self._allow_copy and can_stream_copy(media, spec):
            keyframes = await keyframe_times(
                self._probe, media.path, start=window.start, end=window.end
            )
            if _snapped(window.start, keyframes):
                mode = "copy"

        args = (
            _copy_args(media, window, destination)
            if mode == "copy"
            else _encode_args(media, window, spec, destination)
        )
        await run_ffmpeg(
            self._binary,
            args,
            destination=destination,
            action=f"нарезка клипа {window.start:.2f}–{window.end:.2f}",
            timeout_sec=CUT_TIMEOUT_SEC,
        )

        actual = await probe_duration(self._probe, destination)
        log.info(
            "clip.cut",
            mode=mode,
            requested=round(window.duration, 3),
            actual=round(actual, 3) if actual is not None else None,
        )
        return CutResult(path=destination, mode=mode, duration_sec=actual)


def can_stream_copy(media: LocalMedia, spec: RenderSpec) -> bool:
    """Совместимы ли требования к клипу с копированием потока.

    Проверка только по требованиям и кодекам; попадание в ключевой кадр
    выясняется отдельно, потому что стоит запуска ffprobe, а этот ответ
    бесплатен и в большинстве случаев уже отрицательный.
    """
    if spec.width or spec.height:
        return False
    if spec.burn_subtitles_path is not None:
        return False
    if spec.loudness_lufs is not None:
        return False
    if (media.info.video_codec or "") not in COPYABLE_VIDEO:
        return False
    # Видео без звука копируется спокойно: несовпадать нечему.
    return not media.info.has_audio or (media.info.audio_codec or "") in COPYABLE_AUDIO


def _snapped(start: float, keyframes: list[float]) -> bool:
    return any(abs(start - moment) <= KEYFRAME_TOLERANCE for moment in keyframes)


def _clamp(span: TimeRange, duration_sec: float) -> TimeRange:
    """Обрезать запрос по фактической длине файла.

    Отбор оперирует границами сцен, а те приходят из детектора и могут
    на доли секунды выходить за конец файла. ffmpeg на этом не падает,
    но отдаёт клип короче заказанного — и проверка длительности начинает
    ругаться на несуществующую проблему.
    """
    if duration_sec <= 0:
        return span
    end = min(span.end, duration_sec)
    if end <= span.start:
        raise ValidationError(
            "Клип начинается за пределами файла",
            start=round(span.start, 3),
            duration_sec=round(duration_sec, 3),
        )
    return TimeRange(span.start, end)


def _seek_args(media: LocalMedia, window: TimeRange) -> list[str]:
    return [
        "-ss",
        f"{window.start:.3f}",
        "-i",
        str(media.path),
        "-t",
        f"{window.duration:.3f}",
    ]


def _map_args(media: LocalMedia) -> list[str]:
    """Что берём из исходника.

    Знак вопроса в ``0:a:0?`` — не опечатка: он делает дорожку
    необязательной. Без него видео без звука роняет резку, хотя это
    штатный случай (док 04, §4.5).
    """
    args = ["-map", "0:v:0"]
    args += ["-map", "0:a:0?"] if media.info.has_audio else ["-an"]
    # Субтитры и данные исходника в клип не переносим: в mp4 они чаще
    # всего в несовместимом формате и роняют мукс на ровном месте.
    args += ["-sn", "-dn"]
    return args


def _encode_args(
    media: LocalMedia,
    window: TimeRange,
    spec: RenderSpec,
    destination: Path,
) -> list[str]:
    args = _seek_args(media, window)
    args += _map_args(media)
    args += video_args(spec)
    if media.info.has_audio:
        args += audio_args(spec)
    if spec.faststart:
        args += ["-movflags", "+faststart"]
    args += ["-y", str(destination)]
    return args


def _copy_args(media: LocalMedia, window: TimeRange, destination: Path) -> list[str]:
    args = _seek_args(media, window)
    args += _map_args(media)
    args += [
        "-c",
        "copy",
        # Перемотка оставляет отрицательные метки времени на первых кадрах;
        # без сдвига к нулю плеер показывает их как рассинхрон звука.
        "-avoid_negative_ts",
        "make_zero",
        "-movflags",
        "+faststart",
        "-y",
        str(destination),
    ]
    return args
