"""Извлечение аудиодорожки в формате, который ждёт распознавание речи.

Whisper обучался на моно 16 кГц. Подавать ему 48 кГц стерео бессмысленно:
он всё равно сведёт и ресемплит вниз — только сделает это уже внутри,
потратив память на дорожку вчетверо большего размера. Приводим сразу.

Формат — несжатый WAV (PCM 16 бит). Тому есть две причины. Первая: декодировать
его не нужно вообще, а на часе аудио это заметная экономия. Вторая: этот же
файл на Этапе 5 будут читать анализаторы громкости и пауз, и им нужны сырые
отсчёты, а не результат чужого сжатия с потерями.

Час речи в таком виде занимает 115 МБ. Это много для хранения, поэтому файл
живёт с ограниченным TTL (``RETENTION_AUDIO_H``) и удаляется после анализа.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from core.domain.errors import FFmpegError, TransientError
from core.domain.media import AudioTrack, LocalMedia
from core.domain.values import MediaWindow
from core.infra.proc import death_signal
from core.logging import get_logger

__all__ = ["FFmpegAudioExtractor"]

log = get_logger(__name__)


class FFmpegAudioExtractor:
    """Реализация порта ``AudioExtractor``."""

    def __init__(
        self,
        *,
        binary: str = "ffmpeg",
        workdir: Path,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        self._binary = binary
        self._workdir = workdir
        self._sample_rate = sample_rate
        self._channels = channels

    async def extract(self, media: LocalMedia, window: MediaWindow) -> AudioTrack:
        if not media.info.has_audio:
            # Видео без звука — штатная ситуация (док 04, §4.5), но извлекать
            # тут нечего, и создавать пустой WAV, чтобы конвейер «не заметил»,
            # значит спрятать факт. Пусть решает вызывающий сценарий.
            raise FFmpegError(
                "В медиафайле нет аудиодорожки",
                source_uri=media.source_uri,
            )

        destination = self._workdir / "audio" / f"{_stem(media)}.wav"
        destination.parent.mkdir(parents=True, exist_ok=True)

        args = [
            self._binary,
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-ss",
            f"{window.range.start:.3f}",
            "-t",
            f"{window.range.duration:.3f}",
            "-i",
            str(media.path),
            "-vn",  # видео не декодируем: на двухчасовом фильме это часы работы впустую
            "-sn",
            "-dn",
            "-map",
            "0:a:0",  # первая аудиодорожка; выбор языка дорожки — задел на будущее
            "-ac",
            str(self._channels),
            "-ar",
            str(self._sample_rate),
            "-c:a",
            "pcm_s16le",
            "-y",
            str(destination),
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
            raise TransientError(
                f"ffmpeg не найден: {self._binary}. Проверьте FFMPEG_BIN и образ.",
                binary=self._binary,
            ) from error

        _, stderr = await process.communicate()
        if process.returncode != 0 or not destination.exists():
            destination.unlink(missing_ok=True)
            raise FFmpegError(
                f"Извлечение аудио не удалось (код {process.returncode}): "
                f"{_first_line(stderr, str(media.path)) or 'без диагностики'}",
                returncode=process.returncode,
            )

        size = destination.stat().st_size
        if size == 0:
            destination.unlink(missing_ok=True)
            raise FFmpegError("Извлечённое аудио пустое", source_uri=media.source_uri)

        track = AudioTrack(
            path=destination,
            sample_rate=self._sample_rate,
            channels=self._channels,
            # Длительность считаем из размера файла, а не спрашиваем ffprobe:
            # у несжатого PCM она вычисляется точно, а лишний запуск процесса
            # на каждый файл — это лишние 50 мс и ещё одна точка отказа.
            duration_sec=_pcm_duration(size, self._sample_rate, self._channels),
        )
        log.info(
            "audio.extracted",
            duration_sec=round(track.duration_sec, 2),
            size_bytes=size,
            sample_rate=self._sample_rate,
        )
        return track


def _pcm_duration(size_bytes: int, sample_rate: int, channels: int) -> float:
    """Длительность PCM 16 бит по размеру файла.

    44 байта — заголовок WAV. Два байта на отсчёт: 16 бит.
    """
    payload = max(size_bytes - 44, 0)
    bytes_per_second = sample_rate * channels * 2
    return payload / bytes_per_second if bytes_per_second else 0.0


def _stem(media: LocalMedia) -> str:
    """Имя временного файла, стабильное для одного источника.

    Стабильное намеренно: повторный запуск этапа перезапишет тот же файл,
    а не создаст второй — то же правило, что и у ключей в хранилище.
    """
    return media.path.stem[:64] or "audio"


def _first_line(stderr: bytes, path: str) -> str:
    text = stderr.decode("utf-8", errors="replace").replace(path, "<файл>")
    return next((line.strip() for line in text.splitlines() if line.strip()), "")[:200]
