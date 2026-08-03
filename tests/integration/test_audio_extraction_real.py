"""Извлечение аудио настоящим ffmpeg.

Видео генерируется на месте с известными свойствами: тон 440 Гц, стерео,
44.1 кГц. Значит, есть что проверять — не «файл получился», а что на выходе
ровно моно 16 кГц PCM нужной длительности, как того требует Whisper.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from audio.extractor import FFmpegAudioExtractor
from core.domain.errors import FFmpegError, TransientError
from core.domain.media import LocalMedia, MediaInfo
from core.domain.values import MediaWindow, TimeRange

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

pytestmark.append(
    pytest.mark.skipif(
        FFMPEG is None or FFPROBE is None,
        reason="ffmpeg/ffprobe не установлены (в Docker-образе они есть)",
    )
)

DURATION = 6.0


def _generate(path: Path, *, with_audio: bool = True) -> Path:
    args = [
        str(FFMPEG),
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=s=320x240:r=25:d={DURATION}",
    ]
    if with_audio:
        # Стерео 44.1 кГц — то есть заведомо не тот формат, который нужен
        # модели: проверяем, что приведение действительно происходит.
        args += [
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=44100:d={DURATION}",
            "-ac",
            "2",
        ]
    args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if with_audio:
        args += ["-c:a", "aac", "-shortest"]
    args += [str(path)]
    subprocess.run(args, check=True, capture_output=True, timeout=120)
    return path


def _probe(path: Path) -> dict[str, object]:
    raw = subprocess.run(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return json.loads(raw.stdout)  # type: ignore[no-any-return]


def _media(path: Path, *, with_audio: bool = True) -> LocalMedia:
    return LocalMedia(
        path=path,
        info=MediaInfo(
            duration_sec=DURATION,
            width=320,
            height=240,
            fps=25.0,
            video_codec="h264",
            audio_codec="aac" if with_audio else None,
            container="mp4",
        ),
        source_uri=f"file://{path}",
        owned=False,
    )


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _generate(tmp_path_factory.mktemp("audio") / "with_sound.mp4")


class TestExtraction:
    async def test_output_is_mono_16k_pcm(self, clip: Path, tmp_path: Path) -> None:
        """Формат не «на всякий случай»: именно на нём обучался Whisper,
        и подавать ему что-то другое — значит платить за ресемплинг дважды."""
        extractor = FFmpegAudioExtractor(binary=str(FFMPEG), workdir=tmp_path)
        media = _media(clip)

        track = await extractor.extract(
            media, MediaWindow(source_id=media.source_uri, range=TimeRange(0.0, DURATION))
        )

        probed = _probe(track.path)
        stream = probed["streams"][0]  # type: ignore[index]
        assert stream["codec_name"] == "pcm_s16le"
        assert int(stream["sample_rate"]) == 16000
        assert int(stream["channels"]) == 1
        assert track.sample_rate == 16000 and track.channels == 1

    async def test_duration_matches_the_source(self, clip: Path, tmp_path: Path) -> None:
        """Длительность считается из размера файла, без запуска ffprobe.
        Проверяем, что арифметика совпадает с реальностью."""
        extractor = FFmpegAudioExtractor(binary=str(FFMPEG), workdir=tmp_path)
        media = _media(clip)

        track = await extractor.extract(
            media, MediaWindow(source_id=media.source_uri, range=TimeRange(0.0, DURATION))
        )

        probed_duration = float(probed_format(_probe(track.path)))
        assert track.duration_sec == pytest.approx(probed_duration, abs=0.05)
        assert track.duration_sec == pytest.approx(DURATION, abs=0.2)

    async def test_window_limits_the_output(self, clip: Path, tmp_path: Path) -> None:
        """Окно — не декорация: для стрима этап получит кусок, а не весь файл."""
        extractor = FFmpegAudioExtractor(binary=str(FFMPEG), workdir=tmp_path)
        media = _media(clip)

        track = await extractor.extract(
            media, MediaWindow(source_id=media.source_uri, range=TimeRange(1.0, 3.0))
        )

        assert track.duration_sec == pytest.approx(2.0, abs=0.15)

    async def test_video_without_audio_is_rejected_clearly(self, tmp_path: Path) -> None:
        """Пустой WAV вместо отказа спрятал бы факт «звука нет» от конвейера."""
        path = _generate(tmp_path / "silent.mp4", with_audio=False)
        extractor = FFmpegAudioExtractor(binary=str(FFMPEG), workdir=tmp_path)

        with pytest.raises(FFmpegError, match="нет аудиодорожки"):
            await extractor.extract(
                _media(path, with_audio=False),
                MediaWindow(source_id="file://x", range=TimeRange(0.0, DURATION)),
            )

    async def test_missing_binary_is_a_transient_error(self, clip: Path, tmp_path: Path) -> None:
        """Сломанное окружение — не «битое видео»: разбор инцидента должен
        начинаться с правильного места."""
        extractor = FFmpegAudioExtractor(binary="definitely-not-ffmpeg", workdir=tmp_path)
        media = _media(clip)

        with pytest.raises(TransientError, match="ffmpeg не найден"):
            await extractor.extract(
                media, MediaWindow(source_id=media.source_uri, range=TimeRange(0.0, DURATION))
            )

    async def test_corrupted_source_fails_without_leaking_paths(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken.mp4"
        broken.write_bytes(b"definitely not a video")
        extractor = FFmpegAudioExtractor(binary=str(FFMPEG), workdir=tmp_path)

        with pytest.raises(FFmpegError) as exc_info:
            await extractor.extract(
                _media(broken),
                MediaWindow(source_id="file://x", range=TimeRange(0.0, DURATION)),
            )

        assert str(tmp_path) not in exc_info.value.message


def probed_format(probed: dict[str, object]) -> str:
    return str(probed["format"]["duration"])  # type: ignore[index]
