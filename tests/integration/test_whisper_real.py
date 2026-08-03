"""Настоящая модель на настоящем аудио.

Эталон синтезируется на месте (``espeak-ng``), поэтому текст известен точно,
а тест воспроизводим и не тянет внешних датасетов. Плата за это — речь
синтетическая: WER на ней измеряет **конвейер**, а не качество модели
на живой речи. Порог поэтому мягкий, а проверка качества на настоящей
записи остаётся ручной (см. разбор этапа).

Тесты помечены ``slow``: модель качается с HuggingFace при первом запуске
и живёт в томе ``models``.
"""

from __future__ import annotations

import itertools
import shutil
import subprocess
from pathlib import Path

import pytest
from tests.metrics import wer

from core.domain.media import AudioTrack
from speech.whisper import FasterWhisperTranscriber

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.asyncio]

FFMPEG = shutil.which("ffmpeg")
ESPEAK = shutil.which("espeak-ng") or shutil.which("espeak")

pytestmark.append(
    pytest.mark.skipif(FFMPEG is None, reason="ffmpeg не установлен"),
)

#: Текст подобран так, чтобы в нём не было ни имён собственных, ни цифр:
#: и то и другое модель пишет по-своему, и WER мерил бы орфографию, а не слух.
REFERENCE = (
    "the quick brown fox jumps over the lazy dog while the cat sleeps on the warm window sill"
)


def _synthesize(path: Path, text: str, *, voice: str = "en") -> Path:
    """Речь из текста. Голос синтетический, зато текст известен точно."""
    subprocess.run(
        [str(ESPEAK), "-v", voice, "-s", "150", "-w", str(path), text],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return _to_wav16k(path)


def _to_wav16k(path: Path) -> Path:
    target = path.with_name(f"{path.stem}_16k.wav")
    subprocess.run(
        [
            str(FFMPEG),
            "-y",
            "-v",
            "error",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(target),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return target


def _silence(path: Path, seconds: float = 8.0) -> Path:
    subprocess.run(
        [
            str(FFMPEG),
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r=16000:cl=mono:d={seconds}",
            "-c:a",
            "pcm_s16le",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return path


def _music(path: Path, seconds: float = 8.0) -> Path:
    """«Музыка» без речи: два тона и шум. На таком входе Whisper любит
    выдумывать текст — ровно то, что должен отсечь фильтр."""
    subprocess.run(
        [
            str(FFMPEG),
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=220:d={seconds}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=330:d={seconds}",
            "-filter_complex",
            "[0][1]amix=inputs=2",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return path


def track(path: Path) -> AudioTrack:
    size = path.stat().st_size
    return AudioTrack(
        path=path,
        sample_rate=16000,
        channels=1,
        duration_sec=max(size - 44, 0) / (16000 * 2),
    )


@pytest.fixture(scope="module")
def transcriber() -> FasterWhisperTranscriber:
    """Модель ``tiny``: 75 МБ и секунды на инициализацию.

    Большая модель здесь не нужна — проверяется конвейер, а не рекорд по WER.
    """
    # Параметры — ровно продовые, кроме размера модели. Beam size здесь
    # не мелочь: на beam_size=1 та же tiny даёт WER 0.61 вместо 0.11,
    # то есть тест с «ускоренными» настройками проверял бы не тот конвейер,
    # который работает в проде.
    instance = FasterWhisperTranscriber(model_size="tiny", chunk_sec=0.0)
    try:
        instance._load()
    except Exception as error:  # pragma: no cover — нет сети или сборки под GPU
        pytest.skip(f"Модель недоступна: {error}")
    return instance


@pytest.fixture(scope="module")
def speech_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if ESPEAK is None:
        pytest.skip("espeak-ng не установлен: нечем синтезировать эталон")
    return _synthesize(tmp_path_factory.mktemp("speech") / "reference.wav", REFERENCE)


class TestQuality:
    async def test_wer_on_synthetic_speech(
        self, transcriber: FasterWhisperTranscriber, speech_wav: Path
    ) -> None:
        """Замеренное значение на этой связке — 0.11 (модель ``tiny``,
        синтезированная речь). Порог 0.25 оставляет запас на дрожание
        декодера, но ловит настоящую поломку: пустой результат, тишину
        вместо речи, потерю половины слов."""
        result = await transcriber.transcribe(track(speech_wav), language="en")

        assert result.segments, "на файле с речью транскрипт не может быть пустым"
        rate = wer(REFERENCE, result.text)
        assert rate < 0.25, f"WER={rate:.2f}, распознано: {result.text!r}"

    async def test_language_is_detected(
        self, transcriber: FasterWhisperTranscriber, speech_wav: Path
    ) -> None:
        result = await transcriber.transcribe(track(speech_wav))

        assert result.language == "en"


class TestTimings:
    async def test_word_timings_are_present_and_monotonic(
        self, transcriber: FasterWhisperTranscriber, speech_wav: Path
    ) -> None:
        """DoD этапа. Без пословных таймингов не будет ни караоке-субтитров
        (Этап 9), ни снапа границ клипа к паузе (Этап 7)."""
        result = await transcriber.transcribe(track(speech_wav), language="en")

        all_words = [word for segment in result.segments for word in segment.words]
        assert all_words, "пословные тайминги обязательны"
        assert all(word.start <= word.end for word in all_words)
        for previous, current in itertools.pairwise(all_words):
            assert previous.end <= current.start + 0.01

    async def test_words_stay_inside_their_segment(
        self, transcriber: FasterWhisperTranscriber, speech_wav: Path
    ) -> None:
        result = await transcriber.transcribe(track(speech_wav), language="en")

        for segment in result.segments:
            for word in segment.words:
                assert segment.range.start - 0.01 <= word.start
                assert word.end <= segment.range.end + 0.01


class TestNonSpeech:
    async def test_silence_gives_empty_transcript(
        self, transcriber: FasterWhisperTranscriber, tmp_path: Path
    ) -> None:
        """Тишина — не ошибка. Пустой транскрипт здесь единственный
        честный ответ, и конвейер обязан пойти дальше."""
        wav = _silence(tmp_path / "silence.wav")

        result = await transcriber.transcribe(track(wav))

        assert result.is_empty

    async def test_music_without_speech_is_filtered(
        self, transcriber: FasterWhisperTranscriber, tmp_path: Path
    ) -> None:
        """DoD этапа: на музыке модель галлюцинирует, и мусор не должен
        доехать до скоринга."""
        wav = _music(tmp_path / "music.wav")

        result = await transcriber.transcribe(track(wav))

        assert len(result.segments) <= 1, f"мусор не отфильтрован: {result.text!r}"
