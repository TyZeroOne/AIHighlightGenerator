"""Качество распознавания на живой речи.

Отдельный набор под маркером ``quality`` (док 09, §9.8): он не про
работоспособность, а про то, насколько хорошо система делает свою работу.
Такие тесты нельзя гонять в общем прогоне — им нужны большая модель, минуты
времени и материал, которого нет в репозитории.

Материал именно поэтому лежит снаружи: у записи есть правообладатель, весит
она мегабайты и в diff не читается. Путь задаётся переменными окружения,
эталонный текст — единственное, что хранится рядом с тестом.

Запуск::

    ASR_REFERENCE_MEDIA=/media/in/reference.mp4 \\
    WHISPER_MODEL=large-v3 DEVICE=cuda \\
    python -m pytest tests/quality -m quality -q
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from audio.extractor import FFmpegAudioExtractor
from core.domain.media import LocalMedia, MediaInfo
from core.domain.values import MediaWindow, TimeRange
from speech.whisper import FasterWhisperTranscriber
from tests.metrics import tokens, wer

pytestmark = [pytest.mark.quality, pytest.mark.slow, pytest.mark.asyncio]

#: Строки эталона имеют вид «00:00:38 — текст реплики».
_LINE = re.compile(r"^\d{2}:\d{2}:\d{2}\s+—\s+(?P<text>.+)$")

REFERENCE_TEXT = Path(__file__).parent.parent / "fixtures" / "asr" / "kabanych.txt"

#: Порог из DoD этапа 4 для русского языка.
MAX_WER_RU = 0.25


def load_reference(path: Path) -> str:
    """Собрать эталон из построчной расшифровки с тайм-кодами.

    Формат — то, что отдают сервисы расшифровки: заголовок, имя спикера,
    строки «время — текст». Разбор терпимый: всё, что не похоже на реплику,
    молча пропускается.
    """
    lines = [
        match["text"].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if (match := _LINE.match(line.strip()))
    ]
    return " ".join(lines)


@pytest.fixture(scope="module")
def media_path() -> Path:
    raw = os.environ.get("ASR_REFERENCE_MEDIA", "")
    if not raw:
        pytest.skip("ASR_REFERENCE_MEDIA не задан: нечего распознавать")
    path = Path(raw)
    if not path.exists():
        pytest.skip(f"Эталонный файл не найден: {path}")
    return path


@pytest.fixture(scope="module")
def reference() -> str:
    if not REFERENCE_TEXT.exists():
        pytest.skip(f"Эталонный текст не найден: {REFERENCE_TEXT}")
    text = load_reference(REFERENCE_TEXT)
    if not text:
        pytest.skip("Эталонный текст пуст — проверьте формат строк")
    return text


class TestRussianSpeech:
    async def test_wer_is_within_budget(
        self, media_path: Path, reference: str, tmp_path: Path
    ) -> None:
        """DoD этапа 4: WER < 0.25 на русской речи.

        Меряется весь путь целиком — извлечение аудио, VAD, распознавание,
        постобработка, — а не одна модель. Именно так система и работает,
        и деградация любого звена должна быть видна здесь.
        """
        model = os.environ.get("WHISPER_MODEL", "large-v3")
        extractor = FFmpegAudioExtractor(workdir=tmp_path)
        transcriber = FasterWhisperTranscriber(
            model_size=model,
            device=os.environ.get("DEVICE", "auto"),
            language=os.environ.get("WHISPER_LANGUAGE") or None,
        )

        media = LocalMedia(
            path=media_path,
            info=MediaInfo(duration_sec=_probe_duration(media_path), audio_codec="aac"),
            source_uri=f"file://{media_path}",
            owned=False,
        )
        track = await extractor.extract(
            media,
            MediaWindow(source_id=media.source_uri, range=TimeRange(0.0, media.info.duration_sec)),
        )
        result = await transcriber.transcribe(track)

        rate = wer(reference, result.text)
        print(
            f"\nмодель={model} устройство={transcriber.device}/{transcriber.compute_type}"
            f"\nязык={result.language} сегментов={len(result.segments)}"
            f" слов эталона={len(tokens(reference))} слов распознано={len(tokens(result.text))}"
            f"\nWER={rate:.3f}\n---\n{result.text}\n"
        )

        assert result.language == "ru"
        assert rate < MAX_WER_RU, f"WER={rate:.3f}, распознано: {result.text!r}"


def _probe_duration(path: Path) -> float:
    """Длительность через ffprobe.

    В тесте качества можно позволить себе лишний процесс: он один на весь
    прогон, а знать точную длительность нужно, чтобы окно покрыло файл целиком.
    """
    import json
    import subprocess

    raw = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return float(json.loads(raw.stdout)["format"]["duration"])
