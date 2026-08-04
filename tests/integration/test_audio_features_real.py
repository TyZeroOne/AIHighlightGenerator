"""Аудиопризнаки на настоящем аудио с известной истиной.

Файл собирается на месте так, что ответ известен заранее: всплеск громкости
ровно на 12-й секунде, тишина ровно с 8-й по 10-ю. Это тот самый случай
из DoD этапа — проверяется не «функция что-то вернула», а «нашла именно то,
что мы туда положили».
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from uuid import uuid4

import pytest

from audio.loudness import LoudnessAnalyzer
from audio.pauses import PauseAnalyzer
from core.domain.analysis import AnalysisContext
from core.domain.media import AudioTrack
from core.domain.values import FeatureValue, MediaWindow, TimeRange

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

FFMPEG = shutil.which("ffmpeg")

pytestmark.append(pytest.mark.skipif(FFMPEG is None, reason="ffmpeg не установлен"))

DURATION = 20.0


def _build_audio(path: Path) -> Path:
    """Двадцать секунд с заранее известной раскладкой.

    * 0–8 c   — ровный тон средней громкости;
    * 8–10 c  — тишина;
    * 10–20 c — тон, а на 12-й секунде всплеск на 20 дБ на одну секунду.

    Собирается из четырёх кусков конкатенацией: так границы получаются
    точными до отсчёта, а не «примерно там».
    """
    parts = [
        ("sine=frequency=440:sample_rate=16000:duration=8", "-20dB"),
        ("anullsrc=r=16000:cl=mono:duration=2", None),
        ("sine=frequency=440:sample_rate=16000:duration=2", "-20dB"),
        ("sine=frequency=440:sample_rate=16000:duration=1", "0dB"),
        ("sine=frequency=440:sample_rate=16000:duration=7", "-20dB"),
    ]

    inputs: list[str] = []
    filters: list[str] = []
    for index, (source, gain) in enumerate(parts):
        inputs += ["-f", "lavfi", "-i", source]
        chain = f"[{index}:a]aformat=sample_fmts=s16:channel_layouts=mono"
        if gain:
            chain += f",volume={gain}"
        filters.append(f"{chain}[a{index}]")

    concat = "".join(f"[a{index}]" for index in range(len(parts)))
    graph = ";".join(filters) + f";{concat}concat=n={len(parts)}:v=0:a=1[out]"

    subprocess.run(
        [
            str(FFMPEG),
            "-y",
            "-v",
            "error",
            *inputs,
            "-filter_complex",
            graph,
            "-map",
            "[out]",
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


@pytest.fixture(scope="module")
def track(tmp_path_factory: pytest.TempPathFactory) -> AudioTrack:
    path = _build_audio(tmp_path_factory.mktemp("audio") / "known.wav")
    size = path.stat().st_size
    return AudioTrack(
        path=path,
        sample_rate=16000,
        channels=1,
        duration_sec=max(size - 44, 0) / (16000 * 2),
    )


@pytest.fixture
def context(track: AudioTrack, tmp_path: Path) -> AnalysisContext:
    return AnalysisContext(
        job_id=uuid4(),
        video_id=uuid4(),
        window=MediaWindow(source_id="file://known.wav", range=TimeRange(0.0, DURATION)),
        workdir=tmp_path,
        audio=track,
    )


#: Сцены нарезаны так, чтобы каждая содержала ровно один интересный факт.
SCENES = [
    TimeRange(0.0, 8.0),  # ровный тон
    TimeRange(8.0, 10.0),  # тишина
    TimeRange(10.0, 12.0),  # снова тон
    TimeRange(12.0, 13.0),  # всплеск
    TimeRange(13.0, 20.0),  # ровный тон
]

LOUD_SCENE = 3


def values(features: Mapping[int, Sequence[FeatureValue]], index: int) -> dict[str, float]:
    """Признаки сцены. Отсутствие сцены в результате — штатный случай:
    анализатор вправе не выдать признаки там, где мерить нечего."""
    return {value.key: value.value for value in features.get(index, ())}


class TestLoudness:
    async def test_burst_is_found_exactly_where_it_was_put(self, context: AnalysisContext) -> None:
        """DoD этапа: всплеск громкости ровно на 12.0 с — найден."""
        features = dict(await LoudnessAnalyzer(binary=str(FFMPEG)).analyze(context, SCENES))

        deltas = {
            index: values(features, index)["audio.loudness.delta"]
            for index in features
            if "audio.loudness.delta" in values(features, index)
        }
        assert deltas, "признак delta не посчитан ни для одной сцены"
        assert max(deltas, key=lambda index: deltas[index]) == LOUD_SCENE
        assert deltas[LOUD_SCENE] > 10.0, f"всплеск слишком слабый: {deltas[LOUD_SCENE]:.1f} дБ"

    async def test_loudness_difference_matches_the_generated_gain(
        self, context: AnalysisContext
    ) -> None:
        """Всплеск записан ровно на 20 дБ громче фона — столько и должно
        получиться в разнице.

        Проверяется именно разница, а не абсолютные LUFS: те зависят
        от частотной коррекции (синус 440 Гц ей не безразличен) и от того,
        какую амплитуду генератор ffmpeg считает единичной. Разница же
        от этого свободна — и она же и нужна скорингу (док 05, §5.3).
        """
        features = dict(await LoudnessAnalyzer(binary=str(FFMPEG)).analyze(context, SCENES))

        quiet = values(features, 0)["audio.loudness.mean"]
        loud = values(features, LOUD_SCENE)["audio.loudness.mean"]
        assert loud - quiet == pytest.approx(20.0, abs=3.0)

    async def test_silent_scene_has_no_loudness(self, context: AnalysisContext) -> None:
        """Сцена целиком в тишине: признака нет вовсе. Ноль LUFS означал бы
        «оглушительно», а −70 — выдуманное число."""
        features = dict(await LoudnessAnalyzer(binary=str(FFMPEG)).analyze(context, SCENES))

        assert "audio.loudness.mean" not in values(features, 1)

    async def test_peak_is_higher_on_the_burst(self, context: AnalysisContext) -> None:
        features = dict(await LoudnessAnalyzer(binary=str(FFMPEG)).analyze(context, SCENES))

        assert (
            values(features, LOUD_SCENE)["audio.loudness.peak"]
            > values(features, 0)["audio.loudness.peak"] + 10.0
        )


class TestPauses:
    async def test_silence_is_found_where_it_was_put(self, context: AnalysisContext) -> None:
        """DoD этапа: тишина 8–10 с — найдена.

        VAD ищет **речь**, а синусоидальный тон речью не является: паузами
        окажется всё, кроме, возможно, случайных срабатываний. Проверяем
        главное — что участок 8–10 с целиком помечен тишиной.
        """
        features = dict(await PauseAnalyzer().analyze(context, SCENES))

        silent = values(features, 1)
        assert silent["audio.pause.ratio"] == pytest.approx(1.0, abs=0.01)
        assert silent["audio.pause.max"] == pytest.approx(2.0, abs=0.05)

    async def test_every_scene_gets_pause_features(self, context: AnalysisContext) -> None:
        """Паузы считаются всегда: в отличие от громкости, «тишины не было»
        и «данных нет» здесь различаются нулём, а не отсутствием признака."""
        features = dict(await PauseAnalyzer().analyze(context, SCENES))

        assert set(features) == set(range(len(SCENES)))
        for index in features:
            assert set(values(features, index)) == {
                "audio.pause.total",
                "audio.pause.max",
                "audio.pause.ratio",
            }

    async def test_ratio_is_bounded(self, context: AnalysisContext) -> None:
        features = dict(await PauseAnalyzer().analyze(context, SCENES))

        for index in features:
            assert 0.0 <= values(features, index)["audio.pause.ratio"] <= 1.0
