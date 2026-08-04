"""Видеоанализаторы на настоящем файле с заранее известным содержанием.

Видео собирается из четырёх кусков по три секунды, и про каждый известно,
что в нём происходит:

* 0–3 c   — неподвижная тестовая таблица: движения нет;
* 3–6 c   — та же таблица, но камера панорамирует: движение есть, склеек нет;
* 6–9 c   — быстрый монтаж, картинка меняется каждые полсекунды: склейки;
* 9–12 c  — чёрный экран со вспышкой на 10-й секунде: скачок яркости.

Проверяется не «функция вернула число», а «нашла именно то, что положили».
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from uuid import uuid4

import pytest

from core.domain.analysis import AnalysisContext
from core.domain.media import LocalMedia, MediaInfo
from core.domain.values import FeatureValue, MediaWindow, TimeRange
from vision.cuts import CutsAnalyzer
from vision.dynamics import DynamicsAnalyzer
from vision.frames import FrameSource, reset_sources
from vision.motion import MotionAnalyzer

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

FFMPEG = shutil.which("ffmpeg")

pytestmark.append(pytest.mark.skipif(FFMPEG is None, reason="ffmpeg не установлен"))

STATIC = TimeRange(0.0, 3.0)
PANNING = TimeRange(3.0, 6.0)
MONTAGE = TimeRange(6.0, 9.0)
FLASH = TimeRange(9.0, 12.0)
SCENES = [STATIC, PANNING, MONTAGE, FLASH]


def _build_video(path: Path) -> Path:
    """Куски с известным поведением, склеенные в один файл.

    Монтажная секция набрана чередованием двух **разных, но одинаково
    ярких** таблиц. Мигание чёрное/белое было бы проще, но тогда монтаж
    давал бы и самый большой скачок яркости — и проверка вспышки потеряла
    бы смысл: каждая сцена должна отличаться ровно одним свойством.
    """
    parts = [
        # 0–3 c: неподвижная таблица, никакого движения.
        "smptebars=s=320x240:r=25:d=3",
        # 3–6 c: она же, но кадр плывёт вбок.
        "smptehdbars=s=640x480:r=25:d=3,crop=320:240:'min(t*60,320)':120",
        # 6–9 c: шесть планов по полсекунды, яркость сопоставима.
        *(f"{'smptebars' if index % 2 else 'testsrc2'}=s=320x240:r=25:d=0.5" for index in range(6)),
        # 9–12 c: чёрный экран со вспышкой на десятой секунде.
        "color=c=black:s=320x240:r=25:d=1",
        "color=c=white:s=320x240:r=25:d=0.3",
        "color=c=black:s=320x240:r=25:d=1.7",
    ]

    inputs: list[str] = []
    labels: list[str] = []
    for index, source in enumerate(parts):
        inputs += ["-f", "lavfi", "-i", source]
        labels.append(f"[{index}:v]scale=320:240,setsar=1[v{index}]")

    graph = ";".join(labels) + ";" + "".join(f"[v{i}]" for i in range(len(parts)))
    graph += f"concat=n={len(parts)}:v=1:a=0[out]"

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
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "25",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=180,
    )
    return path


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_video(tmp_path_factory.mktemp("vision") / "known.mp4")


@pytest.fixture
def context(clip: Path, tmp_path: Path) -> AnalysisContext:
    reset_sources()
    media = LocalMedia(
        path=clip,
        info=MediaInfo(duration_sec=12.0, width=320, height=240, fps=25.0, video_codec="h264"),
        source_uri=f"file://{clip}",
        owned=False,
    )
    return AnalysisContext(
        job_id=uuid4(),
        video_id=uuid4(),
        window=MediaWindow(source_id=media.source_uri, range=TimeRange(0.0, 12.0)),
        workdir=tmp_path,
        media=media,
    )


@pytest.fixture
def source(clip: Path) -> FrameSource:
    return FrameSource(clip, sample_fps=5.0, max_side=160)


def values(features: Mapping[int, Sequence[FeatureValue]], index: int) -> dict[str, float]:
    return {value.key: value.value for value in features.get(index, ())}


class TestFrameSource:
    async def test_decodes_only_the_requested_scene(self, source: FrameSource) -> None:
        batch = await source.scene_frames(STATIC)

        assert not batch.is_empty
        assert all(STATIC.start <= at < STATIC.end + 0.1 for at in batch.times)

    async def test_sampling_reduces_the_number_of_frames(self, source: FrameSource) -> None:
        """Три секунды при 25 fps — это 75 кадров; при пяти сэмплах в секунду
        должно остаться около пятнадцати."""
        batch = await source.scene_frames(STATIC)

        assert 10 <= len(batch) <= 20

    async def test_frames_are_downscaled_to_grayscale(self, source: FrameSource) -> None:
        batch = await source.scene_frames(STATIC)

        assert batch.frames[0].ndim == 2, "кадр должен быть одноканальным"
        assert max(batch.frames[0].shape) <= 160

    async def test_second_request_is_cached(self, source: FrameSource) -> None:
        """Три анализатора просят одни и те же кадры. Декодировать файл
        трижды — значит утроить самую дорогую часть этапа."""
        first = await source.scene_frames(PANNING)
        second = await source.scene_frames(PANNING)

        assert first is second


class TestMotion:
    async def test_panning_has_more_motion_than_a_static_shot(
        self, context: AnalysisContext, source: FrameSource
    ) -> None:
        features = await MotionAnalyzer(source=source).analyze(context, SCENES)

        static = values(features, 0)["vision.motion.mean"]
        panning = values(features, 1)["vision.motion.mean"]
        assert panning > static * 3, f"панорама {panning}, статика {static}"

    async def test_static_shot_is_nearly_still(
        self, context: AnalysisContext, source: FrameSource
    ) -> None:
        """Ноль недостижим и не нужен: кодек ultrafast оставляет на статичной
        таблице шум сжатия, и поток его видит. Замер даёт 0.0016 — порог
        поставлен с запасом, но на порядок ниже панорамы (0.013)."""
        features = await MotionAnalyzer(source=source).analyze(context, SCENES)

        assert values(features, 0)["vision.motion.mean"] < 0.005


class TestCuts:
    async def test_montage_is_detected(self, context: AnalysisContext, source: FrameSource) -> None:
        """Мигание дважды в секунду на трёхсекундной сцене — несколько
        склеек; на неподвижной таблице их нет вовсе."""
        features = await CutsAnalyzer(source=source).analyze(context, SCENES)

        assert values(features, 2)["vision.cuts.count"] >= 3
        assert values(features, 0)["vision.cuts.count"] == 0

    async def test_rate_is_per_second(self, context: AnalysisContext, source: FrameSource) -> None:
        features = await CutsAnalyzer(source=source).analyze(context, SCENES)

        montage = values(features, 2)
        assert montage["vision.cuts.rate"] == pytest.approx(
            montage["vision.cuts.count"] / MONTAGE.duration, abs=0.01
        )

    async def test_panning_is_not_mistaken_for_montage(
        self, context: AnalysisContext, source: FrameSource
    ) -> None:
        """Ключевое различие: камера едет, но план тот же.

        Полного нуля здесь не выходит и на гистограммах: панорама по цветным
        полосам меняет их пропорции в кадре, и на одном шаге это выглядит
        как смена содержания. Важно, что разница остаётся качественной —
        одна ложная склейка против пяти настоящих.
        """
        features = await CutsAnalyzer(source=source).analyze(context, SCENES)

        panning = values(features, 1)["vision.cuts.count"]
        montage = values(features, 2)["vision.cuts.count"]
        assert panning <= 1
        assert montage >= panning * 3


class TestDynamics:
    async def test_flash_gives_the_largest_brightness_jump(
        self, context: AnalysisContext, source: FrameSource
    ) -> None:
        """Вспышка на 10-й секунде — самое резкое изменение яркости в файле."""
        features = await DynamicsAnalyzer(source=source).analyze(context, SCENES)

        jumps = {
            index: values(features, index)["vision.dynamics.brightness_delta"]
            for index in range(len(SCENES))
            if "vision.dynamics.brightness_delta" in values(features, index)
        }
        assert max(jumps, key=lambda index: jumps[index]) == 3

    async def test_entropy_separates_rich_and_empty_frames(
        self, context: AnalysisContext, source: FrameSource
    ) -> None:
        """Цветная таблица насыщеннее чёрного экрана — и это должно быть видно."""
        features = await DynamicsAnalyzer(source=source).analyze(context, SCENES)

        assert (
            values(features, 0)["vision.dynamics.entropy"]
            > values(features, 3)["vision.dynamics.entropy"]
        )


class TestSharedSource:
    async def test_all_three_analyzers_reuse_one_decode(
        self, context: AnalysisContext, source: FrameSource
    ) -> None:
        """Проверка главного архитектурного решения группы: файл читается
        один раз на сцену, сколько бы анализаторов его ни просило."""
        decoded: list[TimeRange] = []
        original = source._decode

        def counting(scene: TimeRange) -> object:
            decoded.append(scene)
            return original(scene)

        source._decode = counting  # type: ignore[method-assign]

        for analyzer in (
            MotionAnalyzer(source=source),
            CutsAnalyzer(source=source),
            DynamicsAnalyzer(source=source),
        ):
            await analyzer.analyze(context, SCENES)

        assert len(decoded) == len(SCENES), f"файл декодирован {len(decoded)} раз вместо 4"
