"""Настоящие детекторы на настоящем видео.

Видео генерируется FFmpeg прямо здесь, из одиннадцати кусков по две секунды
с резко разным содержимым. Значит, склеек ровно десять, и мы знаем их
с точностью до кадра — есть с чем сравнивать результат, а не «на глаз
похоже».

Ключевая проверка этапа — **взаимозаменяемость**: оба детектора проходят один
и тот же набор тестов через порт ``SceneDetector``. Если бы у каждого был свой
набор, «взаимозаменяемость» существовала бы только на словах.

Тесты пропускаются без ffmpeg: в Docker он есть всегда, на машине
разработчика — не обязательно.
"""

from __future__ import annotations

import itertools
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from core.domain.media import LocalMedia, MediaInfo
from core.domain.values import MediaWindow, TimeRange
from core.ports.media import SceneDetector
from video_loader.scenes.ffmpeg import FFmpegSceneDetector
from video_loader.scenes.keyframes import FFmpegKeyframeExtractor

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

FFMPEG = shutil.which("ffmpeg")

pytestmark.append(
    pytest.mark.skipif(FFMPEG is None, reason="ffmpeg не установлен (в Docker-образе он есть)")
)

#: Одиннадцать кусков по 2 с → склейки на 2, 4, 6, … 20.
SEGMENT_SEC = 2.0
SEGMENTS = 11
EXPECTED_CUTS = [SEGMENT_SEC * index for index in range(1, SEGMENTS)]
TOTAL_SEC = SEGMENT_SEC * SEGMENTS

#: Источники подобраны так, чтобы соседние отличались радикально: сплошной
#: цвет → тестовая таблица → шум. Мягкие переходы — отдельная история,
#: и проверять на них порог детектора здесь нечего.
_SOURCES = [
    "color=c=red:s=320x240:r=25",
    "testsrc2=s=320x240:r=25",
    "color=c=blue:s=320x240:r=25",
    "smptebars=s=320x240:r=25",
    "color=c=green:s=320x240:r=25",
    "testsrc=s=320x240:r=25",
    "color=c=yellow:s=320x240:r=25",
    "smptehdbars=s=320x240:r=25",
    "color=c=black:s=320x240:r=25",
    "color=c=white:s=320x240:r=25",
    "testsrc2=s=320x240:r=25",
]


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Видео с десятью склейками в заранее известных местах."""
    directory = tmp_path_factory.mktemp("scenes")
    path = directory / "cuts.mp4"

    args: list[str] = [str(FFMPEG), "-y", "-v", "error"]
    for source in _SOURCES:
        args += ["-f", "lavfi", "-i", f"{source}:d={SEGMENT_SEC}"]

    streams = "".join(f"[{index}:v]" for index in range(SEGMENTS))
    args += [
        "-filter_complex",
        f"{streams}concat=n={SEGMENTS}:v=1:a=0[out]",
        "-map",
        "[out]",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        # Ключевой кадр на каждую склейку: так границы сцен ложатся ровно
        # на GOP, и результат не зависит от того, куда попал ближайший I-кадр.
        "-g",
        "50",
        str(path),
    ]
    subprocess.run(args, check=True, capture_output=True, timeout=180)
    return path


@pytest.fixture(scope="module")
def media(clip: Path) -> LocalMedia:
    return LocalMedia(
        path=clip,
        info=MediaInfo(duration_sec=TOTAL_SEC, width=320, height=240, fps=25.0, container="mp4"),
        source_uri=f"file://{clip}",
        owned=False,
    )


def make_detector(name: str) -> SceneDetector:
    if name == "ffmpeg":
        return FFmpegSceneDetector(binary=str(FFMPEG), min_len_sec=1.0, max_len_sec=90.0)

    from video_loader.scenes.pyscenedetect import PySceneDetectDetector

    return PySceneDetectDetector(min_len_sec=1.0, max_len_sec=90.0)


def cuts_of(scenes: Sequence[TimeRange]) -> list[float]:
    """Границы между сценами — то же, что найденные склейки."""
    return [scene.start for scene in scenes[1:]]


def matched(found: Sequence[float], expected: Sequence[float], tolerance: float = 0.5) -> int:
    return sum(1 for moment in expected if any(abs(moment - other) <= tolerance for other in found))


@pytest.fixture(params=["pyscenedetect", "ffmpeg"])
def detector(request: pytest.FixtureRequest) -> SceneDetector:
    """Один и тот же набор тестов для обеих реализаций порта."""
    return make_detector(request.param)


class TestBothDetectors:
    async def test_finds_all_known_cuts(self, detector: SceneDetector, media: LocalMedia) -> None:
        """DoD этапа: найдены все 10 склеек, ложных не больше одной."""
        window = MediaWindow(source_id=media.source_uri, range=TimeRange(0.0, TOTAL_SEC))

        scenes = await detector.detect(media, window)

        found = cuts_of(scenes)
        assert matched(found, EXPECTED_CUTS) == len(EXPECTED_CUTS), (
            f"пропущены склейки: найдено {found}, ожидалось {EXPECTED_CUTS}"
        )
        assert len(found) - len(EXPECTED_CUTS) <= 1, f"слишком много ложных: {found}"

    async def test_scenes_cover_the_window(
        self, detector: SceneDetector, media: LocalMedia
    ) -> None:
        window = MediaWindow(source_id=media.source_uri, range=TimeRange(0.0, TOTAL_SEC))

        scenes = await detector.detect(media, window)

        assert scenes[0].start == pytest.approx(0.0, abs=1e-6)
        assert scenes[-1].end == pytest.approx(TOTAL_SEC, abs=1e-6)
        for previous, current in itertools.pairwise(scenes):
            assert current.start == pytest.approx(previous.end, abs=1e-6)

    async def test_partial_window(self, detector: SceneDetector, media: LocalMedia) -> None:
        """Окно с середины файла — так конвейер пойдёт по стриму (док 02, §2.6)."""
        window = MediaWindow(source_id=media.source_uri, range=TimeRange(6.0, 14.0), is_final=False)

        scenes = await detector.detect(media, window)

        assert scenes[0].start == pytest.approx(6.0, abs=1e-6)
        assert scenes[-1].end == pytest.approx(14.0, abs=1e-6)
        # Внутри окна три склейки: 8, 10, 12.
        assert matched(cuts_of(scenes), [8.0, 10.0, 12.0]) == 3

    async def test_min_length_is_respected(self, media: LocalMedia) -> None:
        """При минимуме в 5 с двухсекундные куски обязаны слиться."""
        window = MediaWindow(source_id=media.source_uri, range=TimeRange(0.0, TOTAL_SEC))
        detector = FFmpegSceneDetector(binary=str(FFMPEG), min_len_sec=5.0, max_len_sec=90.0)

        scenes = await detector.detect(media, window)

        assert all(scene.duration >= 5.0 - 1e-6 for scene in scenes)

    async def test_max_length_forces_split(self, media: LocalMedia) -> None:
        """Статичный длинный план обязан дробиться, иначе скоринг усреднит
        всё подряд и утопит любой всплеск."""
        window = MediaWindow(source_id=media.source_uri, range=TimeRange(0.0, TOTAL_SEC))
        detector = FFmpegSceneDetector(binary=str(FFMPEG), min_len_sec=1.0, max_len_sec=3.0)

        scenes = await detector.detect(media, window)

        assert all(scene.duration <= 3.0 + 1e-6 for scene in scenes)


class TestKeyframes:
    async def test_frames_are_extracted_for_every_scene(
        self, media: LocalMedia, tmp_path: Path
    ) -> None:
        extractor = FFmpegKeyframeExtractor(binary=str(FFMPEG), width=320, max_parallel=4)
        moments = [1.0, 5.0, 9.0, 13.0]

        frames = await extractor.extract(media, moments, tmp_path / "frames")

        assert len(frames) == len(moments)
        assert all(frame is not None for frame in frames)
        for frame in frames:
            assert frame is not None
            assert frame.stat().st_size > 0
            assert frame.read_bytes()[:2] == b"\xff\xd8", "должен быть JPEG"

    async def test_frame_beyond_duration_returns_none(
        self, media: LocalMedia, tmp_path: Path
    ) -> None:
        """Кадра за концом файла не существует. Это не ошибка этапа —
        превью вспомогательный артефакт, и None здесь честнее исключения."""
        extractor = FFmpegKeyframeExtractor(binary=str(FFMPEG), width=320)

        frames = await extractor.extract(media, [TOTAL_SEC + 30.0], tmp_path / "frames")

        assert frames == [None]

    async def test_empty_input_does_not_spawn_processes(
        self, media: LocalMedia, tmp_path: Path
    ) -> None:
        extractor = FFmpegKeyframeExtractor(binary=str(FFMPEG))

        assert await extractor.extract(media, [], tmp_path / "frames") == []
