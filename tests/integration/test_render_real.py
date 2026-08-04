"""Нарезка настоящим ffmpeg — проверки, которые нельзя сделать на фейках.

Юнит-тесты проверяют, что в команду попали нужные аргументы. Здесь
проверяется, что получившийся файл действительно такой, каким его обещали:
той длительности, того размера кадра, с картинкой в первом кадре и без
рассинхрона. Ни одно из этих свойств из аргументов не следует —
их устанавливает ffmpeg, и узнать их можно только запустив его.

Материал генерируется на месте с известными свойствами. Это дороже, чем
взять готовый файл, но зато ответ на вопрос «а сколько там должно быть»
известен точно, а не «примерно как в исходнике».
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from core.domain.media import LocalMedia, MediaInfo
from core.domain.render import VERTICAL_HEIGHT, VERTICAL_WIDTH, CropKeyframe, RenderSpec
from core.domain.values import CropBox, TimeRange
from video_editor.cutter import FFmpegClipCutter
from video_editor.renderer import FFmpegClipRenderer
from video_editor.vertical import MODE_BLURPAD, MODE_STATIC, MODE_TRACK, FFmpegVerticalRenderer

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

pytestmark.append(
    pytest.mark.skipif(
        FFMPEG is None or FFPROBE is None,
        reason="ffmpeg/ffprobe не установлены (в Docker-образе они есть)",
    )
)

SOURCE_DURATION = 20.0
FPS = 25

#: Требование DoD: длительность клипа совпадает с заданной с этой точностью.
DURATION_TOLERANCE = 0.05


def _probe(path: Path, entries: str, select: str | None = None) -> dict[str, object]:
    args = [str(FFPROBE), "-v", "error", "-of", "json", "-show_entries", entries]
    if select:
        args += ["-select_streams", select]
    args.append(str(path))
    return json.loads(subprocess.run(args, capture_output=True, check=True).stdout)


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> LocalMedia:
    """Исходник 640×360 с движущейся картинкой и синусоидой в звуке.

    Горизонтальный намеренно: вертикальное кадрирование имеет смысл
    проверять только на материале, который в 9:16 не помещается.
    """
    path = tmp_path_factory.mktemp("render") / "source.mp4"
    subprocess.run(
        [
            str(FFMPEG),
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=s=640x360:r={FPS}:d={SOURCE_DURATION}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=48000:d={SOURCE_DURATION}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            # Ключевой кадр раз в две секунды: копирование потока должно
            # иметь шанс сработать хоть где-то.
            "-g",
            str(FPS * 2),
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    info = MediaInfo(
        duration_sec=SOURCE_DURATION,
        width=640,
        height=360,
        fps=float(FPS),
        video_codec="h264",
        audio_codec="aac",
        container="mp4",
    )
    return LocalMedia(path=path, info=info, source_uri=f"file://{path}")


class TestPreciseCut:
    @pytest.mark.parametrize("span", [(3.0, 8.0), (5.5, 12.25), (0.0, 4.0)])
    async def test_duration_matches_request(
        self,
        source: LocalMedia,
        tmp_path: Path,
        span: tuple[float, float],
    ) -> None:
        """DoD: длительность вырезанного клипа совпадает с заданной ± 0.05 с.

        Проверяется на трёх отрезках, ни один из которых не начинается
        с ключевого кадра, — то есть в самом неудобном для точности случае.
        """
        window = TimeRange(*span)
        cutter = FFmpegClipCutter(binary=str(FFMPEG), probe_binary=str(FFPROBE))
        result = await cutter.cut_detailed(
            source, window, RenderSpec(preset="ultrafast"), tmp_path / "clip.mp4"
        )

        assert result.mode == "encode"
        assert result.duration_sec is not None
        assert abs(result.duration_sec - window.duration) <= DURATION_TOLERANCE

    async def test_first_frame_is_not_black(self, source: LocalMedia, tmp_path: Path) -> None:
        """DoD: нет чёрных кадров в начале.

        Именно этим и опасна перемотка по ключевым кадрам: клип, начатый
        не с ключевого кадра, открывается пустотой. Яркость первого кадра
        меряется фильтром `signalstats`, а не глазами.
        """
        cutter = FFmpegClipCutter(binary=str(FFMPEG), probe_binary=str(FFPROBE))
        clip = await cutter.cut(
            source, TimeRange(7.3, 12.3), RenderSpec(preset="ultrafast"), tmp_path / "clip.mp4"
        )

        completed = subprocess.run(
            [
                str(FFMPEG),
                "-v",
                "error",
                "-i",
                str(clip),
                "-frames:v",
                "1",
                "-vf",
                # ``file=-`` обязателен: без него фильтр печатает через
                # общий лог, который погашен уровнем `error`, и на выходе
                # оказывается пустая строка вместо числа.
                "signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=-",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
        )
        output = completed.stdout + completed.stderr
        brightness = float(output.rsplit("=", 1)[-1].strip())
        assert brightness > 16.0, f"первый кадр почти чёрный: YAVG={brightness}"

    async def test_audio_and_video_start_together(self, source: LocalMedia, tmp_path: Path) -> None:
        """DoD: нет рассинхрона A/V в начале клипа.

        Проверяется по меткам времени первых пакетов: разошедшиеся начала
        дорожек — это и есть рассинхрон, который на слух заметен уже
        от 0.1 с.
        """
        cutter = FFmpegClipCutter(binary=str(FFMPEG), probe_binary=str(FFPROBE))
        clip = await cutter.cut(
            source, TimeRange(4.4, 9.4), RenderSpec(preset="ultrafast"), tmp_path / "clip.mp4"
        )

        payload = _probe(clip, "stream=codec_type,start_time")
        starts = {
            stream["codec_type"]: float(stream.get("start_time", 0.0))
            for stream in payload["streams"]  # type: ignore[index]
        }
        assert set(starts) == {"video", "audio"}
        assert abs(starts["video"] - starts["audio"]) <= 0.05

    async def test_source_without_audio_is_not_an_error(
        self, source: LocalMedia, tmp_path: Path
    ) -> None:
        silent = tmp_path / "silent.mp4"
        subprocess.run(
            [
                str(FFMPEG),
                "-y",
                "-v",
                "error",
                "-an",
                "-i",
                str(source.path),
                "-c",
                "copy",
                str(silent),
            ],
            check=True,
            capture_output=True,
        )
        media = LocalMedia(
            path=silent,
            info=MediaInfo(
                duration_sec=SOURCE_DURATION,
                width=640,
                height=360,
                video_codec="h264",
                audio_codec=None,
                container="mp4",
            ),
            source_uri=f"file://{silent}",
        )

        cutter = FFmpegClipCutter(binary=str(FFMPEG), probe_binary=str(FFPROBE))
        clip = await cutter.cut(
            media, TimeRange(2.0, 6.0), RenderSpec(preset="ultrafast"), tmp_path / "clip.mp4"
        )

        payload = _probe(clip, "stream=codec_type")
        types = {stream["codec_type"] for stream in payload["streams"]}  # type: ignore[index]
        assert types == {"video"}


class TestVerticalOutput:
    def _track(self, positions: list[int]) -> list[CropKeyframe]:
        return [
            CropKeyframe(
                time_sec=index * 0.25,
                box=CropBox(x=x, y=0, width=202, height=360),
            )
            for index, x in enumerate(positions)
        ]

    @pytest.fixture
    def clip(self, source: LocalMedia, tmp_path_factory: pytest.TempPathFactory) -> Path:
        import asyncio

        target = tmp_path_factory.mktemp("vertical") / "h.mp4"
        cutter = FFmpegClipCutter(binary=str(FFMPEG), probe_binary=str(FFPROBE))
        return asyncio.run(
            cutter.cut(source, TimeRange(2.0, 8.0), RenderSpec(preset="ultrafast"), target)
        )

    @pytest.mark.parametrize(
        ("positions", "expected_mode"),
        [
            ([], MODE_BLURPAD),
            ([200] * 24, MODE_STATIC),
            (list(range(0, 240, 10)), MODE_TRACK),
        ],
        ids=["без лиц", "неподвижный кроп", "по траектории"],
    )
    async def test_frame_is_exactly_vertical(
        self,
        clip: Path,
        tmp_path: Path,
        positions: list[int],
        expected_mode: str,
    ) -> None:
        """DoD: вертикаль ровно 1080×1920 — во всех трёх режимах.

        Включая фоллбек: клип без единого лица обязан дать файл нужного
        размера, а не падение (тоже требование DoD).
        """
        renderer = FFmpegVerticalRenderer(binary=str(FFMPEG), probe_binary=str(FFPROBE))
        path, mode = await renderer.reframe_vertical(
            clip,
            self._track(positions),
            RenderSpec(preset="ultrafast"),
            tmp_path / "v.mp4",
        )

        assert mode == expected_mode
        payload = _probe(path, "stream=width,height", select="v:0")
        stream = payload["streams"][0]  # type: ignore[index]
        assert (stream["width"], stream["height"]) == (VERTICAL_WIDTH, VERTICAL_HEIGHT)

    async def test_vertical_keeps_the_audio(self, clip: Path, tmp_path: Path) -> None:
        renderer = FFmpegVerticalRenderer(binary=str(FFMPEG), probe_binary=str(FFPROBE))
        path, _ = await renderer.reframe_vertical(
            clip, self._track([200] * 24), RenderSpec(preset="ultrafast"), tmp_path / "v.mp4"
        )

        payload = _probe(path, "stream=codec_type")
        types = {stream["codec_type"] for stream in payload["streams"]}  # type: ignore[index]
        assert types == {"video", "audio"}


class TestBurnedSubtitles:
    """DoD Этапа 9: вжигание работает и не ломает A/V-синхронизацию."""

    @pytest.fixture
    def subtitles(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        """ASS под кадр исходника — с текстом на всём протяжении клипа."""
        from core.domain.enums import SubtitleFormat
        from core.domain.media import Transcript, TranscriptSegment, TranscriptWord
        from subtitles.writer import ClipSubtitleWriter

        words: list[TranscriptWord] = []
        moment = 2.0
        for token in ["первое", "слово", "второе", "слово", "третье", "слово"]:
            words.append(TranscriptWord(text=token, start=moment, end=moment + 0.6))
            moment += 0.7
        transcript = Transcript(
            segments=(
                TranscriptSegment(text="текст", range=TimeRange(2.0, moment), words=tuple(words)),
            ),
            language="ru",
        )
        target = tmp_path_factory.mktemp("subs") / "clip.ass"
        return ClipSubtitleWriter().write(
            transcript, TimeRange(2.0, 8.0), SubtitleFormat.ASS, target, width=640, height=360
        )

    async def test_text_appears_in_the_picture(
        self, source: LocalMedia, subtitles: Path, tmp_path: Path
    ) -> None:
        """Проверяется результат, а не факт запуска ffmpeg: кадр с текстом
        обязан отличаться от того же кадра без него."""
        cutter = FFmpegClipCutter(binary=str(FFMPEG), probe_binary=str(FFPROBE))
        span = TimeRange(2.0, 8.0)
        plain = await cutter.cut(source, span, RenderSpec(preset="ultrafast"), tmp_path / "p.mp4")
        burned = await cutter.cut(
            source,
            span,
            RenderSpec(preset="ultrafast", burn_subtitles_path=subtitles),
            tmp_path / "b.mp4",
        )

        frames = [
            _first_frame(path, 1.0, tmp_path / f"{path.stem}.png") for path in (plain, burned)
        ]
        assert frames[0] != frames[1], "кадр не изменился — субтитры не вжглись"

    async def test_burning_keeps_duration_and_sync(
        self, source: LocalMedia, subtitles: Path, tmp_path: Path
    ) -> None:
        cutter = FFmpegClipCutter(binary=str(FFMPEG), probe_binary=str(FFPROBE))
        span = TimeRange(2.0, 8.0)
        result = await cutter.cut_detailed(
            source,
            span,
            RenderSpec(preset="ultrafast", burn_subtitles_path=subtitles),
            tmp_path / "b.mp4",
        )

        assert result.duration_sec is not None
        assert abs(result.duration_sec - span.duration) <= DURATION_TOLERANCE

        payload = _probe(result.path, "stream=codec_type,start_time")
        starts = {
            stream["codec_type"]: float(stream.get("start_time", 0.0))
            for stream in payload["streams"]  # type: ignore[index]
        }
        assert abs(starts["video"] - starts["audio"]) <= 0.05


def _first_frame(video: Path, moment: float, target: Path) -> bytes:
    subprocess.run(
        [
            str(FFMPEG),
            "-y",
            "-v",
            "error",
            "-ss",
            f"{moment}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(target),
        ],
        check=True,
        capture_output=True,
    )
    return target.read_bytes()


class TestThumbnail:
    async def test_thumbnail_is_a_jpeg_of_requested_width(
        self, source: LocalMedia, tmp_path: Path
    ) -> None:
        renderer = FFmpegClipRenderer(
            binary=str(FFMPEG), probe_binary=str(FFPROBE), thumbnail_width=320
        )
        clip = await renderer.cut(
            source, TimeRange(1.0, 5.0), RenderSpec(preset="ultrafast"), tmp_path / "c.mp4"
        )
        path = await renderer.thumbnail(clip, 1.0, tmp_path / "c.jpg")

        payload = _probe(path, "stream=width,codec_name")
        stream = payload["streams"][0]  # type: ignore[index]
        assert stream["codec_name"] == "mjpeg"
        assert stream["width"] == 320
