"""Нарезка клипов: аргументы ffmpeg, траектория кропа, сценарий этапа.

Три уровня, три вида проверок.

**Чистые функции** (сглаживание траектории, интерполяция, сборка команд)
проверяются числами: у них нет ни файлов, ни процессов, и требования
к ним формулируются как свойства — «производная ограничена», «окно
не выезжает за кадр».

**Сборка аргументов ffmpeg** проверяется как текст. Запускать ffmpeg
ради того, чтобы узнать, попал ли `-map 0:v:0` в команду, незачем —
а именно его отсутствие на живом прогоне собрало клип из одного звука.

**Сценарий этапа** проверяется на фейковых портах: вся его логика —
это деградация (что происходит, когда падает вертикаль, один клип,
все клипы), и она не зависит ни от ffmpeg, ни от базы.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from core.domain.entities import Clip, Job, StageState, Video
from core.domain.enums import (
    AssetKind,
    AssetOwnerType,
    ClipStatus,
    JobStage,
    JobStatus,
    SourceKind,
    SubtitleFormat,
    VideoStatus,
)
from core.domain.errors import ProcessingError, ValidationError
from core.domain.media import LocalMedia, MediaInfo, Transcript, TranscriptSegment, TranscriptWord
from core.domain.render import VERTICAL_HEIGHT, VERTICAL_WIDTH, CropKeyframe, RenderSpec
from core.domain.values import CropBox, TimeRange
from core.usecases.render import RenderClips, RenderSettings, thumbnail_moment
from video_editor.cutter import (
    KEYFRAME_TOLERANCE,
    _clamp,
    _encode_args,
    _map_args,
    can_stream_copy,
)
from video_editor.encoder import audio_args, loudnorm_filter, thumbnail_args, video_args
from video_editor.vertical import (
    MODE_BLURPAD,
    MODE_TRACK,
    FFmpegVerticalRenderer,
    blurpad_chain,
    interpolate,
    sendcmd_script,
)
from vision.faces import YuNetFaceDetector
from vision.reframe import build_boxes, crop_size, fill_gaps, limit_speed, smooth, split_on_gaps

pytestmark = pytest.mark.unit


def make_media(
    *, has_audio: bool = True, codec: str = "h264", duration: float = 600.0
) -> LocalMedia:
    return LocalMedia(
        path=Path("/media/in/source.mp4"),
        info=MediaInfo(
            duration_sec=duration,
            width=1920,
            height=1080,
            fps=25.0,
            video_codec=codec,
            audio_codec="aac" if has_audio else None,
            container="mp4",
        ),
        source_uri="file:///media/in/source.mp4",
    )


# ---------------------------------------------------------------- резка
class TestCutArguments:
    def test_seek_goes_before_input(self) -> None:
        """`-ss` до `-i` — это разница между секундами и минутами на перемотке."""
        args = _encode_args(
            make_media(), TimeRange(921.5, 966.0), RenderSpec(), Path("/tmp/clip.mp4")
        )
        assert args.index("-ss") < args.index("-i")
        assert args[args.index("-ss") + 1] == "921.500"
        assert args[args.index("-t") + 1] == "44.500"

    def test_video_stream_is_mapped_explicitly(self) -> None:
        """Регресс живого дефекта: без явного `-map 0:v:0` в клипе остаётся
        только звук — первый же `-map` выключает автоматический выбор."""
        args = _encode_args(make_media(), TimeRange(0.0, 10.0), RenderSpec(), Path("/tmp/clip.mp4"))
        assert "0:v:0" in args

    def test_missing_audio_track_is_not_an_error(self) -> None:
        """Видео без звука — штатная ситуация, а не повод падать."""
        assert "-an" in _map_args(make_media(has_audio=False))
        assert "0:a:0?" in _map_args(make_media())

    def test_range_beyond_file_is_clamped(self) -> None:
        """Границы сцен приходят из детектора и могут на доли секунды
        выходить за конец файла — тогда проверка длительности ругалась бы
        на несуществующую проблему."""
        assert _clamp(TimeRange(90.0, 130.0), 100.0) == TimeRange(90.0, 100.0)

    def test_start_beyond_file_is_an_error(self) -> None:
        with pytest.raises(ValidationError):
            _clamp(TimeRange(120.0, 130.0), 100.0)


class TestStreamCopy:
    """Копирование потока несовместимо с большинством наших требований —
    и это выясняется до запуска ffprobe, а не после."""

    def test_loudness_normalization_forbids_copy(self) -> None:
        assert not can_stream_copy(make_media(), RenderSpec(loudness_lufs=-14.0))

    def test_scaling_forbids_copy(self) -> None:
        spec = RenderSpec(loudness_lufs=None, width=1080, height=1920)
        assert not can_stream_copy(make_media(), spec)

    def test_foreign_codec_forbids_copy(self) -> None:
        spec = RenderSpec(loudness_lufs=None)
        assert not can_stream_copy(make_media(codec="hevc"), spec)

    def test_plain_h264_without_processing_allows_copy(self) -> None:
        assert can_stream_copy(make_media(), RenderSpec(loudness_lufs=None))

    def test_tolerance_is_about_one_frame(self) -> None:
        """40 мс — кадр при 25 fps. Больше означало бы разрешить копию там,
        где начало клипа заведомо не на ключевом кадре."""
        assert pytest.approx(1 / 25, abs=1e-3) == KEYFRAME_TOLERANCE


# ------------------------------------------------------------ кодирование
class TestEncoderArguments:
    def test_loudness_target_reaches_filter(self) -> None:
        assert "I=-14" in loudnorm_filter(-14.0)
        assert "TP=-1.5" in loudnorm_filter(-14.0)

    def test_audio_is_downmixed_to_stereo(self) -> None:
        """5.1 из кинофайла в браузере без даунмикса теряет центральный
        канал, то есть диалоги."""
        args = audio_args(RenderSpec())
        assert args[args.index("-ac") + 1] == "2"

    def test_normalization_can_be_switched_off(self) -> None:
        assert "-af" not in audio_args(RenderSpec(loudness_lufs=None))

    def test_scale_uses_even_placeholder(self) -> None:
        """−2, а не −1: нечётная сторона несовместима с yuv420p."""
        args = video_args(RenderSpec(width=1080))
        assert "scale=1080:-2" in args

    def test_explicit_chain_wins_over_size(self) -> None:
        args = video_args(RenderSpec(width=1080, height=1920), filter_chain="crop=1:2:3:4")
        assert args[args.index("-vf") + 1] == "crop=1:2:3:4"

    def test_thumbnail_takes_single_frame(self) -> None:
        args = thumbnail_args(Path("/tmp/c.mp4"), 5.0, 640, Path("/tmp/c.jpg"))
        assert args[args.index("-frames:v") + 1] == "1"
        assert args.index("-ss") < args.index("-i")


# -------------------------------------------------------------- траектория
class TestTrajectory:
    def test_gaps_are_filled_by_holding_position(self) -> None:
        assert fill_gaps([None, 0.4, None, None, 0.6]) == [0.4, 0.4, 0.4, 0.4, 0.6]

    def test_all_unknown_falls_back_to_centre(self) -> None:
        assert fill_gaps([None, None]) == [0.5, 0.5]

    def test_smoothing_reduces_jitter(self) -> None:
        noisy = [0.5, 0.6, 0.5, 0.6, 0.5, 0.6, 0.5]
        smoothed = smooth(noisy, 5)
        spread = max(smoothed) - min(smoothed)
        assert spread < (max(noisy) - min(noisy))

    def test_speed_limit_bounds_the_derivative(self) -> None:
        """Числовая формулировка требования «кроп не дёргается»."""
        jumpy = [0.0, 1.0, 0.0, 1.0]
        limited = limit_speed(jumpy, 0.1)
        steps = [abs(b - a) for a, b in itertools.pairwise(limited)]
        assert max(steps) <= 0.1 + 1e-9

    def test_long_gap_splits_the_trajectory(self) -> None:
        """Длинный провал без лиц — почти наверняка склейка, и пропуск
        отходит следующему куску, чтобы прыжок пришёлся на кадры без лица."""
        chunks = split_on_gaps([0.2, 0.2, None, None, None, 0.8, 0.8], 2)
        assert len(chunks) == 2
        assert chunks[0] == [0.2, 0.2]
        assert chunks[1] == [None, None, None, 0.8, 0.8]

    def test_short_gap_does_not_split(self) -> None:
        assert len(split_on_gaps([0.2, None, 0.3], 2)) == 1

    def test_crop_window_matches_target_ratio(self) -> None:
        width, height = crop_size(1920, 1080, VERTICAL_WIDTH, VERTICAL_HEIGHT)
        assert height == 1080
        assert width == pytest.approx(1080 * 9 / 16, abs=2)
        assert width % 2 == 0

    def test_vertical_source_is_limited_by_height(self) -> None:
        """Исходник уже 9:16 — окно совпадает с кадром, обрезать нечего."""
        width, height = crop_size(720, 1280, VERTICAL_WIDTH, VERTICAL_HEIGHT)
        assert (width, height) == (720, 1280)

    def test_boxes_never_leave_the_frame(self) -> None:
        boxes = build_boxes(
            [0.0, 0.5, 1.0], [0.0, 1.0, 2.0], frame_width=1920, crop_width=608, crop_height=1080
        )
        for keyframe in boxes:
            assert keyframe.box.x >= 0
            assert keyframe.box.x + keyframe.box.width <= 1920


# ---------------------------------------------------------------- вертикаль
def track_of(*positions: int, step: float = 0.25) -> list[CropKeyframe]:
    return [
        CropKeyframe(time_sec=index * step, box=CropBox(x=x, y=0, width=608, height=1080))
        for index, x in enumerate(positions)
    ]


class TestVerticalModes:
    @pytest.fixture
    def renderer(self) -> FFmpegVerticalRenderer:
        return FFmpegVerticalRenderer()

    def test_empty_track_means_blurpad(self, renderer: FFmpegVerticalRenderer) -> None:
        args = renderer._blurpad_args(Path("/tmp/h.mp4"), RenderSpec(), Path("/tmp/v.mp4"))
        assert "[v]" in args
        assert "gblur" in args[args.index("-filter_complex") + 1]

    def test_blurpad_covers_screen_and_fits_frame(self) -> None:
        """Фон растягивается «в плюс», передний план — «в минус»:
        первый обязан покрыть экран, второй — влезть целиком."""
        chain = blurpad_chain(VERTICAL_WIDTH, VERTICAL_HEIGHT)
        assert "force_original_aspect_ratio=increase" in chain
        assert "force_original_aspect_ratio=decrease" in chain

    def test_static_mode_for_motionless_track(self, renderer: FFmpegVerticalRenderer) -> None:
        args = renderer._static_args(
            Path("/tmp/h.mp4"), track_of(100, 101, 100), RenderSpec(), Path("/tmp/v.mp4")
        )
        chain = args[args.index("-vf") + 1]
        assert chain.startswith("crop=")
        assert f"scale={VERTICAL_WIDTH}:{VERTICAL_HEIGHT}" in chain

    def test_video_is_mapped_in_every_mode(self, renderer: FFmpegVerticalRenderer) -> None:
        """Тот же регресс, что и в резке: `-map` без видеодорожки собирает
        файл из одного звука, молча и с нулевым кодом возврата."""
        static = renderer._static_args(
            Path("/tmp/h.mp4"), track_of(100, 100), RenderSpec(), Path("/tmp/v.mp4")
        )
        blurpad = renderer._blurpad_args(Path("/tmp/h.mp4"), RenderSpec(), Path("/tmp/v.mp4"))
        assert "0:v:0" in static
        assert "[v]" in blurpad

    def test_audio_is_copied_not_reencoded(self, renderer: FFmpegVerticalRenderer) -> None:
        """Источник вертикали — наш же клип: звук в нём уже нормализован."""
        args = renderer._static_args(
            Path("/tmp/h.mp4"), track_of(100, 100), RenderSpec(), Path("/tmp/v.mp4")
        )
        assert args[args.index("-c:a") + 1] == "copy"


class TestSendcmd:
    def test_interpolation_fills_between_keyframes(self) -> None:
        """Точки считаются 4 раза в секунду, команды нужны на каждый кадр:
        sendcmd не интерполирует, он переставляет окно ступенькой."""
        dense = interpolate(track_of(0, 100, step=1.0), fps=25.0)
        assert len(dense) > 25
        values = [value for _, value in dense]
        assert values[0] == 0
        assert values[-1] == 100
        assert values == sorted(values)

    def test_script_skips_repeated_positions(self) -> None:
        script = sendcmd_script([(0.0, 10), (0.04, 10), (0.08, 12)])
        assert script.count("crop x") == 2

    def test_script_syntax(self) -> None:
        line = sendcmd_script([(1.5, 42)]).strip()
        assert line == "1.500 crop x 42;"


# ------------------------------------------------------------ сценарий этапа
@dataclass
class FakeRenderer:
    """Порт ``ClipRenderer`` без ffmpeg."""

    fail_cut_for: set[int] | None = None
    fail_vertical: bool = False

    async def cut(
        self, media: LocalMedia, span: TimeRange, spec: RenderSpec, destination: Path
    ) -> Path:
        rank = int(destination.stem.split("_")[0])
        if self.fail_cut_for and rank in self.fail_cut_for:
            raise ProcessingError("резка не удалась", rank=rank)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"video")
        return destination

    async def reframe_vertical(
        self,
        source: Path,
        track: Sequence[CropKeyframe],
        spec: RenderSpec,
        destination: Path,
    ) -> tuple[Path, str]:
        if self.fail_vertical:
            raise ProcessingError("вертикаль не собралась")
        destination.write_bytes(b"vertical")
        return destination, MODE_TRACK if track else MODE_BLURPAD

    async def thumbnail(self, source: Path, at_sec: float, destination: Path) -> Path:
        destination.write_bytes(b"jpeg")
        return destination


@dataclass
class FakePlanner:
    boxes: list[CropKeyframe] | None = None

    async def plan(
        self, source: Path, *, target_width: int, target_height: int
    ) -> Sequence[CropKeyframe]:
        return self.boxes if self.boxes is not None else track_of(10, 200)


class FakeProvider:
    def __init__(self, media: LocalMedia) -> None:
        self.media = media
        self.released = False

    def supports(self, uri: str) -> bool:
        return True

    async def acquire(self, uri: str) -> LocalMedia:
        return self.media

    async def release(self, media: LocalMedia) -> None:
        self.released = True


class FakeStorage:
    def __init__(self) -> None:
        self.objects: dict[str, int] = {}

    async def put(self, key: str, source: Path, *, content_type: str | None = None) -> str:
        self.objects[key] = source.stat().st_size
        return key


class FakeClipRepo:
    def __init__(self, clips: list[Clip]) -> None:
        self.clips = {clip.id: clip for clip in clips}

    async def list_for_job(self, job_id: UUID) -> Sequence[Clip]:
        return sorted(self.clips.values(), key=lambda clip: clip.rank)

    async def update(self, clip: Clip) -> Clip:
        self.clips[clip.id] = clip
        return clip


class FakeAssetRepo:
    def __init__(self) -> None:
        self.records: list[tuple[AssetOwnerType, UUID, AssetKind, str]] = []

    async def add(
        self,
        *,
        owner_type: AssetOwnerType,
        owner_id: UUID,
        kind: AssetKind,
        storage_key: str,
        mime: str = "",
        size_bytes: int = 0,
        meta: dict[str, object] | None = None,
        checksum: str | None = None,
        ttl_hours: int | None = None,
    ) -> UUID:
        self.records.append((owner_type, owner_id, kind, storage_key))
        return uuid4()

    async def delete_for_owner(self, owner_type: AssetOwnerType, owner_id: UUID) -> None:
        self.records = [
            record
            for record in self.records
            if not (record[0] is owner_type and record[1] == owner_id)
        ]


class FakeJobRepo:
    def __init__(self, job: Job) -> None:
        self.job = job

    async def get(self, job_id: UUID) -> Job:
        return self.job

    async def update(self, job: Job) -> Job:
        self.job = job
        return job


class FakeVideoRepo:
    def __init__(self, video: Video) -> None:
        self.video = video

    async def get(self, video_id: UUID) -> Video:
        return self.video


class FakeTranscriptRepo:
    def __init__(self, transcript: Transcript | None = None) -> None:
        self.transcript = transcript or Transcript()

    async def get_for_job(self, job_id: UUID) -> Transcript:
        return self.transcript


class FakeSubtitleWriter:
    """Порт ``SubtitleWriter`` без разбиения на реплики.

    Проверять здесь надо не содержимое субтитров (для этого есть тесты
    самих форматов), а то, что сценарий их пишет, кладёт в хранилище
    и переживает их отсутствие.
    """

    def __init__(self, *, fails: bool = False) -> None:
        self.fails = fails
        self.calls: list[tuple[str, int | None, int | None]] = []

    def write(
        self,
        transcript: Transcript,
        span: TimeRange,
        fmt: SubtitleFormat,
        destination: Path,
        *,
        width: int | None = None,
        height: int | None = None,
    ) -> Path:
        if self.fails:
            raise ProcessingError("субтитры не собрались")
        self.calls.append((fmt.value, width, height))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("текст", encoding="utf-8")
        return destination


class FakeUow:
    def __init__(
        self,
        jobs: FakeJobRepo,
        videos: FakeVideoRepo,
        clips: FakeClipRepo,
        assets: FakeAssetRepo,
        transcripts: FakeTranscriptRepo | None = None,
    ) -> None:
        self.jobs = jobs
        self.videos = videos
        self.clips = clips
        self.assets = assets
        self.transcripts = transcripts or FakeTranscriptRepo()

    async def __aenter__(self) -> FakeUow:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def commit(self) -> None:
        return None


@pytest.fixture
def scenario(tmp_path: Path) -> dict[str, object]:
    job_id, video_id = uuid4(), uuid4()
    user_id = uuid4()
    video = Video(
        id=video_id,
        user_id=user_id,
        source_uri="file:///media/in/source.mp4",
        source_kind=SourceKind.LOCAL_FILE,
        status=VideoStatus.READY,
        storage_key="videos/x/source.mp4",
        checksum_sha256="0" * 64,
    )
    job = Job(id=job_id, video_id=video_id, user_id=user_id, status=JobStatus.RUNNING)
    clips = [
        Clip(
            id=uuid4(),
            job_id=job_id,
            video_id=video_id,
            rank=rank,
            range=TimeRange(rank * 60.0, rank * 60.0 + 30.0),
            score=90.0 - rank,
            category="action",
            reason="",
        )
        for rank in (1, 2, 3)
    ]
    jobs = FakeJobRepo(job)
    repos = FakeUow(jobs, FakeVideoRepo(video), FakeClipRepo(clips), FakeAssetRepo())
    return {
        "job_id": job_id,
        "uow": repos,
        "provider": FakeProvider(make_media()),
        "storage": FakeStorage(),
        "workdir": tmp_path,
    }


def build(scenario: dict[str, object], renderer: FakeRenderer, **kwargs: object) -> RenderClips:
    return RenderClips(
        uow_factory=lambda: scenario["uow"],  # type: ignore[arg-type,return-value]
        renderer=renderer,  # type: ignore[arg-type]
        storage=scenario["storage"],  # type: ignore[arg-type]
        provider_resolver=lambda uri: scenario["provider"],  # type: ignore[arg-type,return-value]
        workdir=scenario["workdir"],  # type: ignore[arg-type]
        planner=kwargs.get("planner", FakePlanner()),  # type: ignore[arg-type]
        subtitles=kwargs.get("subtitles"),  # type: ignore[arg-type]
        settings=kwargs.get("settings", RenderSettings(concurrency=2)),  # type: ignore[arg-type]
    )


class TestRenderStage:
    @pytest.mark.asyncio
    async def test_renders_every_clip(self, scenario: dict[str, object]) -> None:
        result = await build(scenario, FakeRenderer()).execute(scenario["job_id"])  # type: ignore[arg-type]

        assert result.rendered == 3
        assert result.failed == 0
        assert result.vertical == 3
        assert result.thumbnails == 3
        storage: FakeStorage = scenario["storage"]  # type: ignore[assignment]
        assert len(storage.objects) == 9  # три вида файлов на клип

    @pytest.mark.asyncio
    async def test_one_broken_clip_does_not_stop_the_rest(
        self, scenario: dict[str, object]
    ) -> None:
        """Требование DoD: падение рендера одного клипа не ломает остальные."""
        renderer = FakeRenderer(fail_cut_for={2})
        result = await build(scenario, renderer).execute(scenario["job_id"])  # type: ignore[arg-type]

        assert (result.rendered, result.failed) == (2, 1)
        uow: FakeUow = scenario["uow"]  # type: ignore[assignment]
        statuses = {clip.rank: clip.status for clip in uow.clips.clips.values()}
        assert statuses == {1: ClipStatus.READY, 2: ClipStatus.FAILED, 3: ClipStatus.READY}

    @pytest.mark.asyncio
    async def test_failed_vertical_keeps_the_clip(self, scenario: dict[str, object]) -> None:
        """Горизонтальный файл собран — терять его из-за кадрирования незачем."""
        result = await build(scenario, FakeRenderer(fail_vertical=True)).execute(
            scenario["job_id"]  # type: ignore[arg-type]
        )

        assert result.rendered == 3
        assert result.vertical == 0
        assert len(result.warnings) == 3

    @pytest.mark.asyncio
    async def test_all_clips_broken_fails_the_stage(self, scenario: dict[str, object]) -> None:
        renderer = FakeRenderer(fail_cut_for={1, 2, 3})
        with pytest.raises(ProcessingError):
            await build(scenario, renderer).execute(scenario["job_id"])  # type: ignore[arg-type]

        uow: FakeUow = scenario["uow"]  # type: ignore[assignment]
        assert uow.jobs.job.stages[JobStage.RENDER_CLIPS].status == "failed"

    @pytest.mark.asyncio
    async def test_no_clips_skips_the_stage(self, scenario: dict[str, object]) -> None:
        uow: FakeUow = scenario["uow"]  # type: ignore[assignment]
        uow.clips.clips.clear()

        result = await build(scenario, FakeRenderer()).execute(scenario["job_id"])  # type: ignore[arg-type]

        assert result.skipped
        assert uow.jobs.job.stages[JobStage.RENDER_CLIPS].status == "skipped"

    @pytest.mark.asyncio
    async def test_repeated_run_is_idempotent(self, scenario: dict[str, object]) -> None:
        """``acks_late`` возвращает задачу в очередь после убийства воркера,
        и повторный запуск не должен резать всё заново."""
        uow: FakeUow = scenario["uow"]  # type: ignore[assignment]
        uow.jobs.job = replace(
            uow.jobs.job,
            stages={
                JobStage.RENDER_CLIPS: StageState(
                    stage=JobStage.RENDER_CLIPS, status="succeeded", progress=100.0
                )
            },
        )

        result = await build(scenario, FakeRenderer()).execute(scenario["job_id"])  # type: ignore[arg-type]

        assert result.reused
        storage: FakeStorage = scenario["storage"]  # type: ignore[assignment]
        assert not storage.objects

    @pytest.mark.asyncio
    async def test_source_is_released_even_on_failure(self, scenario: dict[str, object]) -> None:
        """Скачанная копия исходника обязана убираться в любом случае:
        иначе диск воркера кончится за сутки."""
        with pytest.raises(ProcessingError):
            await build(scenario, FakeRenderer(fail_cut_for={1, 2, 3})).execute(
                scenario["job_id"]  # type: ignore[arg-type]
            )

        provider: FakeProvider = scenario["provider"]  # type: ignore[assignment]
        assert provider.released

    @pytest.mark.asyncio
    async def test_vertical_can_be_switched_off(self, scenario: dict[str, object]) -> None:
        settings = RenderSettings(vertical=False, thumbnails=False)
        result = await build(scenario, FakeRenderer(), settings=settings, planner=None).execute(
            scenario["job_id"]  # type: ignore[arg-type]
        )

        assert result.vertical == 0
        assert result.thumbnails == 0
        assert result.rendered == 3


def speaking_transcript() -> Transcript:
    words = []
    moment = 60.0
    for token in ["раз", "два", "три", "четыре", "пять", "шесть"]:
        words.append(TranscriptWord(text=token, start=moment, end=moment + 0.4))
        moment += 0.5
    segment = TranscriptSegment(
        text="раз два три четыре пять шесть",
        range=TimeRange(60.0, moment),
        words=tuple(words),
    )
    return Transcript(segments=(segment,), language="ru")


class TestSubtitlesInRender:
    @pytest.mark.asyncio
    async def test_subtitle_files_are_written_and_stored(self, scenario: dict[str, object]) -> None:
        uow: FakeUow = scenario["uow"]  # type: ignore[assignment]
        uow.transcripts.transcript = speaking_transcript()
        writer = FakeSubtitleWriter()

        result = await build(scenario, FakeRenderer(), subtitles=writer).execute(
            scenario["job_id"]  # type: ignore[arg-type]
        )

        assert result.subtitles == 6  # два формата на каждый из трёх клипов
        storage: FakeStorage = scenario["storage"]  # type: ignore[assignment]
        assert sum(1 for key in storage.objects if key.endswith(".srt")) == 3
        assert sum(1 for key in storage.objects if key.endswith(".ass")) == 3

    @pytest.mark.asyncio
    async def test_video_without_speech_gets_no_subtitles(
        self, scenario: dict[str, object]
    ) -> None:
        """Пустой транскрипт — не ошибка: у видео без речи субтитров нет."""
        writer = FakeSubtitleWriter()

        result = await build(scenario, FakeRenderer(), subtitles=writer).execute(
            scenario["job_id"]  # type: ignore[arg-type]
        )

        assert result.subtitles == 0
        assert result.rendered == 3
        assert not writer.calls

    @pytest.mark.asyncio
    async def test_broken_subtitles_do_not_break_the_clip(
        self, scenario: dict[str, object]
    ) -> None:
        uow: FakeUow = scenario["uow"]  # type: ignore[assignment]
        uow.transcripts.transcript = speaking_transcript()

        result = await build(
            scenario, FakeRenderer(), subtitles=FakeSubtitleWriter(fails=True)
        ).execute(scenario["job_id"])  # type: ignore[arg-type]

        assert result.rendered == 3
        assert result.subtitles == 0
        assert len(result.warnings) == 6

    @pytest.mark.asyncio
    async def test_burning_asks_for_a_vertical_sized_file(
        self, scenario: dict[str, object]
    ) -> None:
        """ASS собирается под каждый кадр отдельно: файл, сделанный для
        горизонтали, в вертикали растянул бы текст поперёк."""
        uow: FakeUow = scenario["uow"]  # type: ignore[assignment]
        uow.transcripts.transcript = speaking_transcript()
        writer = FakeSubtitleWriter()

        await build(
            scenario,
            FakeRenderer(),
            subtitles=writer,
            settings=RenderSettings(burn=True, concurrency=1),
        ).execute(scenario["job_id"])  # type: ignore[arg-type]

        sizes = {(width, height) for fmt, width, height in writer.calls if fmt == "ass"}
        assert (VERTICAL_WIDTH, VERTICAL_HEIGHT) in sizes
        assert (1920, 1080) in sizes  # размер исходника из make_media


class TestDetectorThreadSafety:
    """Регресс живого дефекта: детектор лиц один на процесс, а рендер
    зовёт его из нескольких потоков сразу.

    Первый прогон этапа на трейлере уронил OpenCV прямо в графе сети
    (``buf.u == m.u in function 'forwardGraph'``): два клипа планировали
    кадрирование одновременно. Проверяется не отсутствие падения —
    воспроизвести его без гонки нельзя, — а само свойство, которого
    не хватало: вызовы ``detect`` не пересекаются во времени.
    """

    def test_detect_calls_do_not_overlap(self) -> None:
        import threading

        overlaps = 0
        inside = 0
        guard = threading.Lock()

        class SpyNet:
            def setInputSize(self, size: tuple[int, int]) -> None:  # noqa: N802
                return None

            def detect(self, frame: object) -> tuple[int, None]:
                nonlocal overlaps, inside
                with guard:
                    inside += 1
                    if inside > 1:
                        overlaps += 1
                time.sleep(0.005)
                with guard:
                    inside -= 1
                return 1, None

        class Frame:
            shape = (360, 640, 3)

        detector = YuNetFaceDetector(repo_id="x", filename="y")
        # Подменяем загруженную сеть: скачивать веса ради проверки
        # взаимного исключения незачем.
        detector._detector = SpyNet()

        threads = [threading.Thread(target=lambda: detector.detect(Frame())) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert overlaps == 0


def test_thumbnail_is_taken_from_the_first_third() -> None:
    """Не с начала: первые кадры часто заняты остатком предыдущей склейки."""
    assert thumbnail_moment(30.0) == pytest.approx(10.0)
    assert thumbnail_moment(0.0) == 0.0
