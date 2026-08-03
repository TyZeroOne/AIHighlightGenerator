"""Тесты доменных сущностей и вспомогательных типов."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from core.domain.analysis import (
    ARTIFACT_AUDIO,
    ARTIFACT_TRANSCRIPT,
    ARTIFACT_VIDEO,
    AnalysisContext,
    ScoringProfile,
)
from core.domain.entities import Job, Scene, StageState
from core.domain.enums import JobStage, JobStatus
from core.domain.media import (
    AudioTrack,
    LocalMedia,
    MediaInfo,
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)
from core.domain.values import FeatureValue, MediaWindow, TimeRange

pytestmark = pytest.mark.unit


def make_media(*, with_audio: bool = True, with_video: bool = True) -> LocalMedia:
    return LocalMedia(
        path=Path("/tmp/movie.mkv"),
        info=MediaInfo(
            duration_sec=120.0,
            video_codec="h264" if with_video else None,
            audio_codec="aac" if with_audio else None,
        ),
        source_uri="file:///tmp/movie.mkv",
    )


class TestScene:
    def test_missing_feature_returns_default(self) -> None:
        """Отсутствие признака — норма: анализатор мог упасть или быть отключён."""
        scene = Scene(
            id=uuid4(),
            video_id=uuid4(),
            job_id=uuid4(),
            index=0,
            range=TimeRange(0.0, 10.0),
            features={"audio.loudness.peak": FeatureValue("audio.loudness.peak", -3.2)},
        )
        assert scene.feature("audio.loudness.peak") == pytest.approx(-3.2)
        assert scene.feature("vision.motion.mean") == 0.0
        assert scene.feature("vision.motion.mean", default=0.5) == 0.5


class TestJob:
    def test_stage_completion_drives_idempotency(self) -> None:
        job = Job(
            id=uuid4(),
            video_id=uuid4(),
            user_id=uuid4(),
            status=JobStatus.RUNNING,
            stages={
                JobStage.INGEST: StageState(JobStage.INGEST, "succeeded"),
                JobStage.DETECT_SCENES: StageState(JobStage.DETECT_SCENES, "running"),
            },
        )
        assert job.is_stage_done(JobStage.INGEST)
        assert not job.is_stage_done(JobStage.DETECT_SCENES)
        assert not job.is_stage_done(JobStage.TRANSCRIBE)  # ещё не начинался

    def test_skipped_counts_as_done(self) -> None:
        """Пропущенный этап (например, аудио у видео без звука) не должен
        заставлять конвейер выполнять его снова при перезапуске."""
        state = StageState(JobStage.EXTRACT_AUDIO, "skipped")
        assert state.is_done


class TestJobProgress:
    """Прогресс джоба — взвешенная сумма прогресса этапов (док 04, §4.8)."""

    def _job(self) -> Job:
        return Job(
            id=uuid4(),
            video_id=uuid4(),
            user_id=uuid4(),
            status=JobStatus.QUEUED,
        )

    def test_running_stage_does_not_add_progress_by_itself(self) -> None:
        job = self._job().with_stage(JobStage.DETECT_SCENES, "running")

        assert job.stage is JobStage.DETECT_SCENES
        assert job.progress == pytest.approx(0.0)

    def test_finished_stage_adds_its_weight(self) -> None:
        job = (
            self._job()
            .with_stage(JobStage.INGEST, "succeeded")  # вес 3
            .with_stage(JobStage.DETECT_SCENES, "succeeded")  # вес 7
        )

        assert job.progress == pytest.approx(10.0)

    def test_partial_progress_of_a_long_stage_counts(self) -> None:
        """Транскрипция идёт минутами и рапортует прогресс сама (вес 25)."""
        job = self._job().with_stage(JobStage.TRANSCRIBE, "running", progress=60.0)

        assert job.progress == pytest.approx(15.0)

    def test_succeeded_counts_fully_even_with_stale_progress(self) -> None:
        """Длинный этап может закончиться, не дойдя до ровных 100 %."""
        job = self._job().with_stage(JobStage.TRANSCRIBE, "succeeded", progress=61.0)

        assert job.progress == pytest.approx(25.0)

    def test_skipped_stage_counts_as_done(self) -> None:
        """Видео без звука: транскрипции не будет никогда, и джоб не имеет
        права застрять на 75 % из-за этапа, которого не случится."""
        job = self._job().with_stage(JobStage.TRANSCRIBE, "skipped")

        assert job.progress == pytest.approx(25.0)

    def test_failed_stage_adds_nothing(self) -> None:
        job = self._job().with_stage(JobStage.DETECT_SCENES, "failed", error="boom")

        assert job.progress == pytest.approx(0.0)
        assert job.stages[JobStage.DETECT_SCENES].error == "boom"

    def test_all_stages_done_give_exactly_100(self) -> None:
        """Сумма весов равна 100 — иначе завершённый джоб показывал бы 97 %."""
        job = self._job()
        for stage in JobStage:
            job = job.with_stage(stage, "succeeded")

        assert job.progress == pytest.approx(100.0)

    def test_with_stage_does_not_mutate_the_original(self) -> None:
        original = self._job().with_stage(JobStage.INGEST, "succeeded")

        updated = original.with_stage(JobStage.DETECT_SCENES, "succeeded")

        assert JobStage.DETECT_SCENES not in original.stages
        assert original.progress == pytest.approx(3.0)
        assert updated.progress == pytest.approx(10.0)


class TestTranscript:
    def _transcript(self) -> Transcript:
        return Transcript(
            segments=(
                TranscriptSegment(
                    text="Держись, они прямо за нами",
                    range=TimeRange(10.0, 13.0),
                    words=(
                        TranscriptWord("Держись", 10.0, 10.6),
                        TranscriptWord("они", 10.8, 11.1),
                        TranscriptWord("прямо", 11.2, 11.7),
                    ),
                ),
                TranscriptSegment(text="Сворачивай направо", range=TimeRange(14.0, 16.0)),
            ),
            language="ru",
        )

    def test_text_joins_segments(self) -> None:
        assert "Держись" in self._transcript().text
        assert "Сворачивай" in self._transcript().text

    def test_within_selects_overlapping_segments(self) -> None:
        segments = self._transcript().within(TimeRange(12.0, 15.0))
        assert len(segments) == 2  # оба пересекаются с окном

    def test_within_excludes_non_overlapping(self) -> None:
        assert self._transcript().within(TimeRange(100.0, 110.0)) == []

    def test_words_within(self) -> None:
        words = self._transcript().words_within(TimeRange(10.0, 11.0))
        assert [w.text for w in words] == ["Держись", "они"]

    def test_empty_transcript(self) -> None:
        empty = Transcript()
        assert empty.is_empty
        assert empty.text == ""


class TestAnalysisContext:
    def test_artifacts_reflect_reality(self) -> None:
        context = AnalysisContext(
            job_id=uuid4(),
            video_id=uuid4(),
            window=MediaWindow("v1", TimeRange(0.0, 120.0)),
            media=make_media(),
            workdir=Path("/tmp/work"),
            audio=AudioTrack(Path("/tmp/a.wav"), 16000, 1, 120.0),
            transcript=Transcript(segments=(TranscriptSegment("привет", TimeRange(1.0, 2.0)),)),
        )
        assert context.available_artifacts() == {
            ARTIFACT_VIDEO,
            ARTIFACT_AUDIO,
            ARTIFACT_TRANSCRIPT,
        }

    def test_video_without_audio(self) -> None:
        """Видео без звуковой дорожки — штатный случай, а не ошибка."""
        context = AnalysisContext(
            job_id=uuid4(),
            video_id=uuid4(),
            window=MediaWindow("v1", TimeRange(0.0, 120.0)),
            media=make_media(with_audio=False),
            workdir=Path("/tmp/work"),
        )
        assert context.available_artifacts() == {ARTIFACT_VIDEO}

    def test_empty_transcript_is_not_an_artifact(self) -> None:
        """Пустой транскрипт (музыка без речи) не должен включать речевые анализаторы."""
        context = AnalysisContext(
            job_id=uuid4(),
            video_id=uuid4(),
            window=MediaWindow("v1", TimeRange(0.0, 120.0)),
            media=make_media(),
            workdir=Path("/tmp/work"),
            transcript=Transcript(),
        )
        assert ARTIFACT_TRANSCRIPT not in context.available_artifacts()


class TestScoringProfile:
    def test_valid_profile(self) -> None:
        profile = ScoringProfile(name="movie", group_weights={"action": 0.5, "emotion": 0.5})
        assert profile.max_clips == 10

    def test_rejects_empty_weights(self) -> None:
        with pytest.raises(ValueError, match="весов"):
            ScoringProfile(name="broken", group_weights={})

    def test_rejects_inverted_durations(self) -> None:
        with pytest.raises(ValueError, match="min_duration"):
            ScoringProfile(
                name="broken",
                group_weights={"action": 1.0},
                min_duration_sec=90.0,
                max_duration_sec=15.0,
            )
