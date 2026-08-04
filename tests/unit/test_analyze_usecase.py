"""Этап анализа признаков на фейках.

Здесь проверяется не арифметика конкретного анализатора, а обещания,
которые этап даёт **любому** из них: план строится по реестру, сбой одного
не роняет остальных, отключённый конфигом не запускается, а результат
дописывается к тому, что посчитали предыдущие этапы.

Анализаторы объявлены прямо в тесте — и это само по себе доказательство
расширяемости: новый плагин появляется файлом с декоратором, без единой
правки в ``core``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import ClassVar
from uuid import UUID, uuid4

import pytest
from tests.fakes import (
    FakeSourceProvider,
    FakeStorage,
    FakeUnitOfWork,
    FakeVideoRepository,
    make_media_info,
    uow_factory,
)

from core.domain.analysis import ARTIFACT_AUDIO, ARTIFACT_VIDEO, AnalysisContext
from core.domain.entities import Job, Scene, Video
from core.domain.enums import JobStage, JobStatus, SourceKind, VideoStatus
from core.domain.keys import audio_key
from core.domain.media import LocalMedia
from core.domain.values import FeatureValue, TimeRange
from core.registry import register_analyzer
from core.usecases.analyze import AnalyzeFeatures

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

USER_ID = UUID("33333333-3333-3333-3333-333333333333")


@register_analyzer
class LoudAnalyzer:
    key: ClassVar[str] = "audio.loud"
    requires: ClassVar[frozenset[str]] = frozenset({ARTIFACT_AUDIO})
    produces: ClassVar[frozenset[str]] = frozenset({"audio.loud.mean"})
    cost: ClassVar[int] = 1

    async def analyze(
        self, context: AnalysisContext, scenes: Sequence[TimeRange]
    ) -> Mapping[int, Sequence[FeatureValue]]:
        return {
            index: [FeatureValue("audio.loud.mean", -20.0 + index)] for index in range(len(scenes))
        }


@register_analyzer
class QuietAnalyzer:
    key: ClassVar[str] = "audio.quiet"
    requires: ClassVar[frozenset[str]] = frozenset({ARTIFACT_AUDIO})
    produces: ClassVar[frozenset[str]] = frozenset({"audio.quiet.ratio"})
    cost: ClassVar[int] = 1

    async def analyze(
        self, context: AnalysisContext, scenes: Sequence[TimeRange]
    ) -> Mapping[int, Sequence[FeatureValue]]:
        return {0: [FeatureValue("audio.quiet.ratio", 0.5)]}


@register_analyzer
class BrokenAnalyzer:
    key: ClassVar[str] = "audio.broken"
    requires: ClassVar[frozenset[str]] = frozenset({ARTIFACT_AUDIO})
    produces: ClassVar[frozenset[str]] = frozenset({"audio.broken.value"})
    cost: ClassVar[int] = 1

    async def analyze(
        self, context: AnalysisContext, scenes: Sequence[TimeRange]
    ) -> Mapping[int, Sequence[FeatureValue]]:
        raise RuntimeError("модель не загрузилась")


@register_analyzer
class VisionAnalyzer:
    """Чужая группа: этап аудио не должен его трогать.

    Ключ намеренно выдуманный, а не ``vision.motion``: настоящий анализатор
    движения уже занимает это имя, и реестр — справедливо — не даёт двум
    разным классам зарегистрироваться под одним ключом.
    """

    key: ClassVar[str] = "vision.fake"
    requires: ClassVar[frozenset[str]] = frozenset({ARTIFACT_VIDEO})
    produces: ClassVar[frozenset[str]] = frozenset({"vision.fake.value"})
    cost: ClassVar[int] = 3

    async def analyze(
        self, context: AnalysisContext, scenes: Sequence[TimeRange]
    ) -> Mapping[int, Sequence[FeatureValue]]:
        return {index: [FeatureValue("vision.fake.value", 0.3)] for index in range(len(scenes))}


@pytest.fixture(autouse=True)
def _keep_analyzers_registered() -> None:
    """Вернуть тестовые анализаторы в реестр перед каждым тестом.

    Реестр — глобальное состояние, и общая фикстура чистит его после каждого
    теста (иначе плагины протекали бы между ними). Декораторы же исполняются
    один раз при импорте модуля, поэтому без этой фикстуры все тесты, кроме
    первого, видели бы пустой реестр.

    Повторная регистрация того же класса — не ошибка: реестр возражает
    только против **другого** класса с тем же ключом.
    """
    for cls in (LoudAnalyzer, QuietAnalyzer, BrokenAnalyzer, VisionAnalyzer):
        register_analyzer(cls)


def make_video() -> Video:
    return Video(
        id=uuid4(),
        user_id=USER_ID,
        source_uri="upload://video",
        source_kind=SourceKind.UPLOAD,
        status=VideoStatus.READY,
        storage_key="videos/x/source.mp4",
        checksum_sha256="f" * 64,
        info=make_media_info(duration_sec=120.0),
    )


def make_job(video: Video) -> Job:
    return Job(
        id=uuid4(),
        video_id=video.id,
        user_id=video.user_id,
        status=JobStatus.RUNNING,
        params_hash="hash",
    )


def make_scenes(video: Video, job: Job, count: int = 2) -> list[Scene]:
    return [
        Scene(
            id=uuid4(),
            video_id=video.id,
            job_id=job.id,
            index=index,
            range=TimeRange(index * 60.0, (index + 1) * 60.0),
        )
        for index in range(count)
    ]


def build(
    *,
    tmp_path: Path,
    disabled: frozenset[str] = frozenset(),
    scenes: list[Scene] | None = None,
    with_audio: bool = True,
) -> tuple[AnalyzeFeatures, FakeUnitOfWork, Job]:
    video = make_video()
    job = make_job(video)
    uow = FakeUnitOfWork(videos=FakeVideoRepository({video.id: video}))
    uow.jobs.jobs[job.id] = job
    uow.scenes.scenes.extend(scenes if scenes is not None else make_scenes(video, job))

    storage = FakeStorage()
    if with_audio:
        source = tmp_path / "audio.wav"
        source.write_bytes(b"RIFF" + b"\x00" * 3200)
        storage.objects[audio_key(job.id)] = source.read_bytes()

    media = LocalMedia(
        path=tmp_path / "source.mp4",
        info=make_media_info(duration_sec=120.0),
        source_uri=video.source_uri,
        owned=False,
    )
    media.path.write_bytes(b"video")

    usecase = AnalyzeFeatures(
        uow_factory=uow_factory(uow),
        storage=storage,
        provider_resolver=lambda uri: FakeSourceProvider(media),
        workdir=tmp_path / "work",
        stage=JobStage.ANALYZE_AUDIO,
        group="audio",
        needs=frozenset({ARTIFACT_AUDIO}),
        disabled=disabled,
    )
    return usecase, uow, job


class TestPlanning:
    async def test_only_own_group_runs(self, tmp_path: Path) -> None:
        """Этап аудио не трогает видеоанализаторы: у них своя очередь
        и своё железо (док 04, §4.3)."""
        usecase, uow, job = build(tmp_path=tmp_path)

        result = await usecase.execute(job.id)

        assert "vision.fake" not in result.analyzers
        scenes = await uow.scenes.list_for_job(job.id)
        assert "vision.fake.value" not in scenes[0].features

    async def test_disabled_analyzer_does_not_run(self, tmp_path: Path) -> None:
        """DoD этапа: отключение через ANALYZERS_DISABLED работает."""
        usecase, uow, job = build(tmp_path=tmp_path, disabled=frozenset({"audio.quiet"}))

        result = await usecase.execute(job.id)

        assert "audio.quiet" not in result.analyzers
        scenes = await uow.scenes.list_for_job(job.id)
        assert "audio.quiet.ratio" not in scenes[0].features
        assert "audio.loud.mean" in scenes[0].features

    async def test_empty_group_is_skipped_not_failed(self, tmp_path: Path) -> None:
        """Все анализаторы группы отключены — этап пропускается, конвейер
        едет дальше. Пустая группа не повод ронять джоб."""
        usecase, uow, job = build(
            tmp_path=tmp_path,
            disabled=frozenset({"audio.loud", "audio.quiet", "audio.broken"}),
        )

        result = await usecase.execute(job.id)

        assert result.skipped
        assert uow.jobs.jobs[job.id].stages[JobStage.ANALYZE_AUDIO].status == "skipped"


class TestFailureIsolation:
    async def test_broken_analyzer_does_not_stop_the_others(self, tmp_path: Path) -> None:
        """DoD этапа: сломанный анализатор → джоб partial, остальные
        признаки на месте."""
        usecase, uow, job = build(tmp_path=tmp_path)

        result = await usecase.execute(job.id)

        assert "audio.broken" in result.failed
        assert "модель не загрузилась" in result.failed["audio.broken"]
        assert set(result.analyzers) == {"audio.loud", "audio.quiet"}

        scenes = await uow.scenes.list_for_job(job.id)
        assert scenes[0].features["audio.loud.mean"].value == pytest.approx(-20.0)
        assert "audio.broken.value" not in scenes[0].features

    async def test_job_becomes_partial_not_failed(self, tmp_path: Path) -> None:
        usecase, uow, job = build(tmp_path=tmp_path)

        await usecase.execute(job.id)

        stored = uow.jobs.jobs[job.id]
        assert stored.status is JobStatus.PARTIAL
        assert stored.stages[JobStage.ANALYZE_AUDIO].status == "succeeded"
        assert stored.error is not None
        assert "audio.broken" in stored.error["analyzers"]

    async def test_all_good_leaves_the_job_running(self, tmp_path: Path) -> None:
        usecase, uow, job = build(tmp_path=tmp_path, disabled=frozenset({"audio.broken"}))

        result = await usecase.execute(job.id)

        assert not result.failed
        assert uow.jobs.jobs[job.id].status is JobStatus.RUNNING


class TestFeatures:
    async def test_features_are_added_not_replaced(self, tmp_path: Path) -> None:
        """Аудио, речь и видео приходят разными этапами. Замена словаря
        целиком стёрла бы результат предыдущего."""
        usecase, uow, job = build(tmp_path=tmp_path)

        # Предыдущий этап уже посчитал видеопризнак для первой сцены.
        scenes = list(await uow.scenes.list_for_job(job.id))
        await uow.scenes.bulk_update(
            [
                replace(
                    scenes[0],
                    features={"vision.motion.mean": FeatureValue("vision.motion.mean", 0.7)},
                )
            ]
        )

        await usecase.execute(job.id)

        stored = (await uow.scenes.list_for_job(job.id))[0]
        assert stored.features["vision.motion.mean"].value == pytest.approx(0.7)
        assert "audio.loud.mean" in stored.features

    async def test_analyzer_may_skip_scenes(self, tmp_path: Path) -> None:
        """Контракт порта: признаки не обязаны быть у всех сцен. Там, где
        речи нет, речевых признаков быть и не должно."""
        usecase, uow, job = build(tmp_path=tmp_path)

        await usecase.execute(job.id)

        scenes = await uow.scenes.list_for_job(job.id)
        assert "audio.quiet.ratio" in scenes[0].features
        assert "audio.quiet.ratio" not in scenes[1].features

    async def test_second_run_is_free(self, tmp_path: Path) -> None:
        usecase, _, job = build(tmp_path=tmp_path)

        await usecase.execute(job.id)
        result = await usecase.execute(job.id)

        assert result.reused
        assert result.analyzers == ()

    async def test_analysis_without_scenes_is_an_error(self, tmp_path: Path) -> None:
        """Анализировать нечего: сцены — вход этого этапа, и их отсутствие
        означает нарушенный порядок конвейера, а не пустой результат."""
        usecase, _, job = build(tmp_path=tmp_path, scenes=[])

        with pytest.raises(ValueError, match="detect_scenes"):
            await usecase.execute(job.id)

    async def test_without_audio_analyzers_are_not_planned(self, tmp_path: Path) -> None:
        """Аудио не извлекалось (видео без звука) — план пуст, этап проходит
        вхолостую, но не падает."""
        usecase, uow, job = build(tmp_path=tmp_path, with_audio=False)

        result = await usecase.execute(job.id)

        assert result.analyzers == ()
        assert not result.failed
        assert uow.jobs.jobs[job.id].status is JobStatus.RUNNING
