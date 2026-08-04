"""Этапы анализа признаков — общий сценарий для аудио, речи и видео.

Один класс на три этапа конвейера, а не три похожих: отличаются они только
тем, какие анализаторы брать и какое сырьё для этого нужно. Всё остальное —
сборка плана, параллельное исполнение слоёв, изоляция сбоев, запись
признаков — совпадает дословно.

Ключевое свойство этапа, ради которого затевался реестр плагинов: **сценарий
не знает ни одного анализатора по имени**. Он спрашивает реестр, что
зарегистрировано, спрашивает планировщик, в каком порядке это запускать,
и складывает результат. Новый анализатор добавляется отдельным файлом
с декоратором — здесь не меняется ничего.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import UUID

from core import registry
from core.domain.analysis import (
    ARTIFACT_AUDIO,
    ARTIFACT_TRANSCRIPT,
    ARTIFACT_VIDEO,
    AnalysisContext,
)
from core.domain.entities import Job, Scene, Video
from core.domain.enums import JobStage, JobStatus
from core.domain.keys import audio_key
from core.domain.media import AudioTrack, LocalMedia
from core.domain.values import FeatureValue, MediaWindow, TimeRange
from core.logging import get_logger
from core.ports.analysis import FeatureAnalyzer
from core.ports.repository import UnitOfWorkFactory
from core.ports.source import SourceProvider
from core.ports.storage import ObjectStorage
from core.usecases.stages import StageTracker

__all__ = ["AnalysisResult", "AnalyzeFeatures"]

log = get_logger(__name__)

type ProviderResolver = Callable[[str], SourceProvider]


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    job_id: UUID
    stage: JobStage
    analyzers: tuple[str, ...]
    #: Анализаторы, упавшие с исключением. Джоб при этом ``partial``,
    #: а не провалившийся: остальные признаки посчитаны и годны.
    failed: Mapping[str, str]
    feature_count: int
    scenes_with_features: int
    duration_sec: float
    skipped: bool = False
    reused: bool = False


class AnalyzeFeatures:
    """Этап анализа: признаки одной группы по всем сценам джоба.

    Группа — это префикс ключей анализаторов (``audio``, ``speech``,
    ``vision``), она же определяет, какое сырьё готовить. Разделение
    по группам совпадает с этапами из док 04 §4.1 и нужно ради очередей:
    видеоанализ идёт на GPU, аудио — на CPU, и ждать друг друга им незачем.
    """

    def __init__(
        self,
        *,
        uow_factory: UnitOfWorkFactory,
        storage: ObjectStorage,
        provider_resolver: ProviderResolver,
        workdir: Path,
        stage: JobStage,
        group: str,
        needs: frozenset[str],
        disabled: frozenset[str] = frozenset(),
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        self._uow_factory = uow_factory
        self._storage = storage
        self._resolve_provider = provider_resolver
        self._workdir = workdir
        self._stage = stage
        self._group = group
        self._needs = needs
        self._disabled = disabled
        self._sample_rate = sample_rate
        self._channels = channels
        self._tracker = StageTracker(uow_factory=uow_factory, stage=stage)

    async def execute(self, job_id: UUID) -> AnalysisResult:
        started = time.monotonic()
        job, video = await self._tracker.load(job_id)

        if job.is_stage_done(self._stage):
            return self._empty(job_id, duration=0.0, reused=True)

        analyzers = self._select()
        if not analyzers:
            # Группа пуста: анализаторы не зарегистрированы или все отключены
            # через ANALYZERS_DISABLED. Это не ошибка — конвейер обязан
            # доехать до конца и без них (док 04, §4.5).
            await self._tracker.skip(job, f"нет анализаторов группы {self._group!r}")
            return self._empty(job_id, duration=0.0, skipped=True)

        await self._tracker.begin(job)
        try:
            result = await self._run(job, video, analyzers, started)
        except Exception as error:
            await self._tracker.fail(job, error, duration_sec=round(time.monotonic() - started, 3))
            raise

        await self._tracker.succeed(job, duration_sec=result.duration_sec)
        if result.failed:
            # Джоб «частично выполнен»: результат есть, но неполный.
            # Скоринг переживёт (док 05, §5.4 — вес отсутствующей группы
            # перераспределяется), а вот молча это скрывать нельзя.
            await self._mark_partial(job_id, result.failed)
        return result

    # ------------------------------------------------------------- работа
    def _select(self) -> list[type[FeatureAnalyzer]]:
        """Анализаторы своей группы, кроме отключённых конфигом."""
        prefix = f"{self._group}."
        return [
            cls
            for key, cls in sorted(registry.get_analyzers().items())
            if key.startswith(prefix) and key not in self._disabled
        ]

    async def _run(
        self,
        job: Job,
        video: Video,
        analyzers: Sequence[type[FeatureAnalyzer]],
        started: float,
    ) -> AnalysisResult:
        async with self._uow_factory() as uow:
            scenes = list(await uow.scenes.list_for_job(job.id))
            transcript = (
                await uow.transcripts.get_for_job(job.id)
                if ARTIFACT_TRANSCRIPT in self._needs
                else None
            )

        if not scenes:
            raise ValueError("Нет сцен: этап detect_scenes должен идти раньше анализа")

        ranges = [scene.range for scene in scenes]
        provider = self._resolve_provider(video.source_uri)
        media: LocalMedia | None = None
        audio: AudioTrack | None = None

        try:
            if ARTIFACT_VIDEO in self._needs:
                media = await provider.acquire(video.source_uri)
            audio = await self._fetch_audio(job.id) if ARTIFACT_AUDIO in self._needs else None

            context = AnalysisContext(
                job_id=job.id,
                video_id=video.id,
                window=MediaWindow(
                    source_id=video.source_uri,
                    range=TimeRange(0.0, video.info.duration_sec if video.info else ranges[-1].end),
                ),
                workdir=self._workdir / "analysis" / str(job.id),
                media=media,
                audio=audio,
                transcript=transcript,
            )
            context.workdir.mkdir(parents=True, exist_ok=True)

            features, failed, executed = await self._execute_plan(analyzers, context, ranges)
        finally:
            if media is not None:
                await provider.release(media)
            if audio is not None:
                audio.path.unlink(missing_ok=True)

        stored = await self._store(scenes, features)
        duration = round(time.monotonic() - started, 3)
        log.info(
            "analysis.done",
            job_id=str(job.id),
            stage=self._stage.value,
            analyzers=executed,
            failed=sorted(failed),
            features=sum(len(values) for values in features.values()),
            duration_sec=duration,
        )
        return AnalysisResult(
            job_id=job.id,
            stage=self._stage,
            analyzers=tuple(executed),
            failed=failed,
            feature_count=sum(len(values) for values in features.values()),
            scenes_with_features=stored,
            duration_sec=duration,
        )

    async def _execute_plan(
        self,
        analyzers: Sequence[type[FeatureAnalyzer]],
        context: AnalysisContext,
        ranges: Sequence[TimeRange],
    ) -> tuple[dict[int, dict[str, FeatureValue]], dict[str, str], list[str]]:
        """Прогнать анализаторы по слоям, не позволяя одному уронить остальных.

        Внутри слоя анализаторы независимы и идут параллельно; слои —
        последовательно, потому что следующий может зависеть от результатов
        предыдущего (``requires``/``produces``).
        """
        available = context.available_artifacts()
        runnable, starving = _split_by_artifacts(analyzers, available)
        if starving:
            # Сырья нет — например, у видео не было звука. Это факт про
            # конкретное видео, а не ошибка конфигурации, и падать здесь
            # нельзя: конвейер обязан доехать до конца по тому, что есть.
            log.info(
                "analyzers.skipped_no_artifacts",
                analyzers=sorted(cls.key for cls in starving),
                available=sorted(available),
            )
        if not runnable:
            return {}, {}, []

        plan = registry.build_plan(
            [cls.key for cls in runnable],
            available_artifacts=available,
            disabled=self._disabled,
        )

        features: dict[int, dict[str, FeatureValue]] = {}
        failed: dict[str, str] = {}
        executed: list[str] = []

        for layer in plan:
            outcomes = await asyncio.gather(
                *(self._run_one(cls, context, ranges) for cls in layer),
                return_exceptions=True,
            )
            for cls, outcome in zip(layer, outcomes, strict=True):
                if isinstance(outcome, BaseException):
                    failed[cls.key] = f"{type(outcome).__name__}: {outcome}"
                    log.error("analyzer.failed", analyzer=cls.key, error=str(outcome))
                    continue
                executed.append(cls.key)
                _merge(features, outcome)

        return features, failed, executed

    @staticmethod
    async def _run_one(
        cls: type[FeatureAnalyzer],
        context: AnalysisContext,
        ranges: Sequence[TimeRange],
    ) -> Mapping[int, Sequence[FeatureValue]]:
        # Экземпляр на каждый запуск: анализатор вправе держать внутри
        # состояние (загруженную модель, буферы), и делить его между джобами
        # было бы источником трудноуловимых зависимостей между прогонами.
        analyzer = cls()
        return await analyzer.analyze(context, ranges)

    async def _fetch_audio(self, job_id: UUID) -> AudioTrack | None:
        """Достать WAV, извлечённый на Этапе 4.

        Отсутствие аудио — не ошибка: у видео могло не быть звука.
        Анализаторы, которым оно нужно, просто не попадут в план.
        """
        key = audio_key(job_id)
        if not await self._storage.exists(key):
            return None

        destination = self._workdir / "analysis" / f"{job_id}.wav"
        destination.parent.mkdir(parents=True, exist_ok=True)
        await self._storage.get(key, destination)
        size = destination.stat().st_size
        return AudioTrack(
            path=destination,
            sample_rate=self._sample_rate,
            channels=self._channels,
            duration_sec=max(size - 44, 0) / (self._sample_rate * self._channels * 2),
        )

    async def _store(
        self,
        scenes: Sequence[Scene],
        features: Mapping[int, Mapping[str, FeatureValue]],
    ) -> int:
        """Дописать признаки к сценам.

        Именно дописать: аудио, речь и видео считаются разными этапами
        и приходят сюда по очереди. Замена словаря целиком стёрла бы
        результат предыдущего этапа.
        """
        updated = [
            replace(scene, features={**scene.features, **features[index]})
            for index, scene in enumerate(scenes)
            if features.get(index)
        ]
        if not updated:
            return 0

        async with self._uow_factory() as uow:
            await uow.scenes.bulk_update(updated)
            await uow.commit()
        return len(updated)

    async def _mark_partial(self, job_id: UUID, failed: Mapping[str, str]) -> None:
        async with self._uow_factory() as uow:
            job = await uow.jobs.get(job_id)
            if job is None:  # pragma: no cover — джоб только что читался
                return
            await uow.jobs.update(
                replace(
                    job,
                    status=JobStatus.PARTIAL,
                    error={
                        "stage": self._stage.value,
                        "type": "AnalyzerFailure",
                        "message": "часть анализаторов не отработала",
                        "analyzers": dict(failed),
                    },
                )
            )
            await uow.commit()

    def _empty(
        self,
        job_id: UUID,
        *,
        duration: float,
        skipped: bool = False,
        reused: bool = False,
    ) -> AnalysisResult:
        return AnalysisResult(
            job_id=job_id,
            stage=self._stage,
            analyzers=(),
            failed={},
            feature_count=0,
            scenes_with_features=0,
            duration_sec=duration,
            skipped=skipped,
            reused=reused,
        )


def _split_by_artifacts(
    analyzers: Sequence[type[FeatureAnalyzer]],
    available: frozenset[str],
) -> tuple[list[type[FeatureAnalyzer]], list[type[FeatureAnalyzer]]]:
    """Разделить анализаторы на «есть чем работать» и «сырья нет».

    Замыкание, а не однократная проверка: анализатор может требовать то,
    что производит другой анализатор той же группы, — тогда он тоже
    запустим, но только после него. Порядок здесь не важен, им займётся
    ``build_plan``; здесь решается лишь вопрос принципиальной возможности.
    """
    satisfied = set(available)
    runnable: list[type[FeatureAnalyzer]] = []
    pending = list(analyzers)

    progress = True
    while progress:
        progress = False
        for cls in list(pending):
            if set(cls.requires).issubset(satisfied):
                runnable.append(cls)
                pending.remove(cls)
                satisfied |= set(cls.produces)
                progress = True

    return runnable, pending


def _merge(
    target: dict[int, dict[str, FeatureValue]],
    produced: Mapping[int, Sequence[FeatureValue]],
) -> None:
    for index, values in produced.items():
        bucket = target.setdefault(index, {})
        for value in values:
            bucket[value.key] = value
