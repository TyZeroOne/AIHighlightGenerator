"""Этап ``select_highlights``: оценённые сцены → список клипов.

Последний этап, работающий с решениями, а не с файлами. На выходе — записи
в ``clips``: «вот этот отрезок стоит вырезать, вот почему, вот его место
в рейтинге». Резать будет Этап 8.

## Здесь же происходит слияние с оценкой модели

Эвристика посчитана до LLM-этапа, вердикты — после него. Свести их можно
только сейчас, когда в базе есть и то и другое. Поэтому `final_score`
пересчитывается перед отбором и сохраняется: он показывается в API
и объясняет порядок клипов.

## Паузы для снапа границ берутся из транскрипта

Интервалы речи, которые считал VAD на Этапе 5, в базу не попадают — там
хранятся только агрегаты по сцене (`audio.pause.max` и прочие). Зато есть
транскрипт с посегментными таймингами, а промежуток между сегментами и есть
пауза. Это дешевле повторного прогона VAD и точнее для нашей задачи:
границу клипа надо ставить между **репликами**, а не в любой тишине.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Final
from uuid import UUID

from core.domain.enums import JobStage
from core.domain.media import Transcript
from core.domain.values import TimeRange
from core.logging import get_logger
from core.ports.repository import UnitOfWorkFactory
from core.ports.scoring import HighlightSelector
from core.usecases.stages import StageTracker, params_of

__all__ = ["SelectHighlights", "SelectionResult", "pauses_from_transcript"]

log = get_logger(__name__)

#: Промежуток короче этого — не пауза, а естественный зазор между словами.
#: Ставить границу клипа туда бессмысленно: на слух это середина фразы.
_MIN_PAUSE_SEC: Final = 0.35


@dataclass(frozen=True, slots=True)
class SelectionResult:
    job_id: UUID
    clips: int
    #: Сколько сцен рассматривалось. Разрыв между этим числом и числом
    #: клипов — нормальная работа отбора, а не потеря данных.
    scenes: int
    top_score: float
    total_duration_sec: float
    duration_sec: float
    skipped: bool = False
    reused: bool = False


class SelectHighlights:
    """Сценарий этапа отбора."""

    def __init__(
        self,
        *,
        uow_factory: UnitOfWorkFactory,
        selector: HighlightSelector,
        profile: str = "default",
    ) -> None:
        self._uow_factory = uow_factory
        self._selector = selector
        self._profile = profile
        self._tracker = StageTracker(uow_factory=uow_factory, stage=JobStage.SELECT_HIGHLIGHTS)

    async def execute(
        self,
        job_id: UUID,
        *,
        total_duration_sec: float | None = None,
    ) -> SelectionResult:
        started = time.monotonic()
        job, video = await self._tracker.load(job_id)

        if job.is_stage_done(JobStage.SELECT_HIGHLIGHTS):
            return SelectionResult(job_id, 0, 0, 0.0, 0.0, 0.0, reused=True)

        async with self._uow_factory() as uow:
            scenes = list(await uow.scenes.list_for_job(job_id))
            transcript = await uow.transcripts.get_for_job(job_id)

        if not scenes:
            await self._tracker.skip(job, "нет сцен: этап детекции не выполнялся")
            return SelectionResult(job_id, 0, 0, 0.0, 0.0, 0.0, skipped=True)

        params = params_of(job, self._profile)
        await self._tracker.begin(job)

        try:
            fused = self._selector.fuse_scores(scenes)
            highlights = self._selector.select(
                fused,
                profile=params.profile,
                pauses=pauses_from_transcript(transcript),
                total_duration_sec=total_duration_sec,
            )
            clips = self._selector.to_clips(highlights, job_id=job_id, video_id=video.id)
        except Exception as error:
            await self._tracker.fail(job, error, duration_sec=_elapsed(started))
            raise

        async with self._uow_factory() as uow:
            await uow.scenes.bulk_update(fused)
            # Сначала удалить, потом вставить: повторный запуск этапа не
            # должен дописывать второй комплект клипов к первому.
            await uow.clips.delete_for_job(job_id)
            await uow.clips.bulk_add(clips)
            await uow.commit()

        await self._tracker.succeed(job, duration_sec=_elapsed(started))

        result = SelectionResult(
            job_id=job_id,
            clips=len(clips),
            scenes=len(scenes),
            top_score=round(max((clip.score for clip in clips), default=0.0), 2),
            total_duration_sec=round(sum(clip.range.duration for clip in clips), 2),
            duration_sec=_elapsed(started),
        )
        log.info(
            "select.done",
            job_id=str(job_id),
            clips=result.clips,
            top=result.top_score,
            total=result.total_duration_sec,
        )
        return result


def pauses_from_transcript(transcript: Transcript) -> list[TimeRange]:
    """Промежутки между репликами.

    Чистая функция: тестируется на выдуманном транскрипте без базы.
    Начало первой реплики и конец последней паузами не считаются — там
    границу и так ставит сцена.
    """
    segments = sorted(transcript.segments, key=lambda segment: segment.range.start)
    pauses: list[TimeRange] = []
    for previous, current in itertools.pairwise(segments):
        gap = current.range.start - previous.range.end
        if gap >= _MIN_PAUSE_SEC:
            pauses.append(TimeRange(previous.range.end, current.range.start))
    return pauses


def _elapsed(started: float) -> float:
    return round(time.monotonic() - started, 3)
