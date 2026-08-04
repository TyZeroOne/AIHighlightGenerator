"""Этап ``score_scenes``: признаки → сигналы → эвристический скор.

Стоит **до** LLM-этапа, и это не случайность расстановки. Модель получает
не сырые числа, а посчитанные здесь сигналы (док 06, §6.4): так промпт
короче и понятнее. Значит сигналы должны существовать раньше, чем модель
их увидит.

Отсюда же следует, что итоговый скор здесь ещё не считается: оценки модели
на этот момент нет. `final_score` на этом этапе равен эвристике, а слияние
происходит на этапе отбора, когда вердикты уже в базе.

## Порядок вычислений

1. **Нормализация** — все признаки приводятся к 0..1 по этому видео.
2. **Структура** — новизна и нарастание: их видно только из соседства сцен.
3. **Сигналы** — десять понятий из ТЗ поверх нормализованных признаков.
4. **Скор** — взвешенная сумма по группам с перенормировкой.

Порядок жёсткий: структура считается по нормализованным признакам (иначе
расстояние между сценами определялось бы единицами измерения), а сигналы —
по нормализованным и структурным вместе.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from uuid import UUID

from core.domain.enums import JobStage
from core.logging import get_logger
from core.ports.repository import UnitOfWorkFactory
from core.ports.scoring import SceneScorer
from core.usecases.stages import StageTracker, params_of

__all__ = ["ScoreScenes", "ScoringResult"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ScoringResult:
    job_id: UUID
    scenes_scored: int
    #: Средний и максимальный скор — не украшение отчёта: если весь фильм
    #: получил 12 баллов, дело почти наверняка в отключённой группе
    #: анализаторов, а не в скучном материале.
    mean_score: float
    max_score: float
    duration_sec: float
    skipped: bool = False
    reused: bool = False


class ScoreScenes:
    """Сценарий этапа эвристического скоринга."""

    def __init__(
        self,
        *,
        uow_factory: UnitOfWorkFactory,
        scorer: SceneScorer,
        profile: str = "default",
    ) -> None:
        self._uow_factory = uow_factory
        self._scorer = scorer
        self._profile = profile
        self._tracker = StageTracker(uow_factory=uow_factory, stage=JobStage.SCORE_SCENES)

    async def execute(self, job_id: UUID) -> ScoringResult:
        started = time.monotonic()
        job, _ = await self._tracker.load(job_id)

        if job.is_stage_done(JobStage.SCORE_SCENES):
            return ScoringResult(job_id, 0, 0.0, 0.0, 0.0, reused=True)

        async with self._uow_factory() as uow:
            scenes = list(await uow.scenes.list_for_job(job_id))

        if not scenes:
            await self._tracker.skip(job, "нет сцен: этап детекции не выполнялся")
            return ScoringResult(job_id, 0, 0.0, 0.0, 0.0, skipped=True)

        params = params_of(job, self._profile)
        await self._tracker.begin(job)

        try:
            scored = self._scorer.score(scenes, profile=params.profile)
        except Exception as error:
            await self._tracker.fail(job, error, duration_sec=_elapsed(started))
            raise

        async with self._uow_factory() as uow:
            await uow.scenes.bulk_update(scored)
            await uow.commit()

        await self._tracker.succeed(job, duration_sec=_elapsed(started))

        values = [scene.heuristic_score.value for scene in scored if scene.heuristic_score]
        result = ScoringResult(
            job_id=job_id,
            scenes_scored=len(scored),
            mean_score=round(sum(values) / len(values), 2) if values else 0.0,
            max_score=round(max(values), 2) if values else 0.0,
            duration_sec=_elapsed(started),
        )
        log.info(
            "score.done",
            job_id=str(job_id),
            scenes=result.scenes_scored,
            mean=result.mean_score,
            max=result.max_score,
        )
        return result


def _elapsed(started: float) -> float:
    return round(time.monotonic() - started, 3)
