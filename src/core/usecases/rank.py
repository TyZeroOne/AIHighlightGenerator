"""Этап ``llm_rank``: качественная оценка сцен языковой моделью.

Этапы 3–5 описали сцены числами. Числа отвечают на вопрос «что происходит
в кадре», но не на вопрос «интересно ли это»: громкая сцена с быстрым
монтажом может быть проходной перебивкой, а тихая пауза перед признанием —
лучшим моментом видео. На этот вопрос отвечает модель, и только она.

## Главное свойство этапа: он обязан быть необязательным

LLM — единственная часть конвейера, зависящая от внешнего сервиса, за деньги
и без гарантий доступности. Поэтому правило жёсткое: **недоступность модели
не проваливает джоб**. Сервис лёг, кончилась квота, модель ответила мусором —
сцены остаются с эвристикой, джоб получает статус ``partial`` и доезжает
до клипов.

Обратное — «упасть, раз не смогли спросить модель» — означало бы, что
двухчасовая обработка выбрасывается из-за пятисекундного сбоя чужого API.

## Что этот сценарий знает и чего не знает

Знает: когда этап уместен, куда писать вердикты, что делать при сбое.
Не знает: как выглядит промпт, сколько сцен в батче, какой моделью считаем
и сколько стоит токен. Всё это — за портом ``SceneRanker``.

## Что записывается

В каждую сцену: ``llm_score`` (0..100), ``category`` и ``reason``. Границы
клипа при этом остаются **нашими**: тайм-коды из ответа модели — подсказка
для Этапа 7, но не источник правды.

В джоб: расход (``jobs.cost``). Вопрос «во сколько обошлось это видео»
задают не в момент обработки, а через месяц, когда логи уже ротировались.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from uuid import UUID

from core.domain.analysis import LLMVerdict, VideoContext
from core.domain.entities import Job, Scene
from core.domain.enums import JobStage, JobStatus
from core.domain.errors import ExternalServiceError
from core.logging import get_logger
from core.ports.llm import SceneRanker
from core.ports.repository import UnitOfWorkFactory
from core.usecases.stages import StageTracker

__all__ = ["RankScenes", "RankingResult"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RankingResult:
    job_id: UUID
    scenes_total: int
    #: Сколько сцен ушло в модель после фильтра. Разница с общим числом —
    #: и есть экономия: сцены без речи и без единого заметного сигнала
    #: заведомо не хайлайты, а платим мы за каждую.
    scenes_sent: int
    scenes_scored: int
    batches: int
    duration_sec: float
    cost: Mapping[str, object]
    skipped: bool = False
    reused: bool = False
    #: Модель не ответила, конвейер идёт по эвристике.
    degraded: bool = False
    failed: tuple[str, ...] = ()


class RankScenes:
    """Сценарий этапа ранжирования."""

    def __init__(
        self,
        *,
        uow_factory: UnitOfWorkFactory,
        ranker_factory: Callable[[], SceneRanker],
        enabled: bool = True,
        profile: str = "default",
    ) -> None:
        self._uow_factory = uow_factory
        # Фабрика, а не готовый ранжировщик: его создание требует ключа
        # провайдера, и джоб с выключенным LLM не должен падать на том,
        # что ключа нет.
        self._ranker_factory = ranker_factory
        self._enabled = enabled
        self._profile = profile
        self._tracker = StageTracker(uow_factory=uow_factory, stage=JobStage.LLM_RANK)

    async def execute(self, job_id: UUID) -> RankingResult:
        started = time.monotonic()
        job, video = await self._tracker.load(job_id)

        if job.is_stage_done(JobStage.LLM_RANK):
            return self._empty(job_id, reused=True)

        if not self._enabled:
            await self._tracker.skip(job, "LLM выключен (LLM_ENABLED=false)")
            return self._empty(job_id, skipped=True)

        async with self._uow_factory() as uow:
            scenes = list(await uow.scenes.list_for_job(job_id))

        if not scenes:
            await self._tracker.skip(job, "нет сцен: этап детекции не выполнялся")
            return self._empty(job_id, skipped=True)

        await self._tracker.begin(job)

        context = VideoContext(
            title=video.original_filename,
            duration_sec=video.info.duration_sec if video.info else 0.0,
            language=_language_of(scenes),
            profile=self._profile,
            scene_count=len(scenes),
        )

        try:
            outcome = await self._ranker_factory().rank(scenes, context)
        except ExternalServiceError as error:
            # Единственная ветка, ради которой написан этот сценарий:
            # внешний сервис недоступен — джоб продолжается по эвристике.
            return await self._degrade(job, error, scenes, started)
        except Exception as error:
            await self._tracker.fail(job, error, duration_sec=_elapsed(started))
            raise

        scored = await self._store(job_id, scenes, outcome.verdicts)
        await self._tracker.succeed(job, duration_sec=_elapsed(started))

        # Строго после отметки этапа, и это не вкус. Трекер пишет джоб,
        # прочитанный в начале сценария: запиши мы расход раньше, отметка
        # этапа затёрла бы его пустым значением. Классический lost update —
        # найден на живом прогоне, когда вердикты сохранились, а jobs.cost
        # остался `{}`.
        if outcome.cost:
            await self._write_cost(job_id, outcome.cost)

        if outcome.failed:
            await self._mark_partial(job_id, outcome.failed)

        result = RankingResult(
            job_id=job_id,
            scenes_total=len(scenes),
            scenes_sent=outcome.scenes_sent,
            scenes_scored=scored,
            batches=outcome.batches,
            duration_sec=_elapsed(started),
            cost=dict(outcome.cost),
            degraded=outcome.all_failed,
            failed=tuple(outcome.failed),
        )
        log.info(
            "llm.rank.done",
            job_id=str(job_id),
            sent=result.scenes_sent,
            scored=result.scenes_scored,
            batches=result.batches,
            cost=dict(outcome.cost),
        )
        return result

    # ------------------------------------------------------------ внутреннее
    async def _store(
        self,
        job_id: UUID,
        scenes: Sequence[Scene],
        verdicts: Sequence[LLMVerdict],
    ) -> int:
        """Вердикты → сцены.

        Обновляются только оценённые сцены. Остальные остаются как были,
        и отсутствие ``llm_score`` означает для скоринга честное «модель
        эту сцену не смотрела», а не «оценила в ноль».
        """
        if not verdicts:
            return 0

        by_index = {scene.index: scene for scene in scenes}
        updated: list[Scene] = []
        for verdict in verdicts:
            scene = by_index.get(verdict.scene_index)
            if scene is None:
                continue
            updated.append(
                replace(
                    scene,
                    llm_score=verdict.score,
                    category=verdict.category,
                    reason=verdict.reason,
                )
            )

        if updated:
            async with self._uow_factory() as uow:
                await uow.scenes.bulk_update(updated)
                await uow.commit()
        return len(updated)

    async def _write_cost(self, job_id: UUID, cost: Mapping[str, object]) -> None:
        async with self._uow_factory() as uow:
            job = await uow.jobs.get(job_id)
            if job is None:  # pragma: no cover — джоб только что читался
                return
            await uow.jobs.update(replace(job, cost={**dict(job.cost), "llm": dict(cost)}))
            await uow.commit()

    async def _degrade(
        self,
        job: Job,
        error: Exception,
        scenes: Sequence[Scene],
        started: float,
    ) -> RankingResult:
        """Модель недоступна — этап закрывается, джоб живёт дальше.

        Этап отмечается успешным, а не провалившимся, намеренно: прогресс
        должен доехать до конца. Факт деградации виден в статусе джоба
        (``partial``) и в его ``error``.
        """
        reason = f"{type(error).__name__}: {error}"
        log.warning("llm.rank.degraded", job_id=str(job.id), error=reason)

        await self._tracker.succeed(job, duration_sec=_elapsed(started))
        await self._mark_partial(job.id, [reason])

        return RankingResult(
            job_id=job.id,
            scenes_total=len(scenes),
            scenes_sent=0,
            scenes_scored=0,
            batches=0,
            duration_sec=_elapsed(started),
            cost={},
            degraded=True,
            failed=(reason,),
        )

    async def _mark_partial(self, job_id: UUID, errors: Sequence[str]) -> None:
        """Джоб «частично выполнен»: результат есть, но неполный."""
        async with self._uow_factory() as uow:
            job = await uow.jobs.get(job_id)
            if job is None:  # pragma: no cover
                return
            await uow.jobs.update(
                replace(
                    job,
                    status=JobStatus.PARTIAL,
                    error={"stage": JobStage.LLM_RANK.value, "llm": list(errors)},
                )
            )
            await uow.commit()

    def _empty(
        self,
        job_id: UUID,
        *,
        skipped: bool = False,
        reused: bool = False,
    ) -> RankingResult:
        return RankingResult(
            job_id=job_id,
            scenes_total=0,
            scenes_sent=0,
            scenes_scored=0,
            batches=0,
            duration_sec=0.0,
            cost={},
            skipped=skipped,
            reused=reused,
        )


def _language_of(scenes: Sequence[Scene]) -> str | None:
    """Язык видео — по первой сцене с текстом.

    Определение грубое, по алфавиту: язык уходит в промпт подсказкой,
    а саму речь модель и так видит. Полноценный определитель здесь был бы
    зависимостью ради подсказки.
    """
    for scene in scenes:
        text = (scene.transcript_text or "").strip()
        if text:
            return "ru" if any("а" <= char.lower() <= "я" for char in text) else "en"
    return None


def _elapsed(started: float) -> float:
    return round(time.monotonic() - started, 3)
