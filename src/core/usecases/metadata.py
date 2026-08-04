"""Этап ``generate_metadata``: клипы получают заголовок, описание и хештеги.

Второй и последний этап, зависящий от внешней платной модели. Отсюда —
та же линия поведения, что и у ранжирования (док 06, §6.10): его
недоступность не проваливает джоб.

## Почему это отдельный этап, а не часть нарезки

Соблазн понятен: клип уже собран, метаданные к нему — та же работа над
тем же объектом. Но у них разная природа отказа и разная цена повтора.
Нарезка упирается в процессор и повторяется бесплатно; метаданные
упираются в чужой сервис и повторяются за деньги. Слив их в один этап,
мы получили бы повторный рендер десяти клипов из-за того, что модель
ответила пятисотым кодом на одиннадцатом.

## Что происходит, когда модели нет

Клипы остаются без заголовков. Это ухудшение результата, а не его потеря:
файлы нарезаны, скоры посчитаны, `reason` от ранжирования на месте.
Этап помечается пройденным с признаком деградации, и джоб идёт дальше.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from uuid import UUID

from core.domain.entities import Clip, ClipMetadata, Job
from core.domain.enums import ClipStatus, JobStage
from core.domain.errors import ExternalServiceError
from core.logging import get_logger
from core.ports.llm import ClipAnnotator
from core.ports.repository import UnitOfWorkFactory
from core.usecases.stages import StageTracker

__all__ = ["GenerateMetadata", "MetadataResult", "merge_cost"]

log = get_logger(__name__)

STAGE = JobStage.GENERATE_METADATA

#: Фабрика аннотатора, а не готовый объект: провайдер копит расход внутри
#: себя, и каждому джобу нужен свой экземпляр — иначе стоимость второго
#: джоба включила бы токены первого. Тот же приём, что на Этапе 6.
type AnnotatorFactory = Callable[[], ClipAnnotator]


@dataclass(frozen=True, slots=True)
class MetadataResult:
    job_id: UUID
    clips_total: int
    annotated: int
    failed: int
    duration_sec: float
    cost: Mapping[str, object] = field(default_factory=dict)
    skipped: bool = False
    reused: bool = False
    #: Модель была недоступна — клипы остались без заголовков.
    degraded: bool = False


class GenerateMetadata:
    """Сценарий этапа метаданных."""

    def __init__(
        self,
        *,
        uow_factory: UnitOfWorkFactory,
        annotator_factory: AnnotatorFactory,
        enabled: bool = True,
    ) -> None:
        self._uow_factory = uow_factory
        self._annotator_factory = annotator_factory
        self._enabled = enabled
        self._tracker = StageTracker(uow_factory=uow_factory, stage=STAGE)

    async def execute(self, job_id: UUID) -> MetadataResult:
        started = time.monotonic()
        job, _ = await self._tracker.load(job_id)

        if job.is_stage_done(STAGE):
            return MetadataResult(job_id, 0, 0, 0, 0.0, reused=True)

        if not self._enabled:
            await self._tracker.skip(job, "генерация метаданных выключена (LLM_ENABLED=false)")
            return MetadataResult(job_id, 0, 0, 0, 0.0, skipped=True)

        async with self._uow_factory() as uow:
            clips = list(await uow.clips.list_for_job(job_id))
            transcript = await uow.transcripts.get_for_job(job_id)

        # Клипы, которые не собрались, метаданных не получают: заголовок
        # к несуществующему файлу — это строка, которую некуда показать.
        targets = [clip for clip in clips if clip.status is not ClipStatus.FAILED]
        if not targets:
            await self._tracker.skip(job, "нет клипов: этап отбора не выполнялся")
            return MetadataResult(job_id, len(clips), 0, 0, 0.0, skipped=True)

        await self._tracker.begin(job)

        try:
            outcome = await self._annotator_factory().annotate(
                targets, transcript, transcript.language
            )
        except ExternalServiceError as error:
            return await self._degrade(job, error, len(targets), started)
        except Exception as error:
            await self._tracker.fail(job, error, duration_sec=_elapsed(started))
            raise

        annotated = await self._store(job_id, targets, outcome.metadata)
        await self._tracker.succeed(job, duration_sec=_elapsed(started))

        # Строго после отметки этапа: отметка пишет объект джоба, прочитанный
        # в начале сценария, и запись расхода до неё была бы затёрта
        # (см. разбор Этапа 6 — классический lost update).
        if outcome.cost:
            await self._write_cost(job_id, outcome.cost)

        result = MetadataResult(
            job_id=job_id,
            clips_total=len(clips),
            annotated=annotated,
            failed=len(outcome.failed),
            duration_sec=_elapsed(started),
            cost=dict(outcome.cost),
        )
        log.info(
            "metadata.done",
            job_id=str(job_id),
            annotated=result.annotated,
            failed=result.failed,
            duration_sec=result.duration_sec,
        )
        return result

    # --------------------------------------------------------------- шаги
    async def _store(
        self,
        job_id: UUID,
        clips: Sequence[Clip],
        metadata: Mapping[UUID, ClipMetadata],
    ) -> int:
        updated = [
            replace(clip, metadata=metadata[clip.id]) for clip in clips if clip.id in metadata
        ]
        if not updated:
            return 0

        async with self._uow_factory() as uow:
            for clip in updated:
                await uow.clips.update(clip)
            await uow.commit()
        log.info("metadata.stored", job_id=str(job_id), clips=len(updated))
        return len(updated)

    async def _degrade(
        self,
        job: Job,
        error: ExternalServiceError,
        requested: int,
        started: float,
    ) -> MetadataResult:
        """Модель недоступна — этап пройден, клипы без заголовков.

        ``succeed``, а не ``fail``: провал этапа остановил бы джоб, хотя
        всё, чего не хватает, — это подписи к готовым файлам.
        """
        log.warning("metadata.degraded", job_id=str(job.id), error=str(error))
        await self._tracker.succeed(job, duration_sec=_elapsed(started))
        return MetadataResult(
            job_id=job.id,
            clips_total=requested,
            annotated=0,
            failed=requested,
            duration_sec=_elapsed(started),
            degraded=True,
        )

    async def _write_cost(self, job_id: UUID, cost: Mapping[str, object]) -> None:
        """Дописать расход этапа к тому, что уже накопил джоб.

        **Под тот же ключ ``llm`` и суммой**, а не рядом. Первый живой
        прогон записал расход плоскими ключами, и в ``jobs.cost`` оказалось
        два набора чисел одновременно: вложенный от ранжирования и плоский
        отсюда. По такому полю на вопрос «сколько стоило это видео»
        не ответить — а он единственная причина, по которой поле есть.
        """
        async with self._uow_factory() as uow:
            current = await uow.jobs.get(job_id)
            if current is None:  # pragma: no cover — джоб прочитан выше
                return
            await uow.jobs.update(replace(current, cost=merge_cost(current.cost, cost)))
            await uow.commit()


def merge_cost(
    current: Mapping[str, object],
    addition: Mapping[str, object],
    *,
    key: str = "llm",
) -> dict[str, object]:
    """Сложить расход двух этапов под одним ключом.

    Числа складываются, всё остальное заменяется. Чистая функция:
    правило «сколько всего потратил джоб» проверяется без базы.
    """
    merged = dict(current)
    previous = merged.get(key)
    totals: dict[str, object] = dict(previous) if isinstance(previous, Mapping) else {}

    for name, value in addition.items():
        before = totals.get(name)
        if isinstance(before, int | float) and isinstance(value, int | float):
            totals[name] = before + value
        elif isinstance(before, list) and isinstance(value, list):
            totals[name] = [*before, *value]
        else:
            totals[name] = value

    merged[key] = totals
    return merged


def _elapsed(started: float) -> float:
    return round(time.monotonic() - started, 3)
