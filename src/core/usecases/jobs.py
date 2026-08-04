"""Сценарии жизненного цикла джоба: запуск, статус, повтор, отмена.

Всё, что до Этапа 10 делалось командной строкой, здесь становится
операциями предметной области — и потому доступно и API, и CLI, и любому
будущему интерфейсу без дублирования.

## Идемпотентность запуска

Повторный запрос на то же видео с теми же параметрами возвращает уже
посчитанный джоб, а не запускает сорок минут работы заново. Сравнение
идёт по отпечатку параметров (``AnalysisParams.fingerprint``), а не
по словарю: порядок ключей и явно переданные значения по умолчанию
не должны влиять на ответ.

Форсировать пересчёт можно (``force=True``) — но это осознанное решение
пользователя, а не поведение по умолчанию.

## Повтор с этапа

Перезапуск с середины конвейера обязан не пересчитывать то, что уже
посчитано: разбор кадров двухчасового фильма стоит двадцать минут,
и повторять его из-за отвалившейся модели незачем.

Механизм простой и опирается на уже существующую идемпотентность этапов:
состояния этапов **начиная с указанного** стираются, остальные остаются
``succeeded``. Задача, увидев пройденный этап, немедленно выходит.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from uuid import UUID, uuid4

from core.domain.entities import Job, StageState, User, Video, compute_progress
from core.domain.enums import STAGE_ORDER, JobStage, JobStatus
from core.domain.errors import NotFoundError, QuotaExceededError, ValidationError
from core.domain.params import AnalysisParams
from core.logging import get_logger
from core.ports.repository import UnitOfWork, UnitOfWorkFactory
from core.ports.tasks import PipelineDispatcher

__all__ = [
    "CancelJob",
    "JobView",
    "ListJobs",
    "RetryJob",
    "StartAnalysis",
    "clear_from_stage",
]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class JobView:
    """Джоб вместе с тем, что о нём нужно знать снаружи."""

    job: Job
    video: Video | None = None
    scenes: int = 0
    clips: int = 0
    #: ``True`` — джоб вернули как уже посчитанный, работу не запускали.
    reused: bool = False


class StartAnalysis:
    """Создать джоб и поставить конвейер в очередь."""

    def __init__(
        self,
        *,
        uow_factory: UnitOfWorkFactory,
        dispatcher: PipelineDispatcher,
    ) -> None:
        self._uow_factory = uow_factory
        self._dispatcher = dispatcher

    async def execute(
        self,
        *,
        user_id: UUID,
        video_id: UUID,
        params: AnalysisParams,
        force: bool = False,
    ) -> JobView:
        fingerprint = params.fingerprint()

        async with self._uow_factory() as uow:
            video = await uow.videos.get_owned(video_id, user_id)
            # 404, а не 403: существование чужого видео — тоже информация.
            if video is None:
                raise NotFoundError("Видео не найдено", video_id=str(video_id))
            if not video.is_usable:
                raise ValidationError(
                    f"Видео недоступно для анализа: статус {video.status.value}",
                    video_id=str(video_id),
                    status=video.status.value,
                )

            if not force:
                existing = await uow.jobs.find_succeeded(video_id, fingerprint)
                if existing is not None:
                    log.info("job.reused", job_id=str(existing.id), video_id=str(video_id))
                    return JobView(job=existing, video=video, reused=True)

            # Квота проверяется после идемпотентности: повторный запрос
            # на уже посчитанный джоб ничего не потребляет, и отказывать
            # в нём из-за исчерпанной квоты было бы бессмысленно жестоко.
            user = await uow.users.get(user_id)
            if user is not None:
                await _check_quota(uow, user, video)

            job = await uow.jobs.add(
                Job(
                    id=uuid4(),
                    video_id=video_id,
                    user_id=user_id,
                    status=JobStatus.QUEUED,
                    params=params.as_dict(),
                    params_hash=fingerprint,
                )
            )
            await uow.commit()

        # Постановка в очередь — ПОСЛЕ коммита. Иначе воркер успевает
        # взять задачу и не найти джоб: в брокере сообщение уже есть,
        # а в базе транзакция ещё не закрыта.
        root_id = self._dispatcher.start(job.id)
        if root_id:
            # Идентификатор корневой задачи нужен, чтобы джоб потом можно
            # было отменить, — и хранить его надо в базе: отменять будет,
            # скорее всего, другая реплика API.
            job = replace(job, celery_root_id=root_id)
            async with self._uow_factory() as uow:
                await uow.jobs.update(job)
                await uow.commit()

        log.info(
            "job.started",
            job_id=str(job.id),
            video_id=str(video_id),
            profile=params.profile,
            root_id=root_id,
        )
        return JobView(job=job, video=video)


class ListJobs:
    """Джобы пользователя, новые сверху."""

    def __init__(self, *, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def execute(
        self,
        *,
        user_id: UUID,
        status: JobStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[Job]:
        async with self._uow_factory() as uow:
            jobs = await uow.jobs.list_for_user(user_id, status=status, limit=limit, offset=offset)
        return jobs


class RetryJob:
    """Перезапустить джоб с указанного этапа."""

    def __init__(
        self,
        *,
        uow_factory: UnitOfWorkFactory,
        dispatcher: PipelineDispatcher,
    ) -> None:
        self._uow_factory = uow_factory
        self._dispatcher = dispatcher

    async def execute(
        self,
        *,
        user_id: UUID,
        job_id: UUID,
        from_stage: JobStage | None = None,
    ) -> JobView:
        async with self._uow_factory() as uow:
            job = await uow.jobs.get(job_id)
            if job is None or job.user_id != user_id:
                raise NotFoundError("Джоб не найден", job_id=str(job_id))

            stage = from_stage or _first_incomplete(job)
            updated = clear_from_stage(job, stage)
            await uow.jobs.update(updated)
            await uow.commit()

        self._dispatcher.start(job_id, from_stage=stage)
        log.info("job.retried", job_id=str(job_id), from_stage=stage.value)
        return JobView(job=updated)


class CancelJob:
    """Остановить выполняющийся джоб."""

    def __init__(
        self,
        *,
        uow_factory: UnitOfWorkFactory,
        dispatcher: PipelineDispatcher,
    ) -> None:
        self._uow_factory = uow_factory
        self._dispatcher = dispatcher

    async def execute(self, *, user_id: UUID, job_id: UUID) -> JobView:
        async with self._uow_factory() as uow:
            job = await uow.jobs.get(job_id)
            if job is None or job.user_id != user_id:
                raise NotFoundError("Джоб не найден", job_id=str(job_id))
            if job.status.is_terminal:
                raise ValidationError(
                    f"Джоб уже завершён со статусом {job.status.value}",
                    job_id=str(job_id),
                    status=job.status.value,
                )

            # Статус пишется ДО отзыва задачи. Порядок важен: выполняющаяся
            # задача проверяет статус на границах этапов, и если она успеет
            # перейти к следующему раньше, чем мы отметим отмену, — она
            # его выполнит.
            cancelled = replace(job, status=JobStatus.CANCELLED)
            await uow.jobs.update(cancelled)
            await uow.commit()

        self._dispatcher.cancel(job.celery_root_id)
        log.info("job.cancelled", job_id=str(job_id), stage=job.stage.value if job.stage else None)
        return JobView(job=cancelled)


def clear_from_stage(job: Job, stage: JobStage) -> Job:
    """Стереть состояния этапов начиная с указанного.

    Чистая функция: правило «что именно считается заново» проверяется
    без базы. Всё, что раньше выбранного этапа, остаётся ``succeeded``,
    и задачи этих этапов при повторе выйдут немедленно.
    """
    if stage not in STAGE_ORDER:
        raise ValidationError(f"Неизвестный этап: {stage}", stage=str(stage))

    cutoff = STAGE_ORDER.index(stage)
    kept = {
        name: state
        for name, state in job.stages.items()
        if name in STAGE_ORDER and STAGE_ORDER.index(name) < cutoff
    }
    return replace(
        job,
        status=JobStatus.QUEUED,
        stage=stage,
        stages=kept,
        error=None,
        progress=_progress_of(kept),
    )


async def _check_quota(uow: UnitOfWork, user: User, video: Video) -> None:
    """Влезает ли ещё одно видео в месячную квоту пользователя.

    Месяц календарный, а не скользящий: «600 минут в месяц» люди
    понимают как «с первого числа», и скользящее окно, при котором
    квота освобождается по чуть-чуть, объяснять пришлось бы каждому.

    Квота — не защита от злоупотребления (для этого есть ограничитель
    частоты), а предсказуемость расходов: обработка часа видео стоит
    заметных денег на LLM и часов процессорного времени.
    """
    if user.quota_minutes_month <= 0:
        return

    now = datetime.now(UTC)
    since = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    used = await uow.jobs.minutes_used(user.id, since=since)
    incoming = (video.info.duration_sec if video.info else 0.0) / 60.0

    if used + incoming > user.quota_minutes_month:
        raise QuotaExceededError(
            f"Месячная квота исчерпана: использовано {used:.1f} из "
            f"{user.quota_minutes_month} минут, видео добавит ещё {incoming:.1f}.",
            used_minutes=round(used, 1),
            quota_minutes=user.quota_minutes_month,
            video_minutes=round(incoming, 1),
        )


def _first_incomplete(job: Job) -> JobStage:
    """С какого этапа начинать, если пользователь не сказал.

    С первого непройденного: повторять успешно завершённое незачем,
    а начинать с самого начала — значит потерять всё, что уже посчитано.
    """
    for stage in STAGE_ORDER:
        state = job.stages.get(stage)
        if state is None or not state.is_done:
            return stage
    # Все этапы пройдены — перезапускаем последний: раз пользователь
    # просит повтор, что-то в результате его не устроило.
    return STAGE_ORDER[-1]


def _progress_of(stages: Mapping[JobStage, StageState]) -> float:
    return compute_progress(stages)
