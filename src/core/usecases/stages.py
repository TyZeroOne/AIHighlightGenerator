"""Общая обвязка этапа конвейера.

К третьему этапу стало видно, что у всех задач одинаковый каркас: загрузить
джоб и видео, проверить, не пройден ли этап уже, отметить начало, выполнить
работу, отметить успех или провал. Разное — только середина.

Копировать этот каркас в каждый сценарий нельзя не из эстетических
соображений. Отметки этапа — это то, из чего складывается прогресс джоба
и признак его живости; сценарий, забывший отметить провал, оставит джоб
навсегда в состоянии «выполняется». Один экземпляр этой логики забыть
невозможно.
"""

from __future__ import annotations

from uuid import UUID

from core.domain.entities import Job, StageState, Video
from core.domain.enums import JobStage, JobStatus
from core.domain.errors import JobCancelledError, NotFoundError, ValidationError
from core.domain.params import AnalysisParams
from core.logging import get_logger
from core.ports.repository import UnitOfWorkFactory

__all__ = ["StageTracker", "params_of"]

log = get_logger(__name__)


class StageTracker:
    """Состояние одного этапа джоба: чтение, отметки, переходы статуса."""

    def __init__(self, *, uow_factory: UnitOfWorkFactory, stage: JobStage) -> None:
        self._uow_factory = uow_factory
        self._stage = stage

    @property
    def stage(self) -> JobStage:
        return self._stage

    async def load(self, job_id: UUID) -> tuple[Job, Video]:
        """Джоб и его видео. Бросает, если работать не с чем."""
        async with self._uow_factory() as uow:
            job = await uow.jobs.get(job_id)
            if job is None:
                raise NotFoundError("Джоб не найден", job_id=str(job_id))
            video = await uow.videos.get(job.video_id)

        if job.status is JobStatus.CANCELLED:
            # Проверка отмены — на границе каждого этапа, и это единственное
            # место, где она нужна. Отзыв задачи в Celery останавливает только
            # то, что ещё не началось; уже выполняющаяся цепочка узнаёт
            # об отмене здесь и не берётся за следующий этап.
            raise JobCancelledError("Джоб отменён", job_id=str(job_id), stage=self._stage.value)

        if video is None:
            raise NotFoundError("Видео джоба не найдено", video_id=str(job.video_id))
        if not video.is_usable:
            raise ValidationError(
                f"Видео недоступно для анализа: статус {video.status.value}",
                video_id=str(video.id),
                status=video.status.value,
            )
        return job, video

    async def begin(self, job: Job) -> None:
        await self._write(job, "running", status=JobStatus.RUNNING)

    async def succeed(self, job: Job, *, duration_sec: float) -> None:
        await self._write(
            job,
            "succeeded",
            progress=100.0,
            duration_sec=duration_sec,
            status=JobStatus.RUNNING,
        )

    async def skip(self, job: Job, reason: str) -> None:
        """Этап неприменим к этому видео — например, звука в нём нет.

        ``skipped``, а не ``succeeded``: в прогрессе он засчитывается целиком
        (иначе джоб застрял бы навсегда), но в статусе видно, что работы
        не было, — и разбирая потом пустой транскрипт, не придётся гадать,
        сломалась модель или речи не было изначально.
        """
        await self._write(
            job, "skipped", progress=100.0, stage_error=reason, status=JobStatus.RUNNING
        )
        log.info("stage.skipped", stage=self._stage.value, job_id=str(job.id), reason=reason)

    async def fail(self, job: Job, error: Exception, *, duration_sec: float) -> None:
        message = f"{type(error).__name__}: {error}"
        await self._write(
            job,
            "failed",
            duration_sec=duration_sec,
            stage_error=message,
            status=JobStatus.FAILED,
            error={
                "stage": self._stage.value,
                "type": type(error).__name__,
                "message": str(error),
            },
        )
        log.error("stage.failed", stage=self._stage.value, job_id=str(job.id), error=str(error))

    async def progress(self, job: Job, fraction: float) -> None:
        """Дробный прогресс внутри длинного этапа (док 04, §4.8).

        Нужен там, где этап идёт минутами: без него транскрипция часового
        файла выглядит снаружи как двадцать минут полной тишины.
        """
        await self._write(
            job,
            "running",
            progress=min(max(fraction, 0.0), 1.0) * 100.0,
            status=JobStatus.RUNNING,
        )

    async def _write(
        self,
        job: Job,
        state: str,
        *,
        status: JobStatus,
        progress: float = 0.0,
        duration_sec: float | None = None,
        stage_error: str | None = None,
        error: dict[str, object] | None = None,
    ) -> None:
        """Записать состояние этапа отдельной короткой транзакцией.

        Отдельной — намеренно. Отметка «этап начался» должна быть видна
        снаружи немедленно: она источник прогресса в API и признак живости
        джоба для уборщика зависших. Внутри общей транзакции этапа она стала
        бы видна только по его завершении, то есть через минуты.

        И **только своё поле**, а не джоб целиком. Ветки конвейера идут
        параллельно, и запись объекта, прочитанного в начале этапа,
        затирает отметки соседней ветки. Живой прогон Этапа 10 показал
        именно это: джоб застрял на 52 %, потеряв запись об успешно
        отработавшем видеоанализе, — тот же lost update, что был
        с ``jobs.cost`` на Этапе 6.
        """
        async with self._uow_factory() as uow:
            await uow.jobs.write_stage(
                job.id,
                self._stage,
                StageState(
                    stage=self._stage,
                    status=state,
                    progress=progress,
                    duration_sec=duration_sec,
                    error=stage_error,
                ),
                status=status,
                error=error,
            )
            await uow.commit()


def params_of(job: Job, fallback_profile: str = "default") -> AnalysisParams:
    """Параметры конкретного джоба, а не настройки процесса.

    До Этапа 10 профиль приходил из конфига: конвейер запускала командная
    строка, и другого источника не было. Теперь его задаёт запрос — и если
    бы этап продолжал читать конфиг, ``POST /analyze`` с профилем
    ``podcast`` молча считался бы профилем по умолчанию, то есть API врал бы
    о том, что делает. Настройка остаётся значением на случай, когда джоб
    создан без параметров (например, из старой команды CLI).
    """
    if not job.params:
        return AnalysisParams(profile=fallback_profile)
    try:
        return AnalysisParams.from_mapping(job.params)
    except (TypeError, ValueError):
        # Параметры могли быть записаны прежней версией и не пройти нынешнюю
        # проверку. Уронить из-за этого готовый джоб хуже, чем посчитать его
        # значениями по умолчанию.
        log.warning("job.params_unreadable", job_id=str(job.id))
        return AnalysisParams(profile=fallback_profile)
