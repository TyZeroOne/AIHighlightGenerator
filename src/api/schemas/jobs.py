"""Схемы запуска и состояния джоба.

Граница между HTTP и предметной областью проходит здесь: pydantic
проверяет форму запроса и типы, доменный ``AnalysisParams`` — смысл
(что максимум клипов положительный, что минимум меньше максимума).
Разделение не формальное: те же правила действуют, когда джоб запускают
из командной строки, а pydantic там не участвует.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from core.domain.entities import Job
from core.domain.enums import STAGE_WEIGHTS, JobStage
from core.domain.params import MAX_CLIPS_LIMIT, AnalysisParams

__all__ = [
    "AnalyzeRequest",
    "JobListResponse",
    "JobResponse",
    "JobStatusResponse",
    "RetryRequest",
    "StageStateResponse",
]


class AnalyzeRequest(BaseModel):
    """Запрос на анализ. Всё, кроме ``video_id``, необязательно."""

    video_id: UUID
    profile: str = "default"
    max_clips: int = Field(default=10, ge=1, le=MAX_CLIPS_LIMIT)
    min_duration_sec: float = Field(default=15.0, gt=0)
    max_duration_sec: float = Field(default=90.0, gt=0)
    vertical: bool = True
    burn_subtitles: bool = False
    language: str | None = None
    use_llm: bool = True
    analyzers_disabled: Sequence[str] = ()
    #: Пересчитать, даже если такой джоб уже посчитан. По умолчанию
    #: повторный запрос возвращает готовый результат, а не тратит
    #: сорок минут на то же самое.
    force: bool = False

    def to_params(self) -> AnalysisParams:
        return AnalysisParams(
            profile=self.profile,
            max_clips=self.max_clips,
            min_duration_sec=self.min_duration_sec,
            max_duration_sec=self.max_duration_sec,
            vertical=self.vertical,
            burn_subtitles=self.burn_subtitles,
            language=self.language,
            use_llm=self.use_llm,
            analyzers_disabled=tuple(self.analyzers_disabled),
        )


class RetryRequest(BaseModel):
    #: С какого этапа повторять. ``None`` — с первого непройденного.
    from_stage: JobStage | None = None


class JobResponse(BaseModel):
    """Короткий вид джоба — для ответа на запуск и для списков."""

    job_id: UUID
    video_id: UUID
    status: str
    stage: str | None = None
    progress: float = 0.0
    params_hash: str = ""
    created_at: datetime | None = None
    #: ``true`` — вернули уже посчитанный джоб, работу не запускали.
    reused: bool = False

    @classmethod
    def from_domain(cls, job: Job, *, reused: bool = False) -> JobResponse:
        return cls(
            job_id=job.id,
            video_id=job.video_id,
            status=job.status.value,
            stage=job.stage.value if job.stage else None,
            progress=job.progress,
            params_hash=job.params_hash,
            created_at=job.created_at,
            reused=reused,
        )


class StageStateResponse(BaseModel):
    status: str
    progress: float = 0.0
    duration_sec: float | None = None
    error: str | None = None


class JobStatusResponse(BaseModel):
    """Подробное состояние: по нему клиент рисует прогресс."""

    job_id: UUID
    video_id: UUID
    status: str
    stage: str | None = None
    progress: float = 0.0
    stages: Mapping[str, StageStateResponse] = Field(default_factory=dict)
    counts: Mapping[str, int] = Field(default_factory=dict)
    cost: Mapping[str, object] = Field(default_factory=dict)
    error: Mapping[str, object] | None = None
    created_at: datetime | None = None
    #: Грубая оценка оставшегося времени. Именно грубая: считается
    #: по весам этапов и уже потраченному времени, и на видео,
    #: непохожем на среднее, ошибётся в разы. Отдаём её потому, что
    #: «осталось примерно» полезнее, чем ничего.
    eta_sec: float | None = None

    @classmethod
    def from_domain(
        cls,
        job: Job,
        *,
        scenes: int = 0,
        clips: int = 0,
    ) -> JobStatusResponse:
        return cls(
            job_id=job.id,
            video_id=job.video_id,
            status=job.status.value,
            stage=job.stage.value if job.stage else None,
            progress=job.progress,
            stages={
                stage.value: StageStateResponse(
                    status=state.status,
                    progress=state.progress,
                    duration_sec=state.duration_sec,
                    error=state.error,
                )
                for stage, state in sorted(job.stages.items(), key=lambda item: item[0].value)
            },
            counts={"scenes": scenes, "clips": clips},
            cost=dict(job.cost),
            error=job.error,
            created_at=job.created_at,
            eta_sec=estimate_eta(job),
        )


class JobListResponse(BaseModel):
    total: int
    items: Sequence[JobResponse] = ()


def estimate_eta(job: Job) -> float | None:
    """Сколько примерно осталось, в секундах.

    Считается по уже пройденным этапам: известно, сколько весит каждый
    этап в общем прогрессе и сколько секунд заняли завершённые. Отсюда
    цена одного «процента» — и, умножив на оставшиеся, получаем оценку.

    Почему не по среднему времени этапов из истории: на этом видео
    транскрипция может занять минуту, а на соседнем полчаса — всё зависит
    от длительности и наличия речи. Оценка по **этому же** джобу
    ошибается, но ошибается согласованно.
    """
    if job.status.is_terminal:
        return None

    done_weight = 0.0
    done_seconds = 0.0
    for stage, state in job.stages.items():
        weight = STAGE_WEIGHTS.get(stage, 0)
        if state.is_done and state.duration_sec is not None:
            done_weight += weight
            done_seconds += state.duration_sec

    if done_weight <= 0 or done_seconds <= 0:
        return None

    remaining_weight = max(sum(STAGE_WEIGHTS.values()) - done_weight, 0.0)
    return round(remaining_weight * (done_seconds / done_weight), 1)
