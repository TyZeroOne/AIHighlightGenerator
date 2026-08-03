"""Сущности домена.

Это НЕ ORM-модели. ORM живёт в ``core/infra/db/models.py``, а репозитории
переводят одно в другое. Плата — немного ручного маппинга; выгода — домен
и все его тесты не знают ни про SQLAlchemy, ни про базу вообще.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from uuid import UUID

from core.domain.enums import (
    STAGE_WEIGHTS,
    JobMode,
    JobStage,
    JobStatus,
    SourceKind,
    UserRole,
    VideoStatus,
)
from core.domain.media import MediaInfo
from core.domain.values import FeatureValue, Score, TimeRange

__all__ = [
    "Clip",
    "ClipMetadata",
    "Highlight",
    "Job",
    "Scene",
    "StageState",
    "UploadSession",
    "User",
    "Video",
    "compute_progress",
]


@dataclass(frozen=True, slots=True)
class User:
    id: UUID
    email: str
    role: UserRole = UserRole.USER
    quota_minutes_month: int = 600
    is_active: bool = True


@dataclass(frozen=True, slots=True)
class Video:
    id: UUID
    user_id: UUID
    source_uri: str
    source_kind: SourceKind
    status: VideoStatus
    storage_key: str
    checksum_sha256: str
    info: MediaInfo | None = None
    original_filename: str | None = None
    size_bytes: int | None = None
    error: Mapping[str, object] | None = None
    created_at: datetime | None = None

    @property
    def is_usable(self) -> bool:
        """Готово ли видео к анализу. Проверяется перед постановкой джоба."""
        return self.status is VideoStatus.READY


@dataclass(frozen=True, slots=True)
class UploadSession:
    """Состояние возобновляемой чанковой загрузки.

    Хранится в БД, а не в памяти процесса: загрузка 20 ГБ переживает
    перезапуск API и может продолжиться на другой реплике.
    """

    id: UUID
    user_id: UUID
    filename: str
    chunk_size: int
    total_size: int | None = None
    received_parts: tuple[int, ...] = ()
    received_bytes: int = 0
    completed: bool = False
    created_at: datetime | None = None

    def next_expected_part(self) -> int:
        """Первый недостающий номер части — подсказка клиенту при возобновлении."""
        received = set(self.received_parts)
        part = 1
        while part in received:
            part += 1
        return part


@dataclass(frozen=True, slots=True)
class StageState:
    """Состояние одного этапа внутри джоба."""

    stage: JobStage
    status: str  # pending | running | succeeded | failed | skipped
    progress: float = 0.0  # 0..100 внутри этапа
    duration_sec: float | None = None
    error: str | None = None

    @property
    def is_done(self) -> bool:
        return self.status in {"succeeded", "skipped"}

    @property
    def completed_fraction(self) -> float:
        """Доля этапа, которую можно засчитать в общий прогресс, 0..1.

        Завершённый этап засчитывается целиком независимо от того, что лежит
        в ``progress``: длинные этапы рапортуют дробный прогресс сами и могут
        закончиться, не дойдя до ровных 100.
        """
        if self.is_done:
            return 1.0
        if self.status == "failed":
            return 0.0
        return min(max(self.progress, 0.0), 100.0) / 100.0


@dataclass(frozen=True, slots=True)
class Job:
    id: UUID
    video_id: UUID
    user_id: UUID
    status: JobStatus
    mode: JobMode = JobMode.BATCH
    stage: JobStage | None = None
    progress: float = 0.0
    stages: Mapping[JobStage, StageState] = field(default_factory=dict)
    params: Mapping[str, object] = field(default_factory=dict)
    params_hash: str = ""
    trace_id: UUID | None = None
    error: Mapping[str, object] | None = None
    created_at: datetime | None = None

    def is_stage_done(self, stage: JobStage) -> bool:
        """Основа идемпотентности: задача, увидев ``True``, немедленно выходит."""
        state = self.stages.get(stage)
        return state is not None and state.is_done

    def with_stage(
        self,
        stage: JobStage,
        status: str,
        *,
        progress: float = 0.0,
        duration_sec: float | None = None,
        error: str | None = None,
    ) -> Job:
        """Копия джоба с обновлённым состоянием этапа и пересчитанным прогрессом.

        Пересчёт живёт здесь, а не в репозитории, по двум причинам. Во-первых,
        это правило предметной области: «прогресс джоба — взвешенная сумма
        прогресса этапов», и проверять его надо тестом без базы. Во-вторых,
        любой другой владелец этой логики означал бы, что записать состояние
        этапа можно в обход пересчёта — и рано или поздно кто-нибудь так
        и сделает, а джоб застынет на 41 % при готовом результате.
        """
        states = dict(self.stages)
        states[stage] = StageState(
            stage=stage,
            status=status,
            progress=progress,
            duration_sec=duration_sec,
            error=error,
        )
        return replace(
            self,
            stage=stage,
            stages=states,
            progress=compute_progress(states),
        )


def compute_progress(states: Mapping[JobStage, StageState]) -> float:
    """Общий прогресс джоба, 0..100.

    Веса этапов в сумме дают 100 (док 04, §4.8), поэтому итог — это просто
    сумма «вес × доля выполнения». Неизвестные этапы считаются невыполненными:
    отсутствие записи о `transcribe` означает, что он ещё не начинался.
    """
    total = sum(
        STAGE_WEIGHTS[stage] * state.completed_fraction
        for stage, state in states.items()
        if stage in STAGE_WEIGHTS
    )
    return round(min(total, 100.0), 2)


@dataclass(frozen=True, slots=True)
class Scene:
    id: UUID
    video_id: UUID
    job_id: UUID
    index: int
    range: TimeRange
    features: Mapping[str, FeatureValue] = field(default_factory=dict)
    heuristic_score: Score | None = None
    llm_score: float | None = None
    final_score: float | None = None
    category: str | None = None
    reason: str | None = None
    transcript_text: str | None = None
    keyframe_key: str | None = None

    def feature(self, key: str, default: float = 0.0) -> float:
        """Безопасное чтение признака.

        Отсутствие признака — штатная ситуация: анализатор мог быть отключён
        или упасть (док 04, §4.5). Поэтому вместо KeyError — значение по умолчанию.
        """
        value = self.features.get(key)
        return value.value if value is not None else default


@dataclass(frozen=True, slots=True)
class Highlight:
    """Кандидат в клипы: отрезок, отобранный из одной или нескольких сцен."""

    range: TimeRange
    final_score: float
    category: str
    reason: str
    scene_ids: tuple[UUID, ...] = ()
    breakdown: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ClipMetadata:
    title: str
    description: str
    hashtags: Sequence[str] = field(default_factory=tuple)
    hook: str | None = None


@dataclass(frozen=True, slots=True)
class Clip:
    id: UUID
    job_id: UUID
    video_id: UUID
    rank: int
    range: TimeRange
    score: float
    category: str
    reason: str
    metadata: ClipMetadata | None = None
    scene_ids: tuple[UUID, ...] = ()
