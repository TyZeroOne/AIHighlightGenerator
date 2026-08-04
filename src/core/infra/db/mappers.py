"""Перевод между ORM-строками и доменными сущностями.

Это та самая «скучная плата» за отделение домена от базы, о которой шла речь
на Этапе 1. Скучная — но локализованная: весь маппинг живёт в одном файле,
и больше нигде в проекте про SQLAlchemy знать не нужно.

Обратите внимание на ``Decimal`` → ``float``. В БД секунды хранятся как
``NUMERIC(10,3)``, потому что там важна точная десятичная арифметика.
В домене они ``float``, потому что там идут вычисления, а не хранение.
Граница между этими мирами — здесь.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Any
from uuid import UUID

from core.domain.entities import (
    Asset,
    Clip,
    ClipMetadata,
    Job,
    Scene,
    StageState,
    UploadSession,
    User,
    Video,
)
from core.domain.enums import JobStage
from core.domain.media import MediaInfo, TranscriptSegment, TranscriptWord
from core.domain.values import FeatureValue, Score, TimeRange
from core.infra.db.models import (
    AssetORM,
    ClipORM,
    JobORM,
    SceneORM,
    TranscriptSegmentORM,
    UploadSessionORM,
    UserORM,
    VideoORM,
)

__all__ = [
    "asset_to_domain",
    "clip_to_domain",
    "clip_to_orm_values",
    "job_to_domain",
    "job_to_orm_values",
    "scene_to_domain",
    "scene_to_orm_values",
    "transcript_segment_to_domain",
    "transcript_segment_to_orm_values",
    "upload_session_to_domain",
    "user_to_domain",
    "video_to_domain",
    "video_to_orm_values",
]


def _to_float(value: Decimal | float | None) -> float | None:
    return float(value) if value is not None else None


# ------------------------------------------------------------------ user
def user_to_domain(row: UserORM) -> User:
    return User(
        id=row.id,
        email=row.email,
        role=row.role,
        quota_minutes_month=row.quota_minutes_month,
        is_active=row.is_active,
    )


# ----------------------------------------------------------------- video
def video_to_domain(row: VideoORM) -> Video:
    info: MediaInfo | None = None
    if row.duration_sec is not None:
        info = MediaInfo(
            duration_sec=float(row.duration_sec),
            width=row.width,
            height=row.height,
            fps=_to_float(row.fps),
            video_codec=row.video_codec,
            audio_codec=row.audio_codec,
            container=row.container,
            size_bytes=row.size_bytes,
            raw=row.probe or {},
        )

    return Video(
        id=row.id,
        user_id=row.user_id,
        source_uri=row.source_uri,
        source_kind=row.source_kind,
        status=row.status,
        storage_key=row.storage_key,
        checksum_sha256=row.checksum_sha256,
        info=info,
        original_filename=row.original_filename,
        size_bytes=row.size_bytes,
        error=row.error,
        created_at=row.created_at,
    )


def video_to_orm_values(video: Video) -> dict[str, object]:
    """Домен → словарь колонок.

    Возвращаем словарь, а не готовый ORM-объект: это позволяет использовать
    результат и для ``INSERT``, и для частичного ``UPDATE``, не создавая
    лишнюю сущность в сессии.
    """
    values: dict[str, object] = {
        "id": video.id,
        "user_id": video.user_id,
        "source_uri": video.source_uri,
        "source_kind": video.source_kind,
        "status": video.status,
        "storage_key": video.storage_key,
        "checksum_sha256": video.checksum_sha256,
        "original_filename": video.original_filename,
        "error": dict(video.error) if video.error else None,
    }

    info = video.info
    if info is not None:
        values.update(
            {
                "duration_sec": Decimal(str(round(info.duration_sec, 3))),
                "width": info.width,
                "height": info.height,
                "fps": Decimal(str(round(info.fps, 3))) if info.fps is not None else None,
                "video_codec": info.video_codec,
                "audio_codec": info.audio_codec,
                "container": info.container,
                "size_bytes": info.size_bytes if info.size_bytes is not None else video.size_bytes,
                "has_audio": info.has_audio,
                "probe": dict(info.raw),
            }
        )
    elif video.size_bytes is not None:
        values["size_bytes"] = video.size_bytes

    return values


# ------------------------------------------------------------------- job
def job_to_domain(row: JobORM) -> Job:
    return Job(
        id=row.id,
        video_id=row.video_id,
        user_id=row.user_id,
        status=row.status,
        mode=row.mode,
        stage=row.stage,
        progress=float(row.progress or 0),
        stages=_stages_to_domain(row.stage_states or {}),
        params=dict(row.params or {}),
        params_hash=row.params_hash,
        trace_id=row.trace_id,
        error=row.error,
        cost=dict(row.cost or {}),
        created_at=row.created_at,
    )


def job_to_orm_values(job: Job) -> dict[str, object]:
    return {
        "id": job.id,
        "video_id": job.video_id,
        "user_id": job.user_id,
        "status": job.status,
        "mode": job.mode,
        "stage": job.stage,
        "progress": Decimal(str(round(job.progress, 2))),
        "stage_states": _stages_to_json(job),
        "params": dict(job.params),
        "params_hash": job.params_hash,
        "trace_id": job.trace_id,
        "error": dict(job.error) if job.error else None,
        "cost": dict(job.cost),
    }


def _stages_to_domain(raw: dict[str, Any]) -> dict[JobStage, StageState]:
    """JSONB → состояния этапов.

    Незнакомые ключи молча пропускаются: этап могли переименовать или убрать,
    и старый джоб из базы не должен ронять чтение. Потерять историю одного
    этапа безопаснее, чем не открыть джоб вообще.
    """
    states: dict[JobStage, StageState] = {}
    for key, value in raw.items():
        try:
            stage = JobStage(key)
        except ValueError:
            continue
        if not isinstance(value, dict):
            continue
        states[stage] = StageState(
            stage=stage,
            status=str(value.get("status", "pending")),
            progress=float(value.get("progress") or 0.0),
            duration_sec=_to_float(value.get("duration_sec")),
            error=value.get("error"),
        )
    return states


def _stages_to_json(job: Job) -> dict[str, Any]:
    return {
        stage.value: {
            "status": state.status,
            "progress": state.progress,
            "duration_sec": state.duration_sec,
            "error": state.error,
        }
        for stage, state in job.stages.items()
    }


# ----------------------------------------------------------------- scene
def scene_to_domain(row: SceneORM) -> Scene:
    return Scene(
        id=row.id,
        video_id=row.video_id,
        job_id=row.job_id,
        index=row.index,
        range=TimeRange(float(row.start_sec), float(row.end_sec)),
        features=_features_to_domain(row.features or {}),
        heuristic_score=(
            Score(value=float(row.heuristic_score), breakdown=row.score_breakdown or {})
            if row.heuristic_score is not None
            else None
        ),
        llm_score=_to_float(row.llm_score),
        final_score=_to_float(row.final_score),
        category=row.llm_category,
        reason=row.llm_reason,
        transcript_text=row.transcript_text,
        keyframe_key=row.keyframe_key,
    )


def scene_to_orm_values(scene: Scene) -> dict[str, object]:
    return {
        "id": scene.id,
        "video_id": scene.video_id,
        "job_id": scene.job_id,
        "index": scene.index,
        "start_sec": Decimal(str(round(scene.range.start, 3))),
        "end_sec": Decimal(str(round(scene.range.end, 3))),
        "keyframe_key": scene.keyframe_key,
        "features": {
            key: {"value": value.value, "confidence": value.confidence}
            for key, value in scene.features.items()
        },
        "heuristic_score": (
            Decimal(str(round(scene.heuristic_score.value, 3)))
            if scene.heuristic_score is not None
            else None
        ),
        "score_breakdown": (
            dict(scene.heuristic_score.breakdown) if scene.heuristic_score is not None else None
        ),
        "llm_score": Decimal(str(round(scene.llm_score, 3)))
        if scene.llm_score is not None
        else None,
        "llm_category": scene.category,
        "llm_reason": scene.reason,
        "final_score": (
            Decimal(str(round(scene.final_score, 3))) if scene.final_score is not None else None
        ),
        "transcript_text": scene.transcript_text,
    }


def _features_to_domain(raw: dict[str, Any]) -> dict[str, FeatureValue]:
    features: dict[str, FeatureValue] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            features[key] = FeatureValue(
                key=key,
                value=float(value.get("value", 0.0)),
                confidence=float(value.get("confidence", 1.0)),
            )
        else:
            # Допускаем «плоский» формат {"audio.loudness": 0.7}: его удобно
            # писать руками при отладке, а читаться он должен так же.
            features[key] = FeatureValue(key=key, value=float(value))
    return features


# ------------------------------------------------------------ transcript
def transcript_segment_to_domain(row: TranscriptSegmentORM) -> TranscriptSegment:
    return TranscriptSegment(
        text=row.text_content,
        range=TimeRange(float(row.start_sec), float(row.end_sec)),
        words=tuple(_words_to_domain(row.words or [])),
        language=row.language,
        avg_logprob=_to_float(row.avg_logprob),
        no_speech_prob=_to_float(row.no_speech_prob),
        speaker=row.speaker,
    )


def transcript_segment_to_orm_values(
    segment: TranscriptSegment,
    *,
    job_id: UUID,
    video_id: UUID,
) -> dict[str, object]:
    return {
        "job_id": job_id,
        "video_id": video_id,
        "start_sec": Decimal(str(round(segment.range.start, 3))),
        "end_sec": Decimal(str(round(segment.range.end, 3))),
        "text": segment.text,
        # Ключи короткие намеренно: слов в двухчасовом фильме под двадцать
        # тысяч, и «start»/«end»/«probability» вместо «s»/«e»/«p» — это
        # лишний мегабайт JSONB на каждое видео без единого нового факта.
        "words": [
            {
                "w": word.text,
                "s": round(word.start, 3),
                "e": round(word.end, 3),
                "p": round(word.probability, 4),
            }
            for word in segment.words
        ],
        "language": segment.language,
        "avg_logprob": _decimal_or_none(segment.avg_logprob, 4),
        "no_speech_prob": _decimal_or_none(segment.no_speech_prob, 4),
        "speaker": segment.speaker,
    }


def _words_to_domain(raw: list[dict[str, Any]]) -> list[TranscriptWord]:
    words: list[TranscriptWord] = []
    for item in raw:
        try:
            words.append(
                TranscriptWord(
                    text=str(item["w"]),
                    start=float(item["s"]),
                    end=float(item["e"]),
                    probability=float(item.get("p", 1.0)),
                )
            )
        except (KeyError, TypeError, ValueError):
            # Битое слово пропускаем: терять весь транскрипт из-за одной
            # записи, испорченной ручной правкой или старым форматом, — хуже.
            continue
    return words


def _decimal_or_none(value: float | None, places: int) -> Decimal | None:
    return Decimal(str(round(value, places))) if value is not None else None


# --------------------------------------------------------- upload session
def upload_session_to_domain(row: UploadSessionORM) -> UploadSession:
    return UploadSession(
        id=row.id,
        user_id=row.user_id,
        filename=row.filename,
        chunk_size=row.chunk_size,
        total_size=row.total_size,
        received_parts=tuple(sorted(row.received_parts or [])),
        received_bytes=row.received_bytes,
        completed=row.completed,
        created_at=row.created_at,
    )


# ------------------------------------------------------------------ clip
def clip_to_domain(row: ClipORM, scene_ids: Sequence[UUID] = ()) -> Clip:
    metadata = None
    if row.title or row.description or row.hashtags:
        metadata = ClipMetadata(
            title=row.title or "",
            description=row.description or "",
            hashtags=tuple(row.hashtags or ()),
        )
    return Clip(
        id=row.id,
        job_id=row.job_id,
        video_id=row.video_id,
        rank=row.rank,
        range=TimeRange(float(row.start_sec), float(row.end_sec)),
        score=float(row.score),
        category=row.category,
        reason=row.reason or "",
        metadata=metadata,
        scene_ids=tuple(scene_ids),
        status=row.status,
        error=(row.error or {}).get("message") if row.error else None,
    )


def clip_to_orm_values(clip: Clip) -> dict[str, object]:
    metadata = clip.metadata
    return {
        "id": clip.id,
        "job_id": clip.job_id,
        "video_id": clip.video_id,
        "rank": clip.rank,
        "start_sec": Decimal(str(round(clip.range.start, 3))),
        "end_sec": Decimal(str(round(clip.range.end, 3))),
        "score": Decimal(str(round(clip.score, 3))),
        "category": clip.category,
        "reason": clip.reason,
        "title": metadata.title if metadata else None,
        "description": metadata.description if metadata else None,
        "hashtags": list(metadata.hashtags) if metadata else None,
        "status": clip.status,
        # Ошибка хранится объектом, а не строкой: колонка jsonb, и когда
        # к сообщению понадобится добавить код или подсказку, это будет
        # ключом в словаре, а не миграцией.
        "error": {"message": clip.error} if clip.error else None,
    }


# ----------------------------------------------------------------- asset
def asset_to_domain(row: AssetORM) -> Asset:
    """Строка ORM → неизменяемый доменный объект.

    Отдавать наружу саму строку нельзя: она привязана к сессии и после
    её закрытия отдаёт `DetachedInstanceError` вместо значения поля.
    """
    return Asset(
        id=row.id,
        owner_type=row.owner_type,
        owner_id=row.owner_id,
        kind=row.kind,
        storage_key=row.storage_key,
        mime=row.mime,
        size_bytes=row.size_bytes,
        meta=row.meta or {},
        expires_at=row.expires_at,
    )
