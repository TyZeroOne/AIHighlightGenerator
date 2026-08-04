"""Перечисления домена.

Все наследуют ``str``, чтобы значение сериализовалось в JSON как строка
и совпадало со значением нативного enum-типа PostgreSQL. Это избавляет
от ручных конвертаций на трёх границах: БД → домен → API.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "STAGE_ORDER",
    "AssetKind",
    "AssetOwnerType",
    "ClipStatus",
    "JobMode",
    "JobStage",
    "JobStatus",
    "SourceKind",
    "SubtitleFormat",
    "UserRole",
    "VideoStatus",
]


class UserRole(StrEnum):
    USER = "user"
    ADMIN = "admin"


class SourceKind(StrEnum):
    """Откуда взялось видео.

    ``youtube``/``twitch``/``kick`` объявлены заранее и намеренно: добавление
    значения в enum PostgreSQL требует миграции, и лучше заложить их сейчас,
    чем ломать прод потом. Реализации провайдеров появятся позже — само
    наличие значения ничего не включает.
    """

    UPLOAD = "upload"
    LOCAL_FILE = "local_file"
    YOUTUBE = "youtube"
    TWITCH = "twitch"
    KICK = "kick"


class VideoStatus(StrEnum):
    UPLOADING = "uploading"
    READY = "ready"
    FAILED = "failed"
    DELETED = "deleted"


class JobMode(StrEnum):
    BATCH = "batch"  # файл целиком
    STREAMING = "streaming"  # скользящее окно (задел под стримы)


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    PARTIAL = "partial"  # часть этапов упала, но результат есть
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_JOB_STATUSES


_TERMINAL_JOB_STATUSES = frozenset(
    {JobStatus.PARTIAL, JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}
)


class JobStage(StrEnum):
    """Этапы конвейера. Порядок объявления = порядок исполнения."""

    INGEST = "ingest"
    DETECT_SCENES = "detect_scenes"
    EXTRACT_AUDIO = "extract_audio"
    TRANSCRIBE = "transcribe"
    ANALYZE_AUDIO = "analyze_audio"
    ANALYZE_SPEECH = "analyze_speech"
    ANALYZE_VISION = "analyze_vision"
    SCORE_SCENES = "score_scenes"
    LLM_RANK = "llm_rank"
    SELECT_HIGHLIGHTS = "select_highlights"
    RENDER_CLIPS = "render_clips"
    GENERATE_METADATA = "generate_metadata"
    FINALIZE = "finalize"


#: Порядок этапов как последовательность. Отдельное имя, а не
#: ``list(JobStage)`` по месту: от этого порядка зависит, что именно
#: пересчитывается при повторе с этапа, и такую вещь надо называть.
STAGE_ORDER: tuple[JobStage, ...] = tuple(JobStage)


#: Вклад каждого этапа в общий прогресс (в сумме 100). Держится рядом
#: с самим enum, чтобы новый этап нельзя было добавить, забыв про прогресс:
#: тест сверяет ключи этого словаря с составом JobStage.
STAGE_WEIGHTS: dict[JobStage, int] = {
    # Нулевой вес намеренно. ``ingest`` описан как этап и остался в enum,
    # но отдельной задачи у него нет: файл скачивает тот этап, которому
    # он понадобился, через ``SourceProvider``. Пока вес был ненулевым,
    # успешно завершённый джоб показывал 97 % — три процента ждали шага,
    # которого не существует.
    JobStage.INGEST: 0,
    JobStage.DETECT_SCENES: 10,
    JobStage.EXTRACT_AUDIO: 5,
    JobStage.TRANSCRIBE: 25,
    JobStage.ANALYZE_AUDIO: 5,
    JobStage.ANALYZE_SPEECH: 5,
    JobStage.ANALYZE_VISION: 20,
    JobStage.SCORE_SCENES: 2,
    JobStage.LLM_RANK: 8,
    JobStage.SELECT_HIGHLIGHTS: 2,
    JobStage.RENDER_CLIPS: 15,
    JobStage.GENERATE_METADATA: 2,
    JobStage.FINALIZE: 1,
}


class ClipStatus(StrEnum):
    PENDING = "pending"
    RENDERING = "rendering"
    READY = "ready"
    FAILED = "failed"


class AssetOwnerType(StrEnum):
    VIDEO = "video"
    JOB = "job"
    CLIP = "clip"


class AssetKind(StrEnum):
    SOURCE = "source"
    AUDIO_WAV = "audio_wav"
    KEYFRAME = "keyframe"
    CLIP_H = "clip_h"  # горизонтальный клип
    CLIP_V = "clip_v"  # вертикальный 1080×1920
    SUBTITLE_SRT = "subtitle_srt"
    SUBTITLE_ASS = "subtitle_ass"
    THUMBNAIL = "thumbnail"
    MANIFEST = "manifest"


class SubtitleFormat(StrEnum):
    SRT = "srt"
    ASS = "ass"
    VTT = "vtt"
