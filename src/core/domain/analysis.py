"""Типы, которыми обмениваются анализаторы, скоринг и LLM-слой."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from core.domain.media import AudioTrack, LocalMedia, Transcript
from core.domain.values import MediaWindow, TimeRange

__all__ = [
    "ARTIFACT_AUDIO",
    "ARTIFACT_TRANSCRIPT",
    "ARTIFACT_VIDEO",
    "AnalysisContext",
    "LLMVerdict",
    "RenderSpec",
    "SceneSummary",
    "ScoringProfile",
    "VideoContext",
]

#: Артефакты — «сырьё», которое анализаторы объявляют в ``requires``.
#: Строковые константы, а не enum: сторонний плагин из другого пакета
#: должен уметь объявить зависимость, не импортируя наш enum.
ARTIFACT_VIDEO = "artifact.video"
ARTIFACT_AUDIO = "artifact.audio"
ARTIFACT_TRANSCRIPT = "artifact.transcript"


@dataclass(frozen=True, slots=True)
class AnalysisContext:
    """Всё, что может понадобиться анализатору.

    Один общий контекст вместо отдельной сигнатуры под каждый анализатор:
    добавление нового вида сырья не ломает интерфейс существующих плагинов.
    """

    job_id: UUID
    video_id: UUID
    window: MediaWindow
    workdir: Path
    #: Видеофайл. ``None`` — он не скачивался, потому что этому этапу не нужен.
    #: Аудиоанализу видео ни к чему, а тянуть ради него двадцать гигабайт
    #: из S3 — плата за ничто.
    media: LocalMedia | None = None
    audio: AudioTrack | None = None
    transcript: Transcript | None = None

    def available_artifacts(self) -> frozenset[str]:
        """Какое сырьё реально есть. Отсюда планировщик знает, что можно запускать."""
        artifacts: set[str] = set()
        if self.media is not None and self.media.info.has_video:
            artifacts.add(ARTIFACT_VIDEO)
        if self.audio is not None:
            artifacts.add(ARTIFACT_AUDIO)
        if self.transcript is not None and not self.transcript.is_empty:
            artifacts.add(ARTIFACT_TRANSCRIPT)
        return frozenset(artifacts)


@dataclass(frozen=True, slots=True)
class ScoringProfile:
    """Профиль скоринга: веса групп и ограничения на клипы."""

    name: str
    group_weights: Mapping[str, float]
    signal_weights: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    min_duration_sec: float = 15.0
    max_duration_sec: float = 90.0
    min_score: float = 40.0
    max_clips: int = 10

    def __post_init__(self) -> None:
        if not self.group_weights:
            raise ValueError(f"Профиль {self.name!r} не содержит весов групп")
        if self.min_duration_sec >= self.max_duration_sec:
            raise ValueError(
                f"Профиль {self.name!r}: min_duration >= max_duration "
                f"({self.min_duration_sec} >= {self.max_duration_sec})"
            )


@dataclass(frozen=True, slots=True)
class SceneSummary:
    """Компактное описание сцены для LLM.

    Намеренно не содержит сырых признаков: в модель уходит человекочитаемая
    сводка, а не 40 чисел (док 06, §6.4).
    """

    index: int
    range: TimeRange
    signals: Mapping[str, float]
    transcript_excerpt: str
    notes: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class VideoContext:
    """Контекст видео целиком — уходит в кешируемую часть промпта."""

    title: str | None
    duration_sec: float
    language: str | None
    profile: str
    scene_count: int


@dataclass(frozen=True, slots=True)
class LLMVerdict:
    """Ответ модели по одной сцене (док 06, §6.3)."""

    scene_index: int
    range: TimeRange
    score: float
    category: str
    reason: str


@dataclass(frozen=True, slots=True)
class RenderSpec:
    """Параметры рендера клипа."""

    width: int | None = None
    height: int | None = None
    crf: int = 20
    preset: str = "medium"
    audio_bitrate: str = "192k"
    burn_subtitles_path: Path | None = None
