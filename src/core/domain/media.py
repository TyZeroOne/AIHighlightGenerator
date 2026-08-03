"""Описания медиа: то, что возвращает источник, и то, что получает конвейер."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.domain.enums import SourceKind
from core.domain.values import TimeRange

__all__ = [
    "AudioTrack",
    "LocalMedia",
    "MediaInfo",
    "SourceDescriptor",
    "Transcript",
    "TranscriptSegment",
    "TranscriptWord",
]


@dataclass(frozen=True, slots=True)
class MediaInfo:
    """Технические параметры медиа — по сути разобранный вывод ``ffprobe``."""

    duration_sec: float
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    video_codec: str | None = None
    audio_codec: str | None = None
    container: str | None = None
    size_bytes: int | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_audio(self) -> bool:
        return self.audio_codec is not None

    @property
    def has_video(self) -> bool:
        return self.video_codec is not None

    @property
    def aspect_ratio(self) -> float | None:
        if self.width and self.height:
            return self.width / self.height
        return None


@dataclass(frozen=True, slots=True)
class SourceDescriptor:
    """Что источник знает о медиа ДО скачивания.

    ``is_live`` и ``supports_seek`` — единственные два флага, которые
    отличают файл от стрима на уровне конвейера. Всё остальное одинаково.
    """

    uri: str
    kind: SourceKind
    title: str | None = None
    info: MediaInfo | None = None
    is_live: bool = False
    supports_seek: bool = True


@dataclass(frozen=True, slots=True)
class LocalMedia:
    """Медиа, физически доступное на диске воркера.

    Для локального файла — сам файл. Для стрима — скачанный временный сегмент.
    ``owned`` говорит, обязан ли провайдер удалить файл в ``release()``:
    исходник пользователя удалять нельзя, временный сегмент — нужно.
    """

    path: Path
    info: MediaInfo
    source_uri: str
    owned: bool = False

    @property
    def full_range(self) -> TimeRange:
        return TimeRange(0.0, self.info.duration_sec)


@dataclass(frozen=True, slots=True)
class AudioTrack:
    """Извлечённая аудиодорожка (WAV 16 kHz mono для Whisper)."""

    path: Path
    sample_rate: int
    channels: int
    duration_sec: float


@dataclass(frozen=True, slots=True)
class TranscriptWord:
    """Одно слово с таймингами.

    Пословные тайминги — не роскошь: без них невозможны ни караоке-субтитры
    (Этап 9), ни точный снап границ клипа к паузе в речи (Этап 7).
    """

    text: str
    start: float
    end: float
    probability: float = 1.0


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    text: str
    range: TimeRange
    words: Sequence[TranscriptWord] = field(default_factory=tuple)
    language: str | None = None
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    speaker: str | None = None  # задел под диаризацию


@dataclass(frozen=True, slots=True)
class Transcript:
    segments: Sequence[TranscriptSegment] = field(default_factory=tuple)
    language: str | None = None

    @property
    def is_empty(self) -> bool:
        return not self.segments

    @property
    def text(self) -> str:
        return " ".join(segment.text.strip() for segment in self.segments).strip()

    def within(self, window: TimeRange) -> list[TranscriptSegment]:
        """Сегменты, пересекающиеся с окном. Так текст привязывается к сцене."""
        return [segment for segment in self.segments if segment.range.overlaps(window)]

    def text_within(self, window: TimeRange) -> str:
        """Текст окна одной строкой.

        Пересечение, а не вложенность: реплика может начаться в одной сцене
        и закончиться в другой, и обрезать её по границе склейки нельзя —
        обе сцены должны знать, о чём в них говорят. Небольшое дублирование
        текста на стыке дешевле потерянной фразы.
        """
        return " ".join(segment.text.strip() for segment in self.within(window)).strip()

    def words_within(self, window: TimeRange) -> list[TranscriptWord]:
        return [
            word
            for segment in self.segments
            for word in segment.words
            if window.contains(word.start)
        ]
