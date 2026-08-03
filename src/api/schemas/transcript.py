"""Схемы транскрипта.

Тайм-коды отдаются и числом, и строкой ``HH:MM:SS.mmm`` (док 07, §7.2):
числа удобны машине, строки — человеку и глазу при отладке. Считать одно
из другого на стороне каждого клиента дороже, чем отдать оба.
"""

from __future__ import annotations

from typing import Self
from uuid import UUID

from pydantic import BaseModel

from core.domain.media import Transcript, TranscriptSegment
from core.domain.timecode import format_timecode

__all__ = ["TranscriptResponse", "TranscriptSegmentResponse", "TranscriptWordResponse"]


class TranscriptWordResponse(BaseModel):
    text: str
    start_sec: float
    end_sec: float
    probability: float


class TranscriptSegmentResponse(BaseModel):
    start: str
    end: str
    start_sec: float
    end_sec: float
    text: str
    language: str | None = None
    no_speech_prob: float | None = None
    words: list[TranscriptWordResponse] = []

    @classmethod
    def from_domain(cls, segment: TranscriptSegment, *, with_words: bool = True) -> Self:
        return cls(
            start=format_timecode(segment.range.start),
            end=format_timecode(segment.range.end),
            start_sec=round(segment.range.start, 3),
            end_sec=round(segment.range.end, 3),
            text=segment.text,
            language=segment.language,
            no_speech_prob=segment.no_speech_prob,
            words=[
                TranscriptWordResponse(
                    text=word.text,
                    start_sec=round(word.start, 3),
                    end_sec=round(word.end, 3),
                    probability=round(word.probability, 4),
                )
                for word in segment.words
            ]
            if with_words
            else [],
        )


class TranscriptResponse(BaseModel):
    job_id: UUID
    language: str | None = None
    segment_count: int
    word_count: int
    duration_sec: float
    segments: list[TranscriptSegmentResponse]

    @classmethod
    def from_domain(
        cls,
        job_id: UUID,
        transcript: Transcript,
        *,
        with_words: bool = True,
    ) -> Self:
        segments = list(transcript.segments)
        return cls(
            job_id=job_id,
            language=transcript.language,
            segment_count=len(segments),
            word_count=sum(len(segment.words) for segment in segments),
            duration_sec=round(segments[-1].range.end, 3) if segments else 0.0,
            segments=[
                TranscriptSegmentResponse.from_domain(segment, with_words=with_words)
                for segment in segments
            ],
        )
