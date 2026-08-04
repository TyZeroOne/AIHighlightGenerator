"""Схемы ответа по сценам и клипам джоба.

Отдельные схемы, а не выдача доменных объектов как есть. Причина обычная
для границы API: наружу уходит не всё, что лежит внутри. Сорок сырых
признаков на сцену клиенту не нужны и раздули бы ответ на порядок,
а вот разложение скора по группам нужно — по нему видно, **почему**
момент оказался вверху списка.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from uuid import UUID

from pydantic import BaseModel, Field

from core.domain.entities import Clip, Scene
from core.domain.timecode import format_timecode

__all__ = ["ClipResponse", "SceneListResponse", "SceneResponse"]


class SceneResponse(BaseModel):
    index: int
    start: str
    end: str
    #: Дублируем время числом: тайм-код удобен человеку, секунды — коду.
    #: Заставлять клиента парсить «00:01:23.480» ради арифметики незачем.
    start_sec: float
    end_sec: float
    duration_sec: float
    heuristic_score: float | None = None
    llm_score: float | None = None
    final_score: float | None = None
    category: str | None = None
    reason: str | None = None
    #: Вклад групп признаков — то самое «почему 90».
    breakdown: Mapping[str, float] = Field(default_factory=dict)
    #: Составные сигналы (``signal.*``). Сырые признаки наружу не отдаются:
    #: их два десятка, и без нормализации они всё равно ни о чём не говорят.
    signals: Mapping[str, float] = Field(default_factory=dict)
    has_speech: bool = False

    @classmethod
    def from_domain(cls, scene: Scene) -> SceneResponse:
        return cls(
            index=scene.index,
            start=format_timecode(scene.range.start),
            end=format_timecode(scene.range.end),
            start_sec=round(scene.range.start, 3),
            end_sec=round(scene.range.end, 3),
            duration_sec=round(scene.range.duration, 3),
            heuristic_score=scene.heuristic_score.value if scene.heuristic_score else None,
            llm_score=scene.llm_score,
            final_score=scene.final_score,
            category=scene.category,
            reason=scene.reason,
            breakdown=dict(scene.heuristic_score.breakdown) if scene.heuristic_score else {},
            signals={
                key.removeprefix("signal."): value.value
                for key, value in scene.features.items()
                if key.startswith("signal.")
            },
            has_speech=bool((scene.transcript_text or "").strip()),
        )


class ClipResponse(BaseModel):
    rank: int
    start: str
    end: str
    start_sec: float
    end_sec: float
    duration_sec: float
    score: float
    category: str
    reason: str
    scene_indices: Sequence[int] = ()

    @classmethod
    def from_domain(cls, clip: Clip, scene_indices: Sequence[int] = ()) -> ClipResponse:
        return cls(
            rank=clip.rank,
            start=format_timecode(clip.range.start),
            end=format_timecode(clip.range.end),
            start_sec=round(clip.range.start, 3),
            end_sec=round(clip.range.end, 3),
            duration_sec=round(clip.range.duration, 3),
            score=clip.score,
            category=clip.category,
            reason=clip.reason,
            scene_indices=list(scene_indices),
        )


class SceneListResponse(BaseModel):
    job_id: UUID
    total: int
    scenes: Sequence[SceneResponse]
    #: Отобранные моменты в том же ответе. Их всегда хотят рядом со сценами:
    #: вопрос «почему этот клип, а не соседний» без сцен не разобрать.
    clips: Sequence[ClipResponse] = ()
