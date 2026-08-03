"""Служебные эндпоинты: что за плагины загружены и в каком порядке пойдут.

Не украшение: когда на Этапе 5 анализаторов станет полтора десятка, вопрос
«почему этот признак не посчитался» решается взглядом на ``/system/plan``,
а не чтением логов воркера.
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel

from core import registry
from core.container import get_container
from core.domain.analysis import ARTIFACT_AUDIO, ARTIFACT_TRANSCRIPT, ARTIFACT_VIDEO

__all__ = ["router"]

router = APIRouter(prefix="/api/v1/system", tags=["system"])


class AnalyzerInfo(BaseModel):
    key: str
    requires: list[str]
    produces: list[str]
    cost: int
    enabled: bool


class PlanLayer(BaseModel):
    layer: int
    analyzers: list[str]


@router.get(
    "/analyzers",
    response_model=list[AnalyzerInfo],
    summary="Зарегистрированные анализаторы",
)
async def list_analyzers() -> list[AnalyzerInfo]:
    disabled = get_container().settings.disabled_analyzers_set
    return [
        AnalyzerInfo(
            key=key,
            requires=sorted(cls.requires),
            produces=sorted(cls.produces),
            cost=cls.cost,
            enabled=key not in disabled,
        )
        for key, cls in sorted(registry.get_analyzers().items())
    ]


@router.get("/sources", response_model=list[str], summary="Зарегистрированные источники видео")
async def list_sources() -> list[str]:
    return sorted(registry.get_source_providers())


@router.get("/plan", response_model=list[PlanLayer], summary="План анализа по слоям")
async def analysis_plan(
    with_audio: bool = Query(default=True, description="Считать, что аудио доступно"),
    with_transcript: bool = Query(default=True, description="Считать, что транскрипт доступен"),
) -> list[PlanLayer]:
    artifacts = {ARTIFACT_VIDEO}
    if with_audio:
        artifacts.add(ARTIFACT_AUDIO)
    if with_transcript:
        artifacts.add(ARTIFACT_TRANSCRIPT)

    layers = registry.build_plan(
        available_artifacts=artifacts,
        disabled=get_container().settings.disabled_analyzers_set,
    )
    return [
        PlanLayer(layer=i, analyzers=[cls.key for cls in layer]) for i, layer in enumerate(layers)
    ]
