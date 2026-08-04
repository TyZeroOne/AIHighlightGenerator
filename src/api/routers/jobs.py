"""Эндпоинты джобов.

Пока здесь только транскрипт: он появился вместе с распознаванием речи
(Этап 4) и нужен уже сейчас — иначе результат этапа виден только в базе.
Остальные эндпоинты джобов (`POST /analyze`, `/status`, `/retry`, отмена)
приедут на Этапе 10 вместе с квотами и rate limiting.
"""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Query
from fastapi.responses import PlainTextResponse

from api.deps import ContainerDep, CurrentUser
from api.schemas.scenes import ClipResponse, SceneListResponse, SceneResponse
from api.schemas.transcript import TranscriptResponse
from core.domain.errors import NotFoundError
from subtitles.formats import to_srt, to_text, to_vtt

__all__ = ["router"]

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])

TranscriptFormat = Literal["json", "srt", "vtt", "txt"]

_MEDIA_TYPES: dict[str, str] = {
    "srt": "application/x-subrip",
    "vtt": "text/vtt",
    "txt": "text/plain",
}


@router.get(
    "/{job_id}/transcript",
    summary="Транскрипт джоба",
    response_model=None,
    responses={
        200: {"description": "Транскрипт в выбранном формате"},
        404: {"description": "Джоб не найден или принадлежит другому пользователю"},
    },
)
async def get_transcript(
    container: ContainerDep,
    user: CurrentUser,
    job_id: UUID,
    fmt: Annotated[TranscriptFormat, Query(alias="format")] = "json",
    words: Annotated[bool, Query(description="Включать пословные тайминги")] = True,
) -> TranscriptResponse | PlainTextResponse:
    async with container.uow() as uow:
        job = await uow.jobs.get(job_id)
        # Проверка владения обязательна и здесь: транскрипт — это содержание
        # чужого видео, доступ к нему равносилен доступу к самому видео.
        # 404, а не 403: существование чужого джоба — тоже информация.
        if job is None or job.user_id != user.id:
            raise NotFoundError("Джоб не найден", job_id=str(job_id))
        transcript = await uow.transcripts.get_for_job(job_id)

    # Пустой транскрипт — это 200, а не 404: у видео без речи его законно нет,
    # и клиент должен увидеть «речи нет», а не «джоба не существует».
    if fmt == "json":
        return TranscriptResponse.from_domain(job_id, transcript, with_words=words)

    body = {"srt": to_srt, "vtt": to_vtt, "txt": to_text}[fmt](transcript)
    return PlainTextResponse(
        body,
        media_type=f"{_MEDIA_TYPES[fmt]}; charset=utf-8",
        headers={"Content-Disposition": f'inline; filename="{job_id}.{fmt}"'},
    )


@router.get(
    "/{job_id}/scenes",
    response_model=SceneListResponse,
    summary="Сцены джоба со скорами и отобранные клипы",
    responses={404: {"description": "Джоб не найден или принадлежит другому пользователю"}},
)
async def get_scenes(
    container: ContainerDep,
    user: CurrentUser,
    job_id: UUID,
    scored_only: Annotated[bool, Query(description="Только сцены с посчитанным скором")] = False,
    limit: Annotated[int, Query(ge=1, le=2000)] = 500,
) -> SceneListResponse:
    """Сцены, их скоры и отобранные из них клипы.

    Главный отладочный эндпоинт скоринга: в одном ответе видно и решение
    (клипы), и данные, на которых оно принято (скоры, разложение по группам,
    сигналы). Без этого вопрос «почему выбран этот момент, а не соседний»
    приходится разбирать запросами в базу.

    Лимит по умолчанию 500, а не «все»: у двухчасового фильма сцен полторы
    тысячи, и полный ответ — это мегабайты JSON, которые почти никогда
    не нужны целиком.
    """
    async with container.uow() as uow:
        job = await uow.jobs.get(job_id)
        # Та же проверка владения, что и у транскрипта: сцены — это
        # содержание чужого видео. 404, а не 403.
        if job is None or job.user_id != user.id:
            raise NotFoundError("Джоб не найден", job_id=str(job_id))
        scenes = list(await uow.scenes.list_for_job(job_id))
        clips = list(await uow.clips.list_for_job(job_id))

    by_id = {scene.id: scene.index for scene in scenes}
    selected = scenes
    if scored_only:
        selected = [scene for scene in scenes if scene.final_score is not None]

    return SceneListResponse(
        job_id=job_id,
        total=len(selected),
        scenes=[SceneResponse.from_domain(scene) for scene in selected[:limit]],
        clips=[
            ClipResponse.from_domain(
                clip,
                [by_id[scene_id] for scene_id in clip.scene_ids if scene_id in by_id],
            )
            for clip in clips
        ],
    )
