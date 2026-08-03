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
