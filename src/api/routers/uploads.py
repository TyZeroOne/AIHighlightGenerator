"""Чанковая (возобновляемая) загрузка больших файлов.

Три шага: ``init`` → много раз ``part`` → ``complete``.
Обоснование механики — в ``core/usecases/chunked_upload.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Request, Response, status
from fastapi import Path as PathParam
from pydantic import BaseModel, Field

from api.deps import ContainerDep, CurrentUser
from api.ratelimit import rate_limit
from api.schemas.video import PartAccepted, UploadSessionResponse, VideoResponse
from core.usecases.chunked_upload import MAX_PARTS

__all__ = ["router"]

router = APIRouter(prefix="/api/v1/upload", tags=["uploads"], dependencies=[rate_limit()])


class InitUploadRequest(BaseModel):
    filename: str = Field(examples=["movie.mkv"])
    total_size: int | None = Field(
        default=None,
        ge=1,
        description="Полный размер файла, если известен. Позволяет отказать сразу.",
    )


async def _request_chunks(request: Request) -> AsyncIterator[bytes]:
    """Тело запроса как поток.

    ``request.stream()``, а не ``await request.body()``: второй вариант
    прочитал бы часть целиком в память. При 32 МиБ на часть и десяти
    параллельных загрузках это 320 МиБ на ровном месте.
    """
    async for chunk in request.stream():
        yield chunk


@router.post(
    "/init",
    response_model=UploadSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Начать чанковую загрузку",
)
async def init_upload(
    container: ContainerDep,
    user: CurrentUser,
    payload: InitUploadRequest,
) -> UploadSessionResponse:
    session = await container.init_chunked_upload.execute(
        user_id=user.id,
        filename=payload.filename,
        total_size=payload.total_size,
    )
    return UploadSessionResponse.from_domain(session)


@router.put(
    "/{upload_id}/part/{part_number}",
    response_model=PartAccepted,
    summary="Отправить часть",
    description=(
        "Тело запроса — сырые байты части (Content-Type: application/octet-stream). "
        "Части можно слать не по порядку и повторно: повторная отправка идемпотентна."
    ),
)
async def upload_part(
    container: ContainerDep,
    user: CurrentUser,
    request: Request,
    upload_id: UUID,
    part_number: Annotated[int, PathParam(ge=1, le=MAX_PARTS)],
) -> PartAccepted:
    session = await container.upload_part.execute(
        user_id=user.id,
        upload_id=upload_id,
        part_number=part_number,
        chunks=_request_chunks(request),
    )
    return PartAccepted(
        upload_id=session.id,
        part_number=part_number,
        received_parts=len(session.received_parts),
        received_bytes=session.received_bytes,
        next_expected_part=session.next_expected_part(),
    )


@router.get(
    "/{upload_id}",
    response_model=UploadSessionResponse,
    summary="Состояние загрузки",
    description="Клиент вызывает это после обрыва, чтобы понять, с какой части продолжать.",
)
async def upload_status(
    container: ContainerDep,
    user: CurrentUser,
    upload_id: UUID,
) -> UploadSessionResponse:
    from core.domain.errors import NotFoundError

    async with container.uow() as uow:
        session = await uow.uploads.get_owned(upload_id, user.id)
    if session is None:
        raise NotFoundError("Сессия загрузки не найдена", upload_id=str(upload_id))
    return UploadSessionResponse.from_domain(session)


@router.post(
    "/{upload_id}/complete",
    response_model=VideoResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Завершить загрузку и собрать файл",
    responses={
        200: {"description": "Такой файл уже был загружен ранее"},
        409: {"description": "Не хватает частей или загрузка уже завершена"},
    },
)
async def complete_upload(
    container: ContainerDep,
    user: CurrentUser,
    response: Response,
    upload_id: UUID,
) -> VideoResponse:
    video, duplicate = await container.complete_chunked_upload.execute(
        user_id=user.id,
        upload_id=upload_id,
    )
    if duplicate:
        response.status_code = status.HTTP_200_OK
    return VideoResponse.from_domain(video, duplicate=duplicate)
