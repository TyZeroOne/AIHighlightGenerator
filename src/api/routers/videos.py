"""Эндпоинты работы с видео."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, File, Query, Request, Response, UploadFile, status
from fastapi.responses import RedirectResponse, StreamingResponse

from api.deps import ContainerDep, CurrentUser
from api.schemas.video import (
    RegisterLocalRequest,
    VideoListResponse,
    VideoResponse,
)
from core.domain.errors import NotFoundError
from core.domain.validation import validate_upload_name
from core.usecases.upload_video import mime_for, stream_to_file

__all__ = ["router"]

router = APIRouter(prefix="/api/v1", tags=["videos"])

READ_CHUNK = 1 << 20  # 1 МиБ


async def _file_chunks(upload: UploadFile, size: int = READ_CHUNK) -> AsyncIterator[bytes]:
    while chunk := await upload.read(size):
        yield chunk


@router.post(
    "/upload",
    response_model=VideoResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Загрузить видеофайл",
    responses={
        200: {"description": "Файл уже был загружен ранее — возвращена существующая запись"},
        413: {"description": "Файл превышает лимит размера"},
        415: {"description": "Контейнер не поддерживается"},
    },
)
async def upload_video(
    container: ContainerDep,
    user: CurrentUser,
    response: Response,
    file: Annotated[UploadFile, File(description="Видеофайл mp4/mkv/mov")],
) -> VideoResponse:
    settings = container.settings

    # Проверка имени — до чтения байтов. Отказать по расширению раньше,
    # чем принять 20 ГБ, дешевле для всех.
    validate_upload_name(file.filename or "", settings.allowed_containers_set)

    temp_path = container.workdir / "incoming" / f"{uuid4()}{Path(file.filename or '').suffix}"
    stored = await stream_to_file(
        _file_chunks(file),
        temp_path,
        max_bytes=settings.max_upload_bytes,
    )

    try:
        video, duplicate = await container.upload_video.execute(
            user_id=user.id,
            upload=stored,
            filename=file.filename or temp_path.name,
        )
    finally:
        # Временный файл больше не нужен ни при успехе, ни при ошибке:
        # в хранилище он уже скопирован, а при сбое он и подавно лишний.
        stored.path.unlink(missing_ok=True)

    # 200 вместо 201 на дубликате: ничего не создано, и клиент должен
    # иметь возможность это различить, не разбирая тело ответа.
    if duplicate:
        response.status_code = status.HTTP_200_OK
    return VideoResponse.from_domain(video, duplicate=duplicate)


@router.post(
    "/videos/register",
    response_model=VideoResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Зарегистрировать файл на смонтированном томе",
    description=(
        "Байты не копируются — в БД сохраняется ссылка на файл. "
        "Работает только для каталогов из LOCAL_SOURCE_ROOTS_RAW."
    ),
)
async def register_local_video(
    container: ContainerDep,
    user: CurrentUser,
    payload: RegisterLocalRequest,
    response: Response,
) -> VideoResponse:
    video, duplicate = await container.register_local_video.execute(
        user_id=user.id,
        source_uri=payload.source_uri,
    )
    if duplicate:
        response.status_code = status.HTTP_200_OK
    return VideoResponse.from_domain(video, duplicate=duplicate)


@router.get("/videos", response_model=VideoListResponse, summary="Список видео")
async def list_videos(
    container: ContainerDep,
    user: CurrentUser,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> VideoListResponse:
    items, total = await container.list_videos.execute(user_id=user.id, limit=limit, offset=offset)
    return VideoListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=[VideoResponse.from_domain(video) for video in items],
    )


@router.get("/videos/{video_id}", response_model=VideoResponse, summary="Метаданные видео")
async def get_video(
    container: ContainerDep,
    user: CurrentUser,
    video_id: UUID,
) -> VideoResponse:
    video = await container.get_video.execute(user_id=user.id, video_id=video_id)
    return VideoResponse.from_domain(video)


@router.delete(
    "/videos/{video_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Удалить видео",
    description="Запись помечается удалённой, файлы удаляются из хранилища.",
)
async def delete_video(
    container: ContainerDep,
    user: CurrentUser,
    video_id: UUID,
) -> Response:
    await container.delete_video.execute(user_id=user.id, video_id=video_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/videos/{video_id}/download",
    summary="Скачать исходное видео",
    response_class=StreamingResponse,
)
async def download_video(
    container: ContainerDep,
    user: CurrentUser,
    video_id: UUID,
    request: Request,
) -> Response:
    """Отдача исходника.

    Два пути в зависимости от хранилища:

    * S3 — редирект на presigned-ссылку, байты идут мимо нашего процесса;
    * локальное хранилище — потоковая отдача, потому что прямых ссылок нет.

    Роутер сам не знает, какое хранилище настроено: он спрашивает
    ``signed_url()`` и действует по ответу.
    """
    video = await container.get_video.execute(user_id=user.id, video_id=video_id)
    if not video.storage_key:
        raise NotFoundError(
            "У видео нет файла в хранилище (зарегистрирован локальный источник)",
            video_id=str(video_id),
        )

    signed = await container.storage.signed_url(video.storage_key)
    if signed is not None:
        return RedirectResponse(signed, status_code=status.HTTP_302_FOUND)

    filename = video.original_filename or f"{video_id}.mp4"
    extension = Path(filename).suffix
    return StreamingResponse(
        container.storage.open_stream(video.storage_key),
        media_type=mime_for(extension),
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(video.size_bytes or 0),
            # Заявляем поддержку Range, чтобы плееры показывали перемотку.
            # Полноценная обработка Range — Этап 10 вместе с отдачей клипов.
            "Accept-Ranges": "bytes",
        },
    )
