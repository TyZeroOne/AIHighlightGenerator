"""Клипы джоба и скачивание их файлов.

Появилось вместе с Этапом 8: до него клипы существовали только как записи
о том, что стоит вырезать, и показывались рядом со сценами
(``GET /jobs/{id}/scenes``). Теперь у них есть файлы, и им нужен свой
эндпоинт — со статусом сборки и ссылками.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, Response, status
from fastapi.responses import RedirectResponse, StreamingResponse

from api.deps import ContainerDep, CurrentUser
from api.schemas.clips import ClipItemResponse, ClipListResponse
from core.domain.enums import AssetOwnerType
from core.domain.errors import NotFoundError, ValidationError
from core.ports.repository import AssetRecord, UnitOfWork

__all__ = ["router"]

router = APIRouter(prefix="/api/v1", tags=["clips"])

#: Расширение файла по типу содержимого. Нужно только для имени в
#: ``Content-Disposition``: браузер выбирает приложение по нему, а не
#: по MIME, и файл без расширения открывается «чем-нибудь».
_EXTENSIONS: dict[str, str] = {
    "video/mp4": ".mp4",
    "image/jpeg": ".jpg",
    "application/x-subrip": ".srt",
    "text/x-ssa": ".ass",
    "text/vtt": ".vtt",
}


@router.get(
    "/clips/{job_id}",
    response_model=ClipListResponse,
    summary="Клипы джоба с файлами",
    responses={404: {"description": "Джоб не найден или принадлежит другому пользователю"}},
)
async def list_clips(
    container: ContainerDep,
    user: CurrentUser,
    job_id: UUID,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    category: Annotated[str | None, Query(description="Фильтр по категории")] = None,
    min_score: Annotated[float | None, Query(ge=0, le=100)] = None,
) -> ClipListResponse:
    """Список клипов со ссылками на файлы.

    Незавершённый джоб — это ``200`` с пустым списком, а не ``404``:
    джоб существует, результата пока нет, и клиенту важно различать
    эти два случая.
    """
    async with container.uow() as uow:
        job = await uow.jobs.get(job_id)
        # 404, а не 403: существование чужого джоба — тоже информация.
        if job is None or job.user_id != user.id:
            raise NotFoundError("Джоб не найден", job_id=str(job_id))

        clips = list(await uow.clips.list_for_job(job_id))
        selected = [
            clip
            for clip in clips
            if (category is None or clip.category == category)
            and (min_score is None or clip.score >= min_score)
        ]
        window = selected[offset : offset + limit]
        assets = {
            clip.id: list(await uow.assets.list_for_owner(AssetOwnerType.CLIP, clip.id))
            for clip in window
        }

    return ClipListResponse(
        job_id=job_id,
        status=job.status.value,
        total=len(selected),
        items=[ClipItemResponse.from_domain(clip, assets.get(clip.id, [])) for clip in window],
    )


@router.get(
    "/download/{asset_id}",
    summary="Скачать файл клипа",
    response_class=StreamingResponse,
    responses={404: {"description": "Файл не найден или принадлежит другому пользователю"}},
)
async def download_asset(
    container: ContainerDep,
    user: CurrentUser,
    asset_id: UUID,
    inline: Annotated[bool, Query(description="Открыть в браузере, а не скачать")] = False,
) -> Response:
    """Отдача артефакта — та же развилка, что и у исходного видео.

    S3 отдаёт presigned-ссылку, и байты идут мимо нашего процесса;
    локальное хранилище прямых ссылок не умеет, поэтому файл течёт
    через API.
    """
    async with container.uow() as uow:
        asset = await uow.assets.get(asset_id)
        if asset is None or not await _owned_by(uow, asset, user.id):
            raise NotFoundError("Файл не найден", asset_id=str(asset_id))
        key, mime, size = asset.storage_key, asset.mime, asset.size_bytes

    signed = await container.storage.signed_url(key)
    if signed is not None:
        return RedirectResponse(signed, status_code=status.HTTP_302_FOUND)

    disposition = "inline" if inline else "attachment"
    filename = f"{asset_id}{_EXTENSIONS.get(mime, '')}"
    return StreamingResponse(
        container.storage.open_stream(key),
        media_type=mime,
        headers={
            "Content-Disposition": f'{disposition}; filename="{filename}"',
            "Content-Length": str(size or 0),
            # Заголовок заявляет поддержку перемотки, но обработки Range
            # здесь пока нет — как и у исходного видео. Полноценная
            # частичная отдача относится к работам Этапа 10.
            "Accept-Ranges": "bytes",
        },
    )


async def _owned_by(uow: UnitOfWork, asset: AssetRecord, user_id: UUID) -> bool:
    """Дошагать от артефакта до его владельца.

    Прав на файл у артефакта нет — они есть у сущности, которой он
    принадлежит, и у видео за ней. Поэтому проверка идёт по цепочке,
    а не по полю в самом артефакте: поле пришлось бы поддерживать
    в актуальном состоянии, а цепочка верна по построению.
    """
    owner_type = getattr(asset, "owner_type", None)
    owner_id = getattr(asset, "owner_id", None)
    if owner_id is None:
        return False

    if owner_type is AssetOwnerType.CLIP:
        clip = await uow.clips.get(owner_id)
        if clip is None:
            return False
        job = await uow.jobs.get(clip.job_id)
        return job is not None and job.user_id == user_id

    if owner_type is AssetOwnerType.JOB:
        job = await uow.jobs.get(owner_id)
        return job is not None and job.user_id == user_id

    if owner_type is AssetOwnerType.VIDEO:
        video = await uow.videos.get(owner_id)
        return video is not None and video.user_id == user_id

    raise ValidationError("Неизвестный тип владельца артефакта", owner_type=str(owner_type))
