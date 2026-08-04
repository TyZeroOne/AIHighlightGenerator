"""Эндпоинты джобов: запуск конвейера, состояние, повтор, отмена.

Роутер тонкий: разобрать запрос, позвать сценарий, превратить результат
в схему. Ни одной строчки логики здесь нет и быть не должно — те же
операции вызывает командная строка, и дублировать их в двух местах
значит рано или поздно получить два разных поведения.

Проверка владения — в сценариях, а не здесь, и по той же причине.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any, Literal
from uuid import UUID

from fastapi import APIRouter, Query, Request, Response, status
from fastapi.responses import PlainTextResponse, StreamingResponse

from api.deps import ContainerDep, CurrentUser
from api.ratelimit import rate_limit
from api.schemas.jobs import (
    AnalyzeRequest,
    JobListResponse,
    JobResponse,
    JobStatusResponse,
    RetryRequest,
)
from api.schemas.scenes import ClipResponse, SceneListResponse, SceneResponse
from api.schemas.transcript import TranscriptResponse
from api.sse import SSE_HEADERS, stream_events
from core.domain.enums import JobStatus
from core.domain.errors import NotFoundError
from subtitles.formats import to_srt, to_text, to_vtt

__all__ = ["router"]

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"], dependencies=[rate_limit()])

TranscriptFormat = Literal["json", "srt", "vtt", "txt"]

_MEDIA_TYPES: dict[str, str] = {
    "srt": "application/x-subrip",
    "vtt": "text/vtt",
    "txt": "text/plain",
}


@router.post(
    "",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Запустить анализ видео",
    responses={
        200: {"description": "Такой джоб уже посчитан — возвращён он же"},
        404: {"description": "Видео не найдено или принадлежит другому пользователю"},
        422: {"description": "Видео не готово к анализу или параметры противоречивы"},
    },
)
async def start_analysis(
    container: ContainerDep,
    user: CurrentUser,
    request: AnalyzeRequest,
    response: Response,
) -> JobResponse:
    """Поставить конвейер в очередь.

    Идемпотентно: повторный запрос с теми же параметрами возвращает уже
    посчитанный джоб и код ``200`` вместо ``202``. Разные коды — не
    педантизм: по ним клиент понимает, ждать ему результата или он уже
    готов, не разбирая тело ответа.
    """
    view = await container.start_analysis.execute(
        user_id=user.id,
        video_id=request.video_id,
        params=request.to_params(),
        force=request.force,
    )
    if view.reused:
        response.status_code = status.HTTP_200_OK
    return JobResponse.from_domain(view.job, reused=view.reused)


@router.get(
    "",
    response_model=JobListResponse,
    summary="Джобы пользователя",
)
async def list_jobs(
    container: ContainerDep,
    user: CurrentUser,
    job_status: Annotated[JobStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> JobListResponse:
    jobs = await container.list_jobs.execute(
        user_id=user.id, status=job_status, limit=limit, offset=offset
    )
    return JobListResponse(
        total=len(jobs),
        items=[JobResponse.from_domain(job) for job in jobs],
    )


@router.get(
    "/{job_id}/status",
    response_model=JobStatusResponse,
    summary="Состояние джоба по этапам",
    responses={404: {"description": "Джоб не найден или принадлежит другому пользователю"}},
)
async def get_status(
    container: ContainerDep,
    user: CurrentUser,
    job_id: UUID,
) -> JobStatusResponse:
    """Подробное состояние: по этому ответу клиент рисует прогресс.

    Счётчики сцен и клипов идут в том же ответе: без них «41 %» ничего
    не значит, а второй запрос ради двух чисел — лишний круг.
    """
    async with container.uow() as uow:
        job = await uow.jobs.get(job_id)
        if job is None or job.user_id != user.id:
            raise NotFoundError("Джоб не найден", job_id=str(job_id))
        scenes = await uow.scenes.count_for_job(job_id)
        clips = len(await uow.clips.list_for_job(job_id))

    return JobStatusResponse.from_domain(job, scenes=scenes, clips=clips)


@router.get(
    "/{job_id}/status/stream",
    summary="Поток прогресса (Server-Sent Events)",
    response_class=StreamingResponse,
    responses={404: {"description": "Джоб не найден или принадлежит другому пользователю"}},
)
async def stream_status(
    container: ContainerDep,
    user: CurrentUser,
    job_id: UUID,
    request: Request,
) -> StreamingResponse:
    """Событие на каждое изменение этапа или прогресса от одного процента.

    Порог в процент — не экономия байтов. Прогресс внутри длинного этапа
    меняется на сотые доли, и без порога поток превращается в сплошной
    шум, в котором смена этапа теряется.
    """

    async def snapshot() -> dict[str, Any] | None:
        async with container.uow() as uow:
            job = await uow.jobs.get(job_id)
            if job is None or job.user_id != user.id:
                return None
            return {
                "job_id": str(job.id),
                "status": job.status.value,
                "stage": job.stage.value if job.stage else None,
                "progress": job.progress,
            }

    # Первая проверка — обычным запросом: несуществующий джоб должен
    # дать честный 404, а не поток с ошибкой внутри.
    if await snapshot() is None:
        raise NotFoundError("Джоб не найден", job_id=str(job_id))

    return StreamingResponse(
        stream_events(
            snapshot=snapshot,
            is_disconnected=request.is_disconnected,
            sleep=asyncio.sleep,
            changed=_progress_changed,
            is_final=lambda state: JobStatus(state["status"]).is_terminal,
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


def _progress_changed(previous: dict[str, Any] | None, current: dict[str, Any]) -> bool:
    if previous is None:
        return True
    if previous["stage"] != current["stage"] or previous["status"] != current["status"]:
        return True
    return abs(float(current["progress"]) - float(previous["progress"])) >= 1.0


@router.post(
    "/{job_id}/retry",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Перезапустить джоб с этапа",
    responses={404: {"description": "Джоб не найден или принадлежит другому пользователю"}},
)
async def retry_job(
    container: ContainerDep,
    user: CurrentUser,
    job_id: UUID,
    request: RetryRequest | None = None,
) -> JobResponse:
    """Повтор с указанного этапа; без указания — с первого непройденного.

    Предыдущие этапы не пересчитываются: их состояния остаются
    ``succeeded``, и задачи выходят немедленно. Перезапуск из-за
    отвалившейся модели не стоит двадцати минут разбора кадров.
    """
    view = await container.retry_job.execute(
        user_id=user.id,
        job_id=job_id,
        from_stage=request.from_stage if request else None,
    )
    return JobResponse.from_domain(view.job)


@router.delete(
    "/{job_id}",
    response_model=JobResponse,
    summary="Отменить выполняющийся джоб",
    responses={
        404: {"description": "Джоб не найден или принадлежит другому пользователю"},
        422: {"description": "Джоб уже завершён"},
    },
)
async def cancel_job(
    container: ContainerDep,
    user: CurrentUser,
    job_id: UUID,
) -> JobResponse:
    """Остановить конвейер.

    Гарантируется, что новые этапы не начнутся: статус пишется до отзыва
    задачи, а каждый этап проверяет его на входе. Уже запущенный ffmpeg
    получает сигнал вместе с задачей и умирает вместе с ней.
    """
    view = await container.cancel_job.execute(user_id=user.id, job_id=job_id)
    return JobResponse.from_domain(view.job)


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
