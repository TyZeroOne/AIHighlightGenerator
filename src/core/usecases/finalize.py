"""Этап ``finalize``: манифест и итоговый статус джоба.

Последний этап конвейера. Делает три вещи, каждая из которых мала,
но без каждой результат неполон.

**Манифест.** Один JSON со всем, что получилось: видео, клипы, файлы,
заголовки, расход. Уезжает в хранилище рядом с клипами и живёт дольше
записей в базе.

**Итоговый статус.** До этого момента джоб всё время «выполняется».
Здесь он становится ``succeeded`` — или ``partial``, если что-то по пути
деградировало: не собрался клип, отвалилась модель, пропустился этап
анализа. Разница важна: ``partial`` означает «результат есть, но не весь»,
и по нему видно, что стоит перезапустить.

**Проверка, что результат вообще есть.** Джоб без единого готового клипа
завершается ``failed``, даже если формально все этапы отработали:
пустая папка — это не успех.

Уборка временных артефактов (аудио, кадры) сюда пока не входит — она
требует политики хранения, а та относится к работам Этапа 10.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import UUID

from core.domain.entities import Asset, Clip, Job, Video
from core.domain.enums import AssetKind, AssetOwnerType, ClipStatus, JobStage, JobStatus
from core.domain.keys import manifest_key
from core.logging import get_logger
from core.ports.export import ManifestBuilder
from core.ports.repository import UnitOfWorkFactory
from core.ports.storage import ObjectStorage
from core.usecases.stages import StageTracker

__all__ = ["FinalizeJob", "FinalizeResult", "final_status"]

log = get_logger(__name__)

STAGE = JobStage.FINALIZE

MANIFEST_MIME = "application/json"


@dataclass(frozen=True, slots=True)
class FinalizeResult:
    job_id: UUID
    status: JobStatus
    clips_ready: int
    clips_failed: int
    manifest_key: str | None
    duration_sec: float
    reused: bool = False


class FinalizeJob:
    """Сценарий завершающего этапа."""

    def __init__(
        self,
        *,
        uow_factory: UnitOfWorkFactory,
        storage: ObjectStorage,
        manifest: ManifestBuilder,
        workdir: Path,
    ) -> None:
        self._uow_factory = uow_factory
        self._storage = storage
        self._manifest = manifest
        self._workdir = workdir
        self._tracker = StageTracker(uow_factory=uow_factory, stage=STAGE)

    async def execute(self, job_id: UUID) -> FinalizeResult:
        started = time.monotonic()
        job, video = await self._tracker.load(job_id)

        if job.is_stage_done(STAGE):
            return FinalizeResult(job_id, job.status, 0, 0, None, 0.0, reused=True)

        await self._tracker.begin(job)

        try:
            async with self._uow_factory() as uow:
                # Джоб перечитывается: между началом сценария и этой строкой
                # предыдущие этапы дописали расход, и манифест обязан
                # показать итоговые числа, а не те, что были при загрузке.
                fresh = await uow.jobs.get(job_id) or job
                clips = list(await uow.clips.list_for_job(job_id))
                assets = {
                    clip.id: list(await uow.assets.list_for_owner(AssetOwnerType.CLIP, clip.id))
                    for clip in clips
                }

            status = final_status(job, clips)
            # Статус подставляется в манифест ДО записи. Первый живой прогон
            # писал туда ``running``: манифест собирался раньше, чем джоб
            # получал итоговое состояние, и файл уезжал в хранилище
            # с заведомо неверным полем.
            key = await self._write_manifest(replace(fresh, status=status), video, clips, assets)
        except Exception as error:
            await self._tracker.fail(job, error, duration_sec=_elapsed(started))
            raise

        ready = sum(1 for clip in clips if clip.status is ClipStatus.READY)
        failed = sum(1 for clip in clips if clip.status is ClipStatus.FAILED)

        # Сначала штатная отметка этапа, потом итоговый статус: отметка
        # выставляет ``running``, и обратный порядок затёр бы результат.
        await self._tracker.succeed(job, duration_sec=_elapsed(started))
        await self._set_status(job_id, status)

        log.info(
            "finalize.done",
            job_id=str(job_id),
            status=status.value,
            clips_ready=ready,
            clips_failed=failed,
        )
        return FinalizeResult(
            job_id=job_id,
            status=status,
            clips_ready=ready,
            clips_failed=failed,
            manifest_key=key,
            duration_sec=_elapsed(started),
        )

    # --------------------------------------------------------------- шаги
    async def _write_manifest(
        self,
        job: Job,
        video: Video,
        clips: Sequence[Clip],
        assets: Mapping[UUID, Sequence[Asset]],
    ) -> str:
        payload = self._manifest.build(job, video, clips, assets)
        target = self._workdir / "manifests" / f"{job.id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        # ``ensure_ascii=False``: манифест читают люди, и заголовки клипов
        # в нём должны оставаться словами, а не последовательностью \\uXXXX.
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
        )

        key = manifest_key(job.id)
        await self._storage.put(key, target, content_type=MANIFEST_MIME)
        target.unlink(missing_ok=True)

        async with self._uow_factory() as uow:
            await uow.assets.delete_for_owner(AssetOwnerType.JOB, job.id)
            await uow.assets.add(
                owner_type=AssetOwnerType.JOB,
                owner_id=job.id,
                kind=AssetKind.MANIFEST,
                storage_key=key,
                mime=MANIFEST_MIME,
            )
            await uow.commit()
        return key

    async def _set_status(self, job_id: UUID, status: JobStatus) -> None:
        async with self._uow_factory() as uow:
            current = await uow.jobs.get(job_id)
            if current is None:  # pragma: no cover — джоб прочитан выше
                return
            await uow.jobs.update(replace(current, status=status))
            await uow.commit()


def final_status(job: Job, clips: Sequence[Clip]) -> JobStatus:
    """Чем закончился джоб.

    Чистая функция от состояния: по ней видно правило целиком, и проверить
    его можно без базы.

    Порядок проверок — от худшего к лучшему. Ни одного готового клипа —
    провал независимо от того, как отработали этапы. Есть готовые, но
    что-то по дороге упало или пропустилось — ``partial``. Всё прошло —
    ``succeeded``.
    """
    ready = [clip for clip in clips if clip.status is ClipStatus.READY]
    if not ready:
        return JobStatus.FAILED

    degraded = any(
        state.status in {"failed", "skipped"}
        for stage, state in job.stages.items()
        if stage is not STAGE
    )
    partial_clips = any(clip.status is ClipStatus.FAILED for clip in clips)
    return JobStatus.PARTIAL if degraded or partial_clips else JobStatus.SUCCEEDED


def _elapsed(started: float) -> float:
    return round(time.monotonic() - started, 3)
