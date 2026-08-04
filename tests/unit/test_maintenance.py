"""Периодическая уборка: истёкшие артефакты и зависшие джобы.

Код, который выполняется в три часа ночи без свидетелей, проверять потом
будет некому — поэтому здесь проверяется не «работает вообще», а те
свойства, из-за которых уборка может навредить: порядок удаления, что
происходит при недоступном хранилище, и не трогает ли она живое.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from core.domain.entities import Asset, Job
from core.domain.enums import AssetKind, AssetOwnerType, JobStatus
from core.usecases.maintenance import CleanupExpiredAssets, ReapStuckJobs

pytestmark = pytest.mark.unit


def make_asset(*, key: str = "jobs/x/audio.wav", expired: bool = True) -> Asset:
    return Asset(
        id=uuid4(),
        owner_type=AssetOwnerType.JOB,
        owner_id=uuid4(),
        kind=AssetKind.AUDIO_WAV,
        storage_key=key,
        expires_at=datetime.now(UTC) - timedelta(hours=1) if expired else None,
    )


def make_job(*, status: JobStatus = JobStatus.RUNNING) -> Job:
    return Job(id=uuid4(), video_id=uuid4(), user_id=uuid4(), status=status)


class FakeAssets:
    def __init__(self, assets: list[Asset]) -> None:
        self.assets = assets
        self.marked: list[UUID] = []

    async def expired(self, *, limit: int = 500) -> Sequence[Asset]:
        return self.assets[:limit]

    async def mark_deleted(self, asset_ids: Sequence[UUID]) -> None:
        self.marked.extend(asset_ids)


class FakeJobs:
    def __init__(self, jobs: list[Job]) -> None:
        self.jobs = {job.id: job for job in jobs}
        self.asked: list[float] = []

    async def stuck(self, *, older_than_sec: float, limit: int = 100) -> Sequence[Job]:
        self.asked.append(older_than_sec)
        return list(self.jobs.values())[:limit]

    async def update(self, job: Job) -> Job:
        self.jobs[job.id] = job
        return job


class FakeUow:
    def __init__(self, assets: FakeAssets | None = None, jobs: FakeJobs | None = None) -> None:
        self.assets = assets or FakeAssets([])
        self.jobs = jobs or FakeJobs([])
        self.commits = 0

    async def __aenter__(self) -> FakeUow:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def commit(self) -> None:
        self.commits += 1


class FakeStorage:
    def __init__(self, *, broken: set[str] | None = None) -> None:
        self.deleted: list[str] = []
        self.broken = broken or set()

    async def delete(self, key: str) -> None:
        if key in self.broken:
            raise OSError(f"хранилище недоступно: {key}")
        self.deleted.append(key)

    async def put(self, key: str, source: Path, *, content_type: str | None = None) -> str:
        return key


class TestCleanupExpiredAssets:
    @pytest.mark.asyncio
    async def test_expired_files_are_removed(self) -> None:
        assets = FakeAssets([make_asset(key=f"jobs/x/{i}.wav") for i in range(3)])
        uow = FakeUow(assets=assets)
        storage = FakeStorage()

        result = await CleanupExpiredAssets(
            uow_factory=lambda: uow,  # type: ignore[arg-type,return-value]
            storage=storage,  # type: ignore[arg-type]
        ).execute()

        assert result.removed == 3
        assert len(storage.deleted) == 3
        assert len(assets.marked) == 3

    @pytest.mark.asyncio
    async def test_record_is_marked_only_after_the_file_is_gone(self) -> None:
        """Регресс порядка. Файл без записи не найдёт ни одна уборка —
        он просто занимает место навсегда. Запись без файла всего лишь
        даёт понятную ошибку при скачивании."""
        assets = FakeAssets([make_asset(key="жив"), make_asset(key="сломан")])
        uow = FakeUow(assets=assets)
        storage = FakeStorage(broken={"сломан"})

        result = await CleanupExpiredAssets(
            uow_factory=lambda: uow,  # type: ignore[arg-type,return-value]
            storage=storage,  # type: ignore[arg-type]
        ).execute()

        assert (result.removed, result.failed) == (1, 1)
        # Помечен ровно тот, чей файл действительно удалён.
        assert assets.marked == [assets.assets[0].id]

    @pytest.mark.asyncio
    async def test_one_broken_file_does_not_stop_the_rest(self) -> None:
        keys = ["a", "сломан", "c"]
        assets = FakeAssets([make_asset(key=key) for key in keys])
        storage = FakeStorage(broken={"сломан"})

        result = await CleanupExpiredAssets(
            uow_factory=lambda: FakeUow(assets=assets),  # type: ignore[arg-type,return-value]
            storage=storage,  # type: ignore[arg-type]
        ).execute()

        assert storage.deleted == ["a", "c"]
        assert result.examined == 3

    @pytest.mark.asyncio
    async def test_nothing_to_do_is_not_an_error(self) -> None:
        result = await CleanupExpiredAssets(
            uow_factory=lambda: FakeUow(),  # type: ignore[arg-type,return-value]
            storage=FakeStorage(),  # type: ignore[arg-type]
        ).execute()

        assert (result.examined, result.removed) == (0, 0)


class TestReapStuckJobs:
    @pytest.mark.asyncio
    async def test_stuck_job_becomes_failed_with_a_reason(self) -> None:
        jobs = FakeJobs([make_job()])
        uow = FakeUow(jobs=jobs)

        result = await ReapStuckJobs(
            uow_factory=lambda: uow,  # type: ignore[arg-type,return-value]
            timeout_sec=7200,
        ).execute()

        assert result.count == 1
        reaped = next(iter(jobs.jobs.values()))
        assert reaped.status is JobStatus.FAILED
        assert reaped.error is not None
        assert reaped.error["code"] == "STUCK"

    @pytest.mark.asyncio
    async def test_timeout_reaches_the_query(self) -> None:
        """Порог задаётся конфигом и обязан доехать до запроса: уборщик
        с зашитыми двумя часами невозможно настроить под медленное железо."""
        jobs = FakeJobs([])
        await ReapStuckJobs(
            uow_factory=lambda: FakeUow(jobs=jobs),  # type: ignore[arg-type,return-value]
            timeout_sec=999,
        ).execute()

        assert jobs.asked == [999]

    @pytest.mark.asyncio
    async def test_nothing_stuck_means_no_write(self) -> None:
        """Пустой прогон не должен коммитить: уборщик работает каждые
        десять минут, и лишняя транзакция в базе — это шум в логах
        и лишняя нагрузка на ровном месте."""
        uow = FakeUow(jobs=FakeJobs([]))

        result = await ReapStuckJobs(
            uow_factory=lambda: uow,  # type: ignore[arg-type,return-value]
            timeout_sec=7200,
        ).execute()

        assert result.count == 0
        assert uow.commits == 0

    @pytest.mark.asyncio
    async def test_reason_names_the_stage_it_died_on(self) -> None:
        from core.domain.enums import JobStage

        job = replace(make_job(), stage=JobStage.TRANSCRIBE)
        jobs = FakeJobs([job])

        await ReapStuckJobs(
            uow_factory=lambda: FakeUow(jobs=jobs),  # type: ignore[arg-type,return-value]
            timeout_sec=7200,
        ).execute()

        assert jobs.jobs[job.id].error["stage"] == "transcribe"  # type: ignore[index]
