"""API джобов: аутентификация, владение, лимиты, управление.

Здесь проверяются требования DoD Этапа 10, которые формулируются
в терминах HTTP, а не предметной области: какой код возврата, какой
заголовок, что видит чужой пользователь.

Конвейер при этом никуда не уходит: диспетчер подменён фейком. Проверять
здесь надо решения API, а не то, что Celery умеет доставлять сообщения, —
для второго есть отдельные тесты и живые прогоны.
"""

from __future__ import annotations

from dataclasses import replace
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient

from core.container import Container
from core.domain.entities import Job, User, Video
from core.domain.enums import JobStage, JobStatus, SourceKind, VideoStatus
from core.domain.media import MediaInfo
from core.ports.limits import RateVerdict

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class FakeDispatcher:
    """Диспетчер, который ничего не отправляет, но всё запоминает."""

    def __init__(self) -> None:
        self.started: list[tuple[UUID, JobStage | None]] = []
        self.cancelled: list[str | None] = []

    def start(self, job_id: UUID, *, from_stage: JobStage | None = None) -> str | None:
        self.started.append((job_id, from_stage))
        return f"root-{len(self.started)}"

    def cancel(self, root_id: str | None) -> bool:
        self.cancelled.append(root_id)
        return root_id is not None


@pytest.fixture
def dispatcher(container: Container) -> FakeDispatcher:
    """Подменяем доставку, оставляя всё остальное настоящим."""
    fake = FakeDispatcher()
    # ``cached_property`` хранит значение в ``__dict__`` — записав его
    # напрямую, подменяем зависимость, не трогая ни контейнер, ни сценарии.
    container.__dict__["dispatcher"] = fake
    for name in ("start_analysis", "retry_job", "cancel_job"):
        container.__dict__.pop(name, None)
    return fake


async def make_video(container: Container, user: User, *, duration: float = 120.0) -> Video:
    async with container.uow() as uow:
        video = await uow.videos.add(
            Video(
                id=uuid4(),
                user_id=user.id,
                source_uri=f"upload://{uuid4()}",
                source_kind=SourceKind.UPLOAD,
                status=VideoStatus.READY,
                storage_key=f"videos/{uuid4()}/source.mp4",
                checksum_sha256=uuid4().hex + uuid4().hex,
                info=MediaInfo(duration_sec=duration, width=640, height=360, video_codec="h264"),
            )
        )
        await uow.commit()
    return video


# ------------------------------------------------------------ аутентификация
class TestAuthentication:
    async def test_request_without_key_is_rejected(self, client: AsyncClient) -> None:
        """DoD: неаутентифицированный запрос → 401."""
        response = await client.get("/api/v1/jobs")
        assert response.status_code == 401

    async def test_unknown_key_is_rejected(self, client: AsyncClient) -> None:
        # Ключ латиницей: заголовки HTTP — ascii, и кириллица в них
        # падает на кодировании ещё до отправки.
        response = await client.get("/api/v1/jobs", headers={"X-API-Key": "hg_no-such-key"})
        assert response.status_code == 401

    async def test_health_needs_no_key(self, client: AsyncClient) -> None:
        """Проверку живости дёргает оркестратор, у которого ключа нет."""
        assert (await client.get("/health")).status_code == 200

    async def test_metrics_need_no_key(self, client: AsyncClient) -> None:
        """И Prometheus тоже."""
        assert (await client.get("/metrics")).status_code == 200


# ------------------------------------------------------------------ владение
class TestOwnership:
    async def test_someone_elses_job_is_not_found(
        self, client: AsyncClient, container: Container, user_with_key: tuple[User, str]
    ) -> None:
        """DoD: чужой ресурс → 404, а не 403.

        Именно 404: код 403 сообщал бы, что объект существует, — а это
        уже утечка. Пользователь не должен уметь перебором узнать,
        какие джобы есть у других.
        """
        stranger_key = await _make_stranger(container)
        owner, _ = user_with_key
        video = await make_video(container, owner)

        async with container.uow() as uow:
            job = await uow.jobs.add(
                Job(
                    id=uuid4(),
                    video_id=video.id,
                    user_id=owner.id,
                    status=JobStatus.SUCCEEDED,
                    params_hash="x" * 64,
                )
            )
            await uow.commit()

        for path in (f"/api/v1/jobs/{job.id}/status", f"/api/v1/clips/{job.id}"):
            response = await client.get(path, headers={"X-API-Key": stranger_key})
            assert response.status_code == 404, path

    async def test_someone_elses_video_cannot_be_analysed(
        self, client: AsyncClient, container: Container, user_with_key: tuple[User, str]
    ) -> None:
        owner, _ = user_with_key
        video = await make_video(container, owner)
        stranger_key = await _make_stranger(container)

        response = await client.post(
            "/api/v1/jobs",
            headers={"X-API-Key": stranger_key},
            json={"video_id": str(video.id)},
        )
        assert response.status_code == 404

    async def test_list_shows_only_own_jobs(
        self, client: AsyncClient, container: Container, user_with_key: tuple[User, str]
    ) -> None:
        owner, key = user_with_key
        video = await make_video(container, owner)
        stranger_key = await _make_stranger(container)

        async with container.uow() as uow:
            await uow.jobs.add(
                Job(
                    id=uuid4(),
                    video_id=video.id,
                    user_id=owner.id,
                    status=JobStatus.RUNNING,
                    params_hash="y" * 64,
                )
            )
            await uow.commit()

        mine = await client.get("/api/v1/jobs", headers={"X-API-Key": key})
        theirs = await client.get("/api/v1/jobs", headers={"X-API-Key": stranger_key})

        assert mine.json()["total"] == 1
        assert theirs.json()["total"] == 0


# ----------------------------------------------------------------- запуск
class TestStartAnalysis:
    async def test_job_is_queued_and_dispatched(
        self,
        client: AsyncClient,
        container: Container,
        auth: dict[str, str],
        user_with_key: tuple[User, str],
        dispatcher: FakeDispatcher,
    ) -> None:
        video = await make_video(container, user_with_key[0])

        response = await client.post(
            "/api/v1/jobs",
            headers=auth,
            json={"video_id": str(video.id), "profile": "movie", "max_clips": 3},
        )

        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "queued"
        assert not body["reused"]
        assert len(dispatcher.started) == 1

    async def test_same_params_reuse_the_finished_job(
        self,
        client: AsyncClient,
        container: Container,
        auth: dict[str, str],
        user_with_key: tuple[User, str],
        dispatcher: FakeDispatcher,
    ) -> None:
        """Идемпотентность: повтор не запускает сорок минут работы заново."""
        video = await make_video(container, user_with_key[0])
        payload = {"video_id": str(video.id), "profile": "movie"}

        first = (await client.post("/api/v1/jobs", headers=auth, json=payload)).json()
        async with container.uow() as uow:
            job = await uow.jobs.get(UUID(first["job_id"]))
            assert job is not None
            await uow.jobs.update(replace(job, status=JobStatus.SUCCEEDED))
            await uow.commit()

        second = await client.post("/api/v1/jobs", headers=auth, json=payload)

        assert second.json()["reused"] is True
        assert second.json()["job_id"] == first["job_id"]
        assert len(dispatcher.started) == 1, "повтор не должен ставить задачу второй раз"

    async def test_key_order_does_not_change_the_fingerprint(
        self,
        client: AsyncClient,
        container: Container,
        auth: dict[str, str],
        user_with_key: tuple[User, str],
        dispatcher: FakeDispatcher,
    ) -> None:
        """Отпечаток считается от смысла, а не от того, как клиент собрал JSON."""
        video = await make_video(container, user_with_key[0])

        first = await client.post(
            "/api/v1/jobs",
            headers=auth,
            json={"video_id": str(video.id), "profile": "movie", "max_clips": 3},
        )
        second = await client.post(
            "/api/v1/jobs",
            headers=auth,
            json={"max_clips": 3, "profile": "movie", "video_id": str(video.id)},
        )

        assert first.json()["params_hash"] == second.json()["params_hash"]

    async def test_force_starts_a_new_job(
        self,
        client: AsyncClient,
        container: Container,
        auth: dict[str, str],
        user_with_key: tuple[User, str],
        dispatcher: FakeDispatcher,
    ) -> None:
        video = await make_video(container, user_with_key[0])
        payload = {"video_id": str(video.id)}

        first = (await client.post("/api/v1/jobs", headers=auth, json=payload)).json()
        async with container.uow() as uow:
            job = await uow.jobs.get(UUID(first["job_id"]))
            assert job is not None
            await uow.jobs.update(replace(job, status=JobStatus.SUCCEEDED))
            await uow.commit()

        forced = await client.post("/api/v1/jobs", headers=auth, json={**payload, "force": True})

        assert forced.json()["job_id"] != first["job_id"]
        assert len(dispatcher.started) == 2

    async def test_missing_video_is_404(self, client: AsyncClient, auth: dict[str, str]) -> None:
        response = await client.post("/api/v1/jobs", headers=auth, json={"video_id": str(uuid4())})
        assert response.status_code == 404

    async def test_absurd_parameters_are_rejected(
        self,
        client: AsyncClient,
        container: Container,
        auth: dict[str, str],
        user_with_key: tuple[User, str],
    ) -> None:
        """Запрос на тысячу клипов — это часы работы воркера по одной
        строчке в JSON, и отказывать в нём надо до, а не после."""
        video = await make_video(container, user_with_key[0])
        response = await client.post(
            "/api/v1/jobs",
            headers=auth,
            json={"video_id": str(video.id), "max_clips": 1000},
        )
        assert response.status_code in {400, 422}


# -------------------------------------------------------- повтор и отмена
class TestRetryAndCancel:
    async def _job(self, container: Container, user: User, **kwargs: object) -> Job:
        video = await make_video(container, user)
        async with container.uow() as uow:
            job = await uow.jobs.add(
                Job(
                    id=uuid4(),
                    video_id=video.id,
                    user_id=user.id,
                    status=JobStatus.RUNNING,
                    params_hash=uuid4().hex + uuid4().hex,
                    **kwargs,  # type: ignore[arg-type]
                )
            )
            await uow.commit()
        return job

    async def test_retry_starts_from_the_requested_stage(
        self,
        client: AsyncClient,
        container: Container,
        auth: dict[str, str],
        user_with_key: tuple[User, str],
        dispatcher: FakeDispatcher,
    ) -> None:
        """DoD: повтор с этапа не пересчитывает предыдущие."""
        job = await self._job(container, user_with_key[0])

        response = await client.post(
            f"/api/v1/jobs/{job.id}/retry",
            headers=auth,
            json={"from_stage": "llm_rank"},
        )

        assert response.status_code == 202
        assert dispatcher.started == [(job.id, JobStage.LLM_RANK)]

    async def test_retry_clears_only_the_tail(
        self,
        client: AsyncClient,
        container: Container,
        auth: dict[str, str],
        user_with_key: tuple[User, str],
        dispatcher: FakeDispatcher,
    ) -> None:
        job = await self._job(container, user_with_key[0])
        async with container.uow() as uow:
            marked = job
            for stage in (JobStage.DETECT_SCENES, JobStage.TRANSCRIBE, JobStage.LLM_RANK):
                marked = marked.with_stage(stage, "succeeded", progress=100.0)
            await uow.jobs.update(marked)
            await uow.commit()

        await client.post(
            f"/api/v1/jobs/{job.id}/retry", headers=auth, json={"from_stage": "llm_rank"}
        )

        async with container.uow() as uow:
            updated = await uow.jobs.get(job.id)
        assert updated is not None
        assert JobStage.DETECT_SCENES in updated.stages, "предыдущие этапы должны остаться"
        assert JobStage.TRANSCRIBE in updated.stages
        assert JobStage.LLM_RANK not in updated.stages, "указанный этап должен быть забыт"

    async def test_cancel_marks_the_job_and_revokes_the_flow(
        self,
        client: AsyncClient,
        container: Container,
        auth: dict[str, str],
        user_with_key: tuple[User, str],
        dispatcher: FakeDispatcher,
    ) -> None:
        job = await self._job(container, user_with_key[0], celery_root_id="root-42")

        response = await client.delete(f"/api/v1/jobs/{job.id}", headers=auth)

        assert response.status_code == 200
        assert dispatcher.cancelled == ["root-42"]
        async with container.uow() as uow:
            updated = await uow.jobs.get(job.id)
        assert updated is not None
        assert updated.status is JobStatus.CANCELLED

    async def test_finished_job_cannot_be_cancelled(
        self,
        client: AsyncClient,
        container: Container,
        auth: dict[str, str],
        user_with_key: tuple[User, str],
        dispatcher: FakeDispatcher,
    ) -> None:
        video = await make_video(container, user_with_key[0])
        async with container.uow() as uow:
            job = await uow.jobs.add(
                Job(
                    id=uuid4(),
                    video_id=video.id,
                    user_id=user_with_key[0].id,
                    status=JobStatus.SUCCEEDED,
                    params_hash=uuid4().hex + uuid4().hex,
                )
            )
            await uow.commit()

        response = await client.delete(f"/api/v1/jobs/{job.id}", headers=auth)

        # 409, а не 422: запрос корректен, конфликтует состояние.
        assert response.status_code == 409
        assert not dispatcher.cancelled


# --------------------------------------------------------------- лимиты
class TestRateLimit:
    async def test_limit_gives_429_with_retry_after(
        self, client: AsyncClient, container: Container, auth: dict[str, str]
    ) -> None:
        """DoD: rate limit срабатывает и отдаёт ``Retry-After``.

        Ограничитель подменяется на «всё запрещено»: проверяется реакция
        API, а не арифметика окна — она проверена отдельно, на своих
        тестах, и не требует шестидесяти запросов подряд.
        """

        class Denying:
            async def hit(self, key: str, *, limit: int, window_sec: int) -> RateVerdict:
                return RateVerdict(allowed=False, remaining=0, retry_after_sec=17, limit=limit)

        container.__dict__["rate_limiter"] = Denying()

        response = await client.get("/api/v1/jobs", headers=auth)

        assert response.status_code == 429
        assert response.headers["Retry-After"] == "17"
        assert response.json()["code"]

    async def test_health_is_not_rate_limited(
        self, client: AsyncClient, container: Container
    ) -> None:
        """Ограничивать проверку живости — значит уронить её под нагрузкой
        ровно тогда, когда она нужнее всего."""

        class Denying:
            async def hit(self, key: str, *, limit: int, window_sec: int) -> RateVerdict:
                return RateVerdict(allowed=False, remaining=0, retry_after_sec=1, limit=limit)

        container.__dict__["rate_limiter"] = Denying()

        assert (await client.get("/health")).status_code == 200
        assert (await client.get("/health/ready")).status_code in {200, 503}


async def _make_stranger(container: Container) -> str:
    from api.deps import generate_api_key

    key, key_hash = generate_api_key()
    async with container.uow() as uow:
        await uow.users.create(email=f"{uuid4()}@stranger.local", api_key_hash=key_hash)
        await uow.commit()
    return key
