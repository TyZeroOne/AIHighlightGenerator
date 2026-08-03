"""Тесты эндпоинтов загрузки на живой БД.

ffprobe здесь подменяется фейком: цель — проверить HTTP-слой, права доступа
и работу с БД, а не умение ffmpeg читать mp4. Разбор реального вывода ffprobe
покрыт отдельно в ``tests/unit/test_probe.py``, а работа настоящего бинарника —
в ``test_ffprobe_real.py``.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from tests.fakes import FakeProbe, make_media_info

from core.container import Container

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

MP4 = b"\x00\x00\x00\x20ftypisom" + b"fake-video-payload" * 100


@pytest.fixture(autouse=True)
def _fake_probe(container: Container) -> None:
    """Подменяем ffprobe в контейнере.

    ``cached_property`` хранит значение в ``__dict__`` экземпляра, поэтому
    достаточно записать туда своё — обращение к ``container.probe`` вернёт его.
    """
    container.__dict__["probe"] = FakeProbe(make_media_info(duration_sec=95.5))


class TestAuth:
    async def test_requires_api_key(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/videos")

        assert response.status_code == 401
        assert response.json()["code"] == "UNAUTHENTICATED"

    async def test_rejects_wrong_key(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/videos", headers={"X-API-Key": "hg_nope"})

        assert response.status_code == 401


class TestUpload:
    async def test_uploads_and_returns_metadata(
        self, client: AsyncClient, auth: dict[str, str]
    ) -> None:
        response = await client.post(
            "/api/v1/upload",
            headers=auth,
            files={"file": ("movie.mp4", MP4, "video/mp4")},
        )

        assert response.status_code == 201
        body = response.json()
        assert body["status"] == "ready"
        assert body["duration_sec"] == pytest.approx(95.5)
        assert body["width"] == 1920
        assert body["has_audio"] is True
        assert body["duplicate"] is False
        assert body["size_bytes"] == len(MP4)

    async def test_file_lands_in_storage(
        self, client: AsyncClient, auth: dict[str, str], container: Container
    ) -> None:
        response = await client.post(
            "/api/v1/upload", headers=auth, files={"file": ("m.mp4", MP4, "video/mp4")}
        )
        video_id = response.json()["video_id"]

        assert await container.storage.exists(f"videos/{video_id}/source.mp4")

    async def test_duplicate_returns_200_and_same_id(
        self, client: AsyncClient, auth: dict[str, str]
    ) -> None:
        """Повторная загрузка — не ошибка и не новая запись.

        Код 200 вместо 201 позволяет клиенту понять, что ничего не создано,
        не разбирая тело ответа.
        """
        first = await client.post(
            "/api/v1/upload", headers=auth, files={"file": ("a.mp4", MP4, "video/mp4")}
        )
        second = await client.post(
            "/api/v1/upload", headers=auth, files={"file": ("b.mp4", MP4, "video/mp4")}
        )

        assert first.status_code == 201
        assert second.status_code == 200
        assert second.json()["duplicate"] is True
        assert first.json()["video_id"] == second.json()["video_id"]

    @pytest.mark.parametrize(
        "filename",
        ["notes.txt", "archive.zip", "movie.avi", "noextension"],
    )
    async def test_rejects_bad_extension_with_415(
        self, client: AsyncClient, auth: dict[str, str], filename: str
    ) -> None:
        response = await client.post(
            "/api/v1/upload",
            headers=auth,
            files={"file": (filename, b"whatever", "application/octet-stream")},
        )

        assert response.status_code == 415
        problem = response.json()
        assert problem["code"] == "UNSUPPORTED_MEDIA"
        assert response.headers["content-type"].startswith("application/problem+json")

    async def test_rejects_renamed_text_file_by_content(
        self, client: AsyncClient, auth: dict[str, str], container: Container
    ) -> None:
        """Главная проверка этапа: расширение mp4, содержимое — не медиа.

        Отличить их можно только прочитав файл. Здесь ffprobe-фейк говорит,
        что потоков нет, — ровно то, что сказал бы настоящий ffprobe.
        """
        container.__dict__["probe"] = FakeProbe(make_media_info(video_codec=None, audio_codec=None))

        response = await client.post(
            "/api/v1/upload",
            headers=auth,
            files={"file": ("definitely_video.mp4", b"just plain text", "video/mp4")},
        )

        assert response.status_code == 415
        assert "потока" in response.json()["detail"]

    async def test_rejects_too_large_file(self, client: AsyncClient, auth: dict[str, str]) -> None:
        # Лимит в тестовых настройках — 8 МиБ.
        oversized = b"x" * (9 * 1024 * 1024)

        response = await client.post(
            "/api/v1/upload", headers=auth, files={"file": ("big.mp4", oversized, "video/mp4")}
        )

        assert response.status_code in (413, 422)

    async def test_rejects_too_long_video(
        self, client: AsyncClient, auth: dict[str, str], container: Container
    ) -> None:
        container.__dict__["probe"] = FakeProbe(make_media_info(duration_sec=7200.0))

        response = await client.post(
            "/api/v1/upload", headers=auth, files={"file": ("long.mp4", MP4, "video/mp4")}
        )

        assert response.status_code == 422
        assert "Длительность" in response.json()["detail"]


class TestListGetDelete:
    async def _upload(self, client: AsyncClient, auth: dict[str, str], payload: bytes) -> str:
        response = await client.post(
            "/api/v1/upload", headers=auth, files={"file": ("v.mp4", payload, "video/mp4")}
        )
        assert response.status_code == 201
        return str(response.json()["video_id"])

    async def test_list_returns_own_videos(self, client: AsyncClient, auth: dict[str, str]) -> None:
        await self._upload(client, auth, MP4)
        await self._upload(client, auth, MP4 + b"different")

        body = (await client.get("/api/v1/videos", headers=auth)).json()

        assert body["total"] == 2
        assert len(body["items"]) == 2

    async def test_get_single_video(self, client: AsyncClient, auth: dict[str, str]) -> None:
        video_id = await self._upload(client, auth, MP4)

        response = await client.get(f"/api/v1/videos/{video_id}", headers=auth)

        assert response.status_code == 200
        assert response.json()["video_id"] == video_id

    async def test_foreign_video_is_404(
        self, client: AsyncClient, auth: dict[str, str], container: Container
    ) -> None:
        """Чужое видео неотличимо от несуществующего — это защита от разведки."""
        video_id = await self._upload(client, auth, MP4)

        from api.deps import generate_api_key

        other_key, other_hash = generate_api_key()
        async with container.uow() as uow:
            await uow.users.create(email="other@test.local", api_key_hash=other_hash)
            await uow.commit()

        response = await client.get(f"/api/v1/videos/{video_id}", headers={"X-API-Key": other_key})

        assert response.status_code == 404

    async def test_delete_removes_from_list_and_storage(
        self, client: AsyncClient, auth: dict[str, str], container: Container
    ) -> None:
        video_id = await self._upload(client, auth, MP4)
        key = f"videos/{video_id}/source.mp4"
        assert await container.storage.exists(key)

        response = await client.delete(f"/api/v1/videos/{video_id}", headers=auth)

        assert response.status_code == 204
        assert not await container.storage.exists(key)
        assert (await client.get(f"/api/v1/videos/{video_id}", headers=auth)).status_code == 404
        assert (await client.get("/api/v1/videos", headers=auth)).json()["total"] == 0

    async def test_download_streams_original(
        self, client: AsyncClient, auth: dict[str, str]
    ) -> None:
        video_id = await self._upload(client, auth, MP4)

        response = await client.get(f"/api/v1/videos/{video_id}/download", headers=auth)

        assert response.status_code == 200
        assert response.content == MP4
        assert "attachment" in response.headers["content-disposition"]


class TestRegisterLocal:
    async def test_registers_file_without_copying(
        self, client: AsyncClient, auth: dict[str, str], container: Container, tmp_path
    ) -> None:
        mounted = tmp_path / "mounted"
        mounted.mkdir(parents=True, exist_ok=True)
        source = mounted / "archive.mkv"
        source.write_bytes(MP4)

        response = await client.post(
            "/api/v1/videos/register",
            headers=auth,
            json={"source_uri": source.as_uri()},
        )

        assert response.status_code == 201
        body = response.json()
        assert body["source_kind"] == "local_file"
        # Байты не копировались: в хранилище пусто.
        assert not await container.storage.exists(f"videos/{body['video_id']}/source.mkv")

    async def test_rejects_path_outside_allowed_roots(
        self, client: AsyncClient, auth: dict[str, str], tmp_path
    ) -> None:
        """`file:///etc/passwd` — валидный URI и полностью недопустимый запрос."""
        outside = tmp_path / "outside.mkv"
        outside.write_bytes(MP4)

        response = await client.post(
            "/api/v1/videos/register",
            headers=auth,
            json={"source_uri": outside.as_uri()},
        )

        assert response.status_code == 400
        assert "разрешённых каталогов" in response.json()["detail"]

    async def test_rejects_non_file_scheme(self, client: AsyncClient, auth: dict[str, str]) -> None:
        response = await client.post(
            "/api/v1/videos/register",
            headers=auth,
            json={"source_uri": "https://youtube.com/watch?v=x"},
        )

        assert response.status_code == 422
