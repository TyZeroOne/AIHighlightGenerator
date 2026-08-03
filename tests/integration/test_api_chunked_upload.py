"""Тесты возобновляемой загрузки: init → part → complete."""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from tests.fakes import FakeProbe, make_media_info

from core.container import Container

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

PAYLOAD = b"chunked-video-payload" * 500
PART_SIZE = 4096


@pytest.fixture(autouse=True)
def _fake_probe(container: Container) -> None:
    container.__dict__["probe"] = FakeProbe(make_media_info())


def parts_of(data: bytes, size: int = PART_SIZE) -> list[bytes]:
    return [data[offset : offset + size] for offset in range(0, len(data), size)]


async def _init(client: AsyncClient, auth: dict[str, str], filename: str = "big.mkv") -> str:
    response = await client.post(
        "/api/v1/upload/init",
        headers=auth,
        json={"filename": filename, "total_size": len(PAYLOAD)},
    )
    assert response.status_code == 201
    return str(response.json()["upload_id"])


async def _send(
    client: AsyncClient,
    auth: dict[str, str],
    upload_id: str,
    number: int,
    payload: bytes,
):  # type: ignore[no-untyped-def]
    return await client.put(
        f"/api/v1/upload/{upload_id}/part/{number}",
        headers={**auth, "Content-Type": "application/octet-stream"},
        content=payload,
    )


class TestHappyPath:
    async def test_full_cycle(self, client: AsyncClient, auth: dict[str, str]) -> None:
        upload_id = await _init(client, auth)
        chunks = parts_of(PAYLOAD)

        for number, chunk in enumerate(chunks, start=1):
            response = await _send(client, auth, upload_id, number, chunk)
            assert response.status_code == 200
            assert response.json()["received_parts"] == number

        complete = await client.post(f"/api/v1/upload/{upload_id}/complete", headers=auth)

        assert complete.status_code == 201
        assert complete.json()["size_bytes"] == len(PAYLOAD)

    async def test_assembled_file_is_byte_identical(
        self, client: AsyncClient, auth: dict[str, str], container: Container
    ) -> None:
        """Главное свойство сборки: результат совпадает с исходником побайтно.

        Ошибка на один байт здесь дала бы битое видео, которое упало бы
        далеко отсюда — на этапе анализа, где причину уже не найти.
        """
        upload_id = await _init(client, auth)
        for number, chunk in enumerate(parts_of(PAYLOAD), start=1):
            await _send(client, auth, upload_id, number, chunk)

        video_id = (await client.post(f"/api/v1/upload/{upload_id}/complete", headers=auth)).json()[
            "video_id"
        ]

        download = await client.get(f"/api/v1/videos/{video_id}/download", headers=auth)
        assert download.content == PAYLOAD

    async def test_parts_may_arrive_out_of_order(
        self, client: AsyncClient, auth: dict[str, str]
    ) -> None:
        """Параллельная отправка не гарантирует порядок — и не должна.

        Склейка идёт по номеру части, а не по времени прибытия.
        """
        upload_id = await _init(client, auth)
        chunks = parts_of(PAYLOAD)

        for number in reversed(range(1, len(chunks) + 1)):
            await _send(client, auth, upload_id, number, chunks[number - 1])

        video_id = (await client.post(f"/api/v1/upload/{upload_id}/complete", headers=auth)).json()[
            "video_id"
        ]
        download = await client.get(f"/api/v1/videos/{video_id}/download", headers=auth)

        assert download.content == PAYLOAD


class TestResumability:
    async def test_status_tells_where_to_resume(
        self, client: AsyncClient, auth: dict[str, str]
    ) -> None:
        """После обрыва клиент спрашивает состояние и дошлёт недостающее."""
        upload_id = await _init(client, auth)
        chunks = parts_of(PAYLOAD)
        for number in (1, 2):
            await _send(client, auth, upload_id, number, chunks[number - 1])

        status = (await client.get(f"/api/v1/upload/{upload_id}", headers=auth)).json()

        assert status["received_parts"] == [1, 2]
        assert status["next_expected_part"] == 3
        assert status["completed"] is False

    async def test_resending_a_part_is_idempotent(
        self, client: AsyncClient, auth: dict[str, str]
    ) -> None:
        """Повторная отправка после таймаута не должна ломать учёт байтов."""
        upload_id = await _init(client, auth)
        chunk = parts_of(PAYLOAD)[0]

        first = await _send(client, auth, upload_id, 1, chunk)
        second = await _send(client, auth, upload_id, 1, chunk)

        assert first.json()["received_bytes"] == second.json()["received_bytes"]
        assert second.json()["received_parts"] == 1


class TestFailures:
    async def test_complete_with_missing_part_is_409(
        self, client: AsyncClient, auth: dict[str, str]
    ) -> None:
        """Дыра в нумерации — отказ, а не битый файл.

        Собрать без части технически можно; получится видео, которое
        сломается на анализе. Лучше честный отказ сейчас.
        """
        upload_id = await _init(client, auth)
        chunks = parts_of(PAYLOAD)
        for number in (1, 3):  # части 2 нет
            await _send(client, auth, upload_id, number, chunks[number - 1])

        response = await client.post(f"/api/v1/upload/{upload_id}/complete", headers=auth)

        assert response.status_code == 409
        assert 2 in response.json()["context"]["missing"]

    async def test_complete_without_parts_is_422(
        self, client: AsyncClient, auth: dict[str, str]
    ) -> None:
        upload_id = await _init(client, auth)

        response = await client.post(f"/api/v1/upload/{upload_id}/complete", headers=auth)

        assert response.status_code == 422

    async def test_second_complete_is_409(self, client: AsyncClient, auth: dict[str, str]) -> None:
        upload_id = await _init(client, auth)
        for number, chunk in enumerate(parts_of(PAYLOAD), start=1):
            await _send(client, auth, upload_id, number, chunk)
        await client.post(f"/api/v1/upload/{upload_id}/complete", headers=auth)

        again = await client.post(f"/api/v1/upload/{upload_id}/complete", headers=auth)

        assert again.status_code == 409

    async def test_init_rejects_bad_extension(
        self, client: AsyncClient, auth: dict[str, str]
    ) -> None:
        """Отказ на init, а не после 20 ГБ трафика."""
        response = await client.post(
            "/api/v1/upload/init", headers=auth, json={"filename": "archive.zip"}
        )

        assert response.status_code == 415

    async def test_init_rejects_oversized_declaration(
        self, client: AsyncClient, auth: dict[str, str]
    ) -> None:
        response = await client.post(
            "/api/v1/upload/init",
            headers=auth,
            json={"filename": "huge.mp4", "total_size": 100 * 1024**3},
        )

        assert response.status_code == 422

    async def test_foreign_upload_session_is_404(
        self, client: AsyncClient, auth: dict[str, str], container: Container
    ) -> None:
        upload_id = await _init(client, auth)

        from api.deps import generate_api_key

        other_key, other_hash = generate_api_key()
        async with container.uow() as uow:
            await uow.users.create(email="stranger@test.local", api_key_hash=other_hash)
            await uow.commit()

        response = await client.get(f"/api/v1/upload/{upload_id}", headers={"X-API-Key": other_key})

        assert response.status_code == 404

    async def test_parts_are_cleaned_up_after_completion(
        self, client: AsyncClient, auth: dict[str, str], container: Container
    ) -> None:
        """Временные части не должны пережить сборку — это чистый мусор."""
        upload_id = await _init(client, auth)
        for number, chunk in enumerate(parts_of(PAYLOAD), start=1):
            await _send(client, auth, upload_id, number, chunk)

        await client.post(f"/api/v1/upload/{upload_id}/complete", headers=auth)

        assert not await container.storage.exists(f"uploads/{upload_id}/000001.part")
