"""Эндпоинт транскрипта на живой БД.

Проверяется то, ради чего он существует: четыре формата, проверка владения
и поведение на пустом транскрипте — у видео без речи его законно нет,
и это не повод отвечать 404.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from httpx import AsyncClient
from tests.fakes import make_media_info, make_transcript

from core.container import Container
from core.domain.entities import Job, User, Video
from core.domain.enums import JobStatus, SourceKind, VideoStatus

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def _seed(container: Container, user: User, *, with_speech: bool = True) -> Job:
    async with container.uow() as uow:
        video = await uow.videos.add(
            Video(
                id=uuid4(),
                user_id=user.id,
                source_uri="upload://x",
                source_kind=SourceKind.UPLOAD,
                status=VideoStatus.READY,
                storage_key="videos/x/source.mp4",
                checksum_sha256="d" * 64,
                info=make_media_info(duration_sec=120.0),
            )
        )
        job = await uow.jobs.add(
            Job(
                id=uuid4(),
                video_id=video.id,
                user_id=user.id,
                status=JobStatus.RUNNING,
                params_hash=str(uuid4()),
            )
        )
        if with_speech:
            await uow.transcripts.replace_for_job(
                job.id,
                make_transcript(
                    (921.48, 924.0, "Держись, они прямо за нами"),
                    (925.0, 927.5, "Я вижу их"),
                ),
                video.id,
            )
        await uow.commit()
    return job


class TestFormats:
    async def test_json_is_the_default(
        self,
        container: Container,
        client: AsyncClient,
        user_with_key: tuple[User, str],
        auth: dict[str, str],
    ) -> None:
        job = await _seed(container, user_with_key[0])

        response = await client.get(f"/api/v1/jobs/{job.id}/transcript", headers=auth)

        assert response.status_code == 200
        body = response.json()
        assert body["language"] == "ru"
        assert body["segment_count"] == 2
        assert body["word_count"] > 0
        # Тайм-код отдаётся и строкой, и числом (док 07, §7.2): парсить
        # «00:15:21.480» на стороне каждого клиента дороже, чем отдать оба.
        assert body["segments"][0]["start"] == "00:15:21.480"
        assert body["segments"][0]["start_sec"] == pytest.approx(921.48)

    async def test_words_can_be_dropped(
        self,
        container: Container,
        client: AsyncClient,
        user_with_key: tuple[User, str],
        auth: dict[str, str],
    ) -> None:
        """Пословные тайминги — это мегабайты на фильм; клиенту, которому
        нужна расшифровка, они ни к чему."""
        job = await _seed(container, user_with_key[0])

        response = await client.get(
            f"/api/v1/jobs/{job.id}/transcript", params={"words": "false"}, headers=auth
        )

        assert response.json()["segments"][0]["words"] == []

    @pytest.mark.parametrize(
        ("fmt", "media_type", "marker"),
        [
            ("srt", "application/x-subrip", "00:15:21,480 --> 00:15:24,000"),
            ("vtt", "text/vtt", "WEBVTT"),
            ("txt", "text/plain", "Я вижу их"),
        ],
    )
    async def test_text_formats(
        self,
        container: Container,
        client: AsyncClient,
        user_with_key: tuple[User, str],
        auth: dict[str, str],
        fmt: str,
        media_type: str,
        marker: str,
    ) -> None:
        job = await _seed(container, user_with_key[0])

        response = await client.get(
            f"/api/v1/jobs/{job.id}/transcript", params={"format": fmt}, headers=auth
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith(media_type)
        assert marker in response.text

    async def test_unknown_format_is_rejected(
        self,
        container: Container,
        client: AsyncClient,
        user_with_key: tuple[User, str],
        auth: dict[str, str],
    ) -> None:
        job = await _seed(container, user_with_key[0])

        response = await client.get(
            f"/api/v1/jobs/{job.id}/transcript", params={"format": "docx"}, headers=auth
        )

        assert response.status_code == 422


class TestAccess:
    async def test_empty_transcript_is_200_not_404(
        self,
        container: Container,
        client: AsyncClient,
        user_with_key: tuple[User, str],
        auth: dict[str, str],
    ) -> None:
        """У видео без речи транскрипта нет законно. Клиент должен увидеть
        «речи нет», а не «джоба не существует»."""
        job = await _seed(container, user_with_key[0], with_speech=False)

        response = await client.get(f"/api/v1/jobs/{job.id}/transcript", headers=auth)

        assert response.status_code == 200
        assert response.json()["segments"] == []

    async def test_foreign_job_is_404(
        self,
        container: Container,
        client: AsyncClient,
        user_with_key: tuple[User, str],
        auth: dict[str, str],
    ) -> None:
        """404, а не 403: существование чужого джоба — тоже информация.
        Транскрипт — это содержание видео, доступ к нему равен доступу к видео."""
        async with container.uow() as uow:
            stranger = await uow.users.create(
                email=f"{uuid4()}@other.local", api_key_hash=str(uuid4())
            )
            await uow.commit()
        job = await _seed(container, stranger)

        response = await client.get(f"/api/v1/jobs/{job.id}/transcript", headers=auth)

        assert response.status_code == 404

    async def test_unknown_job_is_404(self, client: AsyncClient, auth: dict[str, str]) -> None:
        response = await client.get(f"/api/v1/jobs/{uuid4()}/transcript", headers=auth)

        assert response.status_code == 404

    async def test_authentication_is_required(
        self, container: Container, client: AsyncClient, user_with_key: tuple[User, str]
    ) -> None:
        job = await _seed(container, user_with_key[0])

        response = await client.get(f"/api/v1/jobs/{job.id}/transcript")

        assert response.status_code == 401
