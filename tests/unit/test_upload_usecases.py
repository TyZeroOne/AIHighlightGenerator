"""Тесты сценариев загрузки — без БД, без диска, без ffmpeg.

Возможность так тестировать — прямое следствие того, что сценарии зависят
от портов, а не от реализаций. Здесь подставляются фейки из ``tests/fakes.py``,
и весь сценарий целиком проверяется за миллисекунды.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import pytest
from tests.fakes import (
    FakeProbe,
    FakeStorage,
    FakeUnitOfWork,
    make_media_info,
    sha256_of,
    uow_factory,
)

from core.domain.enums import AssetKind, SourceKind, VideoStatus
from core.domain.errors import NotFoundError, UnsupportedMediaError, ValidationError
from core.usecases.upload_video import (
    DeleteVideo,
    GetVideo,
    ListVideos,
    UploadVideo,
    VideoLimits,
    stream_to_file,
)

pytestmark = pytest.mark.unit

LIMITS = VideoLimits(
    allowed_containers=frozenset({"mp4", "mkv", "mov"}),
    max_duration_sec=21600,
    max_bytes=10 * 1024 * 1024,
)


async def chunks_of(data: bytes, size: int = 1024) -> AsyncIterator[bytes]:
    for offset in range(0, len(data), size):
        yield data[offset : offset + size]


class TestStreamToFile:
    async def test_writes_and_hashes_in_one_pass(self, tmp_path: Path) -> None:
        data = b"video-bytes" * 500
        result = await stream_to_file(chunks_of(data), tmp_path / "f.mp4", max_bytes=1 << 20)

        assert result.path.read_bytes() == data
        assert result.size_bytes == len(data)
        assert result.checksum_sha256 == sha256_of(data)

    async def test_aborts_on_limit_and_removes_partial_file(self, tmp_path: Path) -> None:
        """Ключевое поведение: приём обрывается в момент превышения.

        Если ждать конца загрузки, чтобы проверить размер, злоумышленник
        занимает диск на весь объём — а лимит существует именно чтобы
        этого не случилось.
        """
        destination = tmp_path / "big.mp4"

        with pytest.raises(ValidationError, match="лимит"):
            await stream_to_file(chunks_of(b"x" * 5000), destination, max_bytes=1000)

        assert not destination.exists()

    async def test_rejects_empty_upload(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="пустой"):
            await stream_to_file(chunks_of(b""), tmp_path / "empty.mp4", max_bytes=1000)

    async def test_removes_file_when_source_raises(self, tmp_path: Path) -> None:
        """Обрыв соединения не должен оставлять мусор на диске."""

        async def failing() -> AsyncIterator[bytes]:
            yield b"start"
            raise ConnectionError("клиент отвалился")

        destination = tmp_path / "broken.mp4"
        with pytest.raises(ConnectionError):
            await stream_to_file(failing(), destination, max_bytes=1 << 20)

        assert not destination.exists()


class TestUploadVideo:
    async def _upload(
        self,
        tmp_path: Path,
        *,
        uow: FakeUnitOfWork,
        storage: FakeStorage,
        probe: FakeProbe,
        data: bytes = b"movie-data",
        filename: str = "movie.mp4",
        user_id=None,  # type: ignore[no-untyped-def]
    ):  # type: ignore[no-untyped-def]
        stored = await stream_to_file(
            chunks_of(data), tmp_path / filename, max_bytes=LIMITS.max_bytes
        )
        usecase = UploadVideo(
            uow_factory=uow_factory(uow),
            storage=storage,
            probe=probe,
            limits=LIMITS,
        )
        return await usecase.execute(user_id=user_id or uuid4(), upload=stored, filename=filename)

    async def test_happy_path(self, tmp_path: Path) -> None:
        uow, storage, probe = FakeUnitOfWork(), FakeStorage(), FakeProbe()

        video, duplicate = await self._upload(tmp_path, uow=uow, storage=storage, probe=probe)

        assert not duplicate
        assert video.status is VideoStatus.READY
        assert video.source_kind is SourceKind.UPLOAD
        assert video.source_uri == f"upload://{video.id}"
        assert video.storage_key == f"videos/{video.id}/source.mp4"
        # Файл действительно попал в хранилище, а не только в БД.
        assert video.storage_key in storage.objects
        assert uow.commits == 1

    async def test_registers_asset(self, tmp_path: Path) -> None:
        """Артефакт нужен, чтобы уборка знала, что удалять."""
        uow, storage, probe = FakeUnitOfWork(), FakeStorage(), FakeProbe()

        video, _ = await self._upload(tmp_path, uow=uow, storage=storage, probe=probe)

        assets = await uow.assets.list_for_owner("video", video.id)  # type: ignore[arg-type]
        assert len(assets) == 1
        assert assets[0]["kind"] is AssetKind.SOURCE
        assert assets[0]["mime"] == "video/mp4"

    async def test_deduplicates_identical_file(self, tmp_path: Path) -> None:
        """Повторная загрузка того же файла не создаёт вторую запись."""
        uow, storage, probe = FakeUnitOfWork(), FakeStorage(), FakeProbe()
        user_id = uuid4()

        first, dup1 = await self._upload(
            tmp_path, uow=uow, storage=storage, probe=probe, user_id=user_id, filename="a.mp4"
        )
        second, dup2 = await self._upload(
            tmp_path, uow=uow, storage=storage, probe=probe, user_id=user_id, filename="b.mp4"
        )

        assert not dup1 and dup2
        assert first.id == second.id
        assert len(uow.videos.videos) == 1
        assert len(storage.objects) == 1  # второй раз в хранилище не писали

    async def test_same_file_different_users_are_separate(self, tmp_path: Path) -> None:
        """Дедупликация — в пределах пользователя.

        Иначе загрузка популярного файла отдавала бы чужую запись,
        а вместе с ней и чужие результаты анализа.
        """
        uow, storage, probe = FakeUnitOfWork(), FakeStorage(), FakeProbe()

        first, _ = await self._upload(
            tmp_path, uow=uow, storage=storage, probe=probe, user_id=uuid4(), filename="a.mp4"
        )
        second, duplicate = await self._upload(
            tmp_path, uow=uow, storage=storage, probe=probe, user_id=uuid4(), filename="b.mp4"
        )

        assert not duplicate
        assert first.id != second.id

    async def test_rejects_non_video_content(self, tmp_path: Path) -> None:
        """Переименованный .txt: расширение mp4, содержимое — не медиа."""
        probe = FakeProbe(info=make_media_info(video_codec=None, audio_codec=None))

        with pytest.raises(UnsupportedMediaError, match="потока"):
            await self._upload(tmp_path, uow=FakeUnitOfWork(), storage=FakeStorage(), probe=probe)

    async def test_rejects_foreign_container(self, tmp_path: Path) -> None:
        probe = FakeProbe(info=make_media_info(container="avi"))

        with pytest.raises(UnsupportedMediaError, match="avi"):
            await self._upload(tmp_path, uow=FakeUnitOfWork(), storage=FakeStorage(), probe=probe)

    async def test_rejects_too_long_video(self, tmp_path: Path) -> None:
        probe = FakeProbe(info=make_media_info(duration_sec=30000.0))

        with pytest.raises(ValidationError, match="Длительность"):
            await self._upload(tmp_path, uow=FakeUnitOfWork(), storage=FakeStorage(), probe=probe)

    async def test_nothing_is_stored_when_validation_fails(self, tmp_path: Path) -> None:
        """Проверка идёт ДО записи в хранилище — иначе мы платили бы
        за место под файлы, которые сами же и отвергли."""
        uow, storage = FakeUnitOfWork(), FakeStorage()
        probe = FakeProbe(info=make_media_info(container="avi"))

        with pytest.raises(UnsupportedMediaError):
            await self._upload(tmp_path, uow=uow, storage=storage, probe=probe)

        assert storage.objects == {}
        assert uow.videos.videos == {}
        assert uow.commits == 0


class TestGetAndList:
    async def test_get_returns_own_video(self, tmp_path: Path) -> None:
        uow, storage, probe = FakeUnitOfWork(), FakeStorage(), FakeProbe()
        user_id = uuid4()
        stored = await stream_to_file(chunks_of(b"data"), tmp_path / "m.mp4", max_bytes=1 << 20)
        upload = UploadVideo(
            uow_factory=uow_factory(uow), storage=storage, probe=probe, limits=LIMITS
        )
        video, _ = await upload.execute(user_id=user_id, upload=stored, filename="m.mp4")

        found = await GetVideo(uow_factory=uow_factory(uow)).execute(
            user_id=user_id, video_id=video.id
        )
        assert found.id == video.id

    async def test_foreign_video_gives_404_not_403(self, tmp_path: Path) -> None:
        """Ответ 403 подтвердил бы существование объекта постороннему.

        404 не раскрывает ничего: «не найдено» одинаково для чужого
        и несуществующего.
        """
        uow, storage, probe = FakeUnitOfWork(), FakeStorage(), FakeProbe()
        stored = await stream_to_file(chunks_of(b"data"), tmp_path / "m.mp4", max_bytes=1 << 20)
        upload = UploadVideo(
            uow_factory=uow_factory(uow), storage=storage, probe=probe, limits=LIMITS
        )
        video, _ = await upload.execute(user_id=uuid4(), upload=stored, filename="m.mp4")

        with pytest.raises(NotFoundError) as exc_info:
            await GetVideo(uow_factory=uow_factory(uow)).execute(user_id=uuid4(), video_id=video.id)
        assert exc_info.value.http_status == 404

    async def test_list_is_scoped_to_user(self, tmp_path: Path) -> None:
        uow, storage, probe = FakeUnitOfWork(), FakeStorage(), FakeProbe()
        mine, theirs = uuid4(), uuid4()
        upload = UploadVideo(
            uow_factory=uow_factory(uow), storage=storage, probe=probe, limits=LIMITS
        )
        for owner, payload in ((mine, b"a"), (mine, b"bb"), (theirs, b"ccc")):
            stored = await stream_to_file(
                chunks_of(payload), tmp_path / f"{owner}{len(payload)}.mp4", max_bytes=1 << 20
            )
            await upload.execute(user_id=owner, upload=stored, filename="v.mp4")

        items, total = await ListVideos(uow_factory=uow_factory(uow)).execute(user_id=mine)

        assert total == 2
        assert len(items) == 2


class TestDeleteVideo:
    async def test_soft_deletes_record_and_removes_files(self, tmp_path: Path) -> None:
        uow, storage, probe = FakeUnitOfWork(), FakeStorage(), FakeProbe()
        user_id = uuid4()
        stored = await stream_to_file(chunks_of(b"data"), tmp_path / "m.mp4", max_bytes=1 << 20)
        upload = UploadVideo(
            uow_factory=uow_factory(uow), storage=storage, probe=probe, limits=LIMITS
        )
        video, _ = await upload.execute(user_id=user_id, upload=stored, filename="m.mp4")

        await DeleteVideo(uow_factory=uow_factory(uow), storage=storage).execute(
            user_id=user_id, video_id=video.id
        )

        # Запись осталась, но помечена удалённой: на неё могут ссылаться джобы.
        assert uow.videos.videos[video.id].status is VideoStatus.DELETED
        # Файлы удалены по-настоящему — место они занимают реально.
        assert storage.objects == {}
        # И видео больше не выдаётся.
        with pytest.raises(NotFoundError):
            await GetVideo(uow_factory=uow_factory(uow)).execute(user_id=user_id, video_id=video.id)

    async def test_storage_failure_does_not_break_deletion(self, tmp_path: Path) -> None:
        """Уборка файлов — best effort.

        Если хранилище недоступно, пользователь всё равно должен получить
        удаление: запись помечена, а файлы подберёт фоновая задача.
        """
        uow, storage, probe = FakeUnitOfWork(), FakeStorage(), FakeProbe()
        user_id = uuid4()
        stored = await stream_to_file(chunks_of(b"data"), tmp_path / "m.mp4", max_bytes=1 << 20)
        upload = UploadVideo(
            uow_factory=uow_factory(uow), storage=storage, probe=probe, limits=LIMITS
        )
        video, _ = await upload.execute(user_id=user_id, upload=stored, filename="m.mp4")

        async def broken_delete_prefix(prefix: str) -> int:
            raise ConnectionError("S3 недоступен")

        storage.delete_prefix = broken_delete_prefix  # type: ignore[method-assign]

        await DeleteVideo(uow_factory=uow_factory(uow), storage=storage).execute(
            user_id=user_id, video_id=video.id
        )
        assert uow.videos.videos[video.id].status is VideoStatus.DELETED

    async def test_deleting_foreign_video_is_404(self) -> None:
        uow = FakeUnitOfWork()
        with pytest.raises(NotFoundError):
            await DeleteVideo(uow_factory=uow_factory(uow), storage=FakeStorage()).execute(
                user_id=uuid4(), video_id=uuid4()
            )
