"""Один набор тестов для обеих реализаций ``ObjectStorage``.

Это главный тест абстракции. Если локальное хранилище и S3 проходят
**одни и те же** проверки, значит, вызывающий код действительно может
не знать, что за ними стоит. Разошлось поведение — абстракция протекла,
и это видно сразу, а не на проде через полгода.

``LocalStorage`` работает всегда. ``S3Storage`` требует MinIO
(``docker compose --profile s3 up -d minio``) и пропускается, если его нет.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from core.domain.errors import ObjectNotFoundError, StorageError
from core.infra.storage.local import LocalStorage

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

MINIO_ENDPOINT = os.environ.get("TEST_S3_ENDPOINT", "http://localhost:9000")
MINIO_KEY = os.environ.get("TEST_S3_ACCESS_KEY", "minioadmin")
MINIO_SECRET = os.environ.get("TEST_S3_SECRET_KEY", "minioadmin")


async def _s3_storage():  # type: ignore[no-untyped-def]
    from core.infra.storage.s3 import S3Storage

    storage = S3Storage(
        bucket="highlight-tests",
        endpoint_url=MINIO_ENDPOINT,
        access_key=MINIO_KEY,
        secret_key=MINIO_SECRET,
    )
    await storage.ensure_bucket()
    return storage


@pytest.fixture(params=["local", "s3"])
async def storage(request: pytest.FixtureRequest, tmp_path: Path):  # type: ignore[no-untyped-def]
    """Параметризованная фикстура: каждый тест ниже выполняется дважды.

    Именно параметризация, а не два похожих файла тестов: копия неизбежно
    разъедется с оригиналом, и один из бэкендов окажется проверен хуже.
    """
    if request.param == "local":
        yield LocalStorage(tmp_path / "storage")
        return

    try:
        yield await _s3_storage()
    except Exception as error:
        pytest.skip(f"MinIO недоступен на {MINIO_ENDPOINT}: {error}")


@pytest.fixture
def sample(tmp_path: Path) -> Path:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"highlight-generator" * 1000)
    return path


class TestRoundTrip:
    async def test_put_then_get(self, storage, sample: Path, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        await storage.put("videos/a/source.mp4", sample, content_type="video/mp4")

        restored = await storage.get("videos/a/source.mp4", tmp_path / "out.bin")

        assert restored.read_bytes() == sample.read_bytes()

    async def test_put_overwrites(self, storage, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        """Перезапись, а не дубль — на этом держится идемпотентность этапов."""
        first, second = tmp_path / "1.bin", tmp_path / "2.bin"
        first.write_bytes(b"first")
        second.write_bytes(b"second")

        await storage.put("k", first)
        await storage.put("k", second)

        assert await storage.size("k") == len(b"second")

    async def test_exists_and_size(self, storage, sample: Path) -> None:  # type: ignore[no-untyped-def]
        assert not await storage.exists("nope")

        await storage.put("k", sample)

        assert await storage.exists("k")
        assert await storage.size("k") == sample.stat().st_size

    async def test_stream_reassembles_exactly(self, storage, sample: Path) -> None:  # type: ignore[no-untyped-def]
        await storage.put("k", sample)

        received = b""
        async for chunk in storage.open_stream("k", chunk_size=1024):
            received += chunk

        assert received == sample.read_bytes()

    async def test_stream_yields_multiple_chunks(self, storage, sample: Path) -> None:  # type: ignore[no-untyped-def]
        """Проверяем, что это действительно поток, а не «прочитали всё и отдали».

        Иначе отдача 20 ГБ съела бы 20 ГБ памяти, и обнаружилось бы это
        только под нагрузкой.
        """
        await storage.put("k", sample)

        sizes = [len(chunk) async for chunk in storage.open_stream("k", chunk_size=512)]

        assert len(sizes) > 1
        assert max(sizes) <= 512


class TestMissingObjects:
    async def test_get_missing_raises(self, storage, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ObjectNotFoundError):
            await storage.get("missing", tmp_path / "x")

    async def test_size_missing_raises(self, storage) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ObjectNotFoundError):
            await storage.size("missing")

    async def test_stream_missing_raises(self, storage) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ObjectNotFoundError):
            async for _ in storage.open_stream("missing"):
                pass

    async def test_delete_missing_is_not_an_error(self, storage) -> None:  # type: ignore[no-untyped-def]
        """Удаление идемпотентно: повторный вызов не должен падать.

        Уборка мусора запускается по расписанию и вполне может удалять
        то, что удалили до неё.
        """
        await storage.delete("never-existed")
        await storage.delete("never-existed")


class TestPrefixOperations:
    async def test_delete_prefix_removes_group(self, storage, sample: Path) -> None:  # type: ignore[no-untyped-def]
        await storage.put("jobs/1/a.mp4", sample)
        await storage.put("jobs/1/b.mp4", sample)
        await storage.put("jobs/2/c.mp4", sample)

        removed = await storage.delete_prefix("jobs/1")

        assert removed == 2
        assert not await storage.exists("jobs/1/a.mp4")
        assert await storage.exists("jobs/2/c.mp4")


class TestSignedUrls:
    async def test_contract_is_url_or_none(self, storage, sample: Path) -> None:  # type: ignore[no-untyped-def]
        """Контракт: либо ссылка, либо ``None``, но не исключение.

        Роутер по этому ответу выбирает, редиректить или отдавать байты
        самому. Исключение вместо ``None`` заставило бы его знать тип
        хранилища — то есть ровно то, от чего мы уходим.
        """
        await storage.put("k", sample)

        url = await storage.signed_url("k", ttl_sec=60)

        assert url is None or url.startswith("http")


class TestHealthcheck:
    async def test_passes_when_available(self, storage) -> None:  # type: ignore[no-untyped-def]
        await storage.healthcheck()


class TestS3PresignedUrlHost:
    """Ссылка должна подписываться для адреса, по которому пойдёт клиент.

    Хост входит в подпись SigV4. Сервер ходит в MinIO по внутреннему имени
    (``http://minio:9000``), а браузер пользователя — по внешнему. Подписав
    ссылку для внутреннего адреса, мы получим 403 SignatureDoesNotMatch,
    а подменить хост в готовой ссылке нельзя — подпись развалится.

    Ошибка найдена сквозной проверкой на живом стеке: набор тестов выше
    её не ловил, потому что проверял только «ссылка или None».
    """

    async def test_signs_for_public_endpoint(self, tmp_path: Path) -> None:
        from core.infra.storage.s3 import S3Storage

        public = "http://public.example.com:9000"
        storage = S3Storage(
            bucket="highlight-tests",
            endpoint_url=MINIO_ENDPOINT,
            public_endpoint_url=public,
            access_key=MINIO_KEY,
            secret_key=MINIO_SECRET,
        )
        try:
            await storage.ensure_bucket()
        except Exception as error:
            pytest.skip(f"MinIO недоступен: {error}")

        sample = tmp_path / "s.bin"
        sample.write_bytes(b"data")
        await storage.put("signed/probe.bin", sample)

        url = await storage.signed_url("signed/probe.bin")

        assert url is not None
        assert url.startswith(public), f"подписано не для публичного адреса: {url}"
        assert "X-Amz-Signature" in url

    async def test_falls_back_to_internal_endpoint(self, tmp_path: Path) -> None:
        """Публичный адрес не задан — подписываем внутренним.

        Это нормальный случай прямого доступа к S3 из интернета.
        """
        from core.infra.storage.s3 import S3Storage

        storage = S3Storage(
            bucket="highlight-tests",
            endpoint_url=MINIO_ENDPOINT,
            access_key=MINIO_KEY,
            secret_key=MINIO_SECRET,
        )
        try:
            await storage.ensure_bucket()
        except Exception as error:
            pytest.skip(f"MinIO недоступен: {error}")

        url = await storage.signed_url("signed/probe.bin")

        assert url is not None
        assert url.startswith(MINIO_ENDPOINT)


class TestLocalStorageSpecifics:
    """Проверки, осмысленные только для локальной ФС."""

    async def test_rejects_path_traversal(self, tmp_path: Path) -> None:
        """Ключ вида ``../../etc/passwd`` не должен выводить за корень.

        Ключ приходит из БД, а туда — косвенно из данных пользователя.
        Проверяется результат разрешения пути, а не строка: строку легко
        обмануть кодированием.
        """
        storage = LocalStorage(tmp_path / "root")

        for bad_key in ("../outside.txt", "a/../../outside.txt", "/etc/passwd"):
            with pytest.raises(StorageError):
                await storage.exists(bad_key)

    async def test_local_path_avoids_copying(self, tmp_path: Path) -> None:
        """Оптимизация: воркеру незачем копировать файл сам в себя.

        Не часть порта — специфика локального хранилища, поэтому вызывающий
        код проверяет наличие метода через getattr.
        """
        storage = LocalStorage(tmp_path / "root")
        source = tmp_path / "s.bin"
        source.write_bytes(b"data")
        await storage.put("k", source)

        assert await storage.local_path("k") is not None
        assert await storage.local_path("missing") is None

    async def test_write_is_atomic(self, tmp_path: Path) -> None:
        """После put не должно оставаться .partial-файлов."""
        storage = LocalStorage(tmp_path / "root")
        source = tmp_path / "s.bin"
        source.write_bytes(b"data" * 1000)

        await storage.put("dir/k.bin", source)

        leftovers = list((tmp_path / "root").rglob("*.partial"))
        assert leftovers == []
