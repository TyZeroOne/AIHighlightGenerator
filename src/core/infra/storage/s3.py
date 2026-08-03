"""Хранилище на S3-совместимом сервисе (AWS S3, MinIO, Ceph).

Главное, что даёт S3 по сравнению с локальной ФС: воркерам **не нужен общий
диск с API**. Их можно разнести по разным машинам и масштабировать независимо.
Второе — presigned-ссылки: клиент скачивает клип напрямую из хранилища,
а не через наш процесс.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.domain.errors import ExternalServiceError, ObjectNotFoundError
from core.logging import get_logger

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
    from mypy_boto3_s3.type_defs import ObjectIdentifierTypeDef

__all__ = ["S3Storage"]

log = get_logger(__name__)

_NOT_FOUND_CODES = frozenset({"404", "NoSuchKey", "NotFound"})


class S3Storage:
    """Реализация ``ObjectStorage`` поверх boto3.

    **Почему boto3 в пуле потоков, а не асинхронный клиент.**
    Асинхронные обёртки (`aioboto3`, `aiobotocore`) существуют, но тянут
    собственный форк botocore и отстают от него по версиям. Наши обращения
    к хранилищу — единицы в секунду и почти всегда крупные файлы, где
    время уходит на сеть, а не на переключение задач. `asyncio.to_thread`
    даёт ту же неблокирующесть при одной зависимости вместо трёх.

    Если профилирование покажет, что пул потоков стал узким местом, замена
    затронет только этот файл — интерфейс не изменится.
    """

    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: str | None = None,
        public_endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        region: str = "us-east-1",
        signed_url_ttl: int = 900,
    ) -> None:
        import boto3
        from botocore.config import Config

        self._bucket = bucket
        self._signed_url_ttl = signed_url_ttl

        config = Config(
            # path-style обязателен для MinIO: virtual-host-style требует
            # поддомена вида bucket.minio, которого в docker-сети нет.
            s3={"addressing_style": "path"},
            retries={"max_attempts": 3, "mode": "standard"},
            signature_version="s3v4",
        )

        def make_client(url: str | None) -> S3Client:
            # Аргументы перечислены явно, а не через **kwargs: у boto3.client
            # десятки перегрузок по имени сервиса, и распаковка словаря
            # ломает их разрешение — mypy теряет тип и возвращает Any.
            return boto3.client(
                "s3",
                endpoint_url=url,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region,
                config=config,
            )

        self._client: S3Client = make_client(endpoint_url)

        # Отдельный клиент для подписи ссылок.
        #
        # Хост входит в подпись SigV4. Сервер ходит в хранилище по внутреннему
        # адресу (`http://minio:9000`), а браузер пользователя — по внешнему
        # (`https://s3.example.com`). Ссылка, подписанная для внутреннего
        # хоста, при обращении по внешнему даёт 403 SignatureDoesNotMatch,
        # и подменить хост в готовой ссылке нельзя — подпись развалится.
        #
        # Поэтому ссылки подписываются клиентом, привязанным к **публичному**
        # адресу. Если он не задан, оба совпадают — так бывает при прямом
        # доступе к S3 из интернета.
        self._signing_client: S3Client = (
            make_client(public_endpoint_url)
            if public_endpoint_url and public_endpoint_url != endpoint_url
            else self._client
        )

    # -------------------------------------------------------- служебное
    @staticmethod
    def _is_not_found(error: Exception) -> bool:
        response = getattr(error, "response", None)
        if not isinstance(response, dict):
            return False
        code = str(response.get("Error", {}).get("Code", ""))
        status = str(response.get("ResponseMetadata", {}).get("HTTPStatusCode", ""))
        return code in _NOT_FOUND_CODES or status == "404"

    def _wrap(self, error: Exception, key: str) -> Exception:
        if self._is_not_found(error):
            return ObjectNotFoundError(f"Объект не найден: {key}", key=key)
        return ExternalServiceError(f"Ошибка S3: {error}", key=key)

    async def ensure_bucket(self) -> None:
        """Создать бакет, если его нет. Для локального MinIO это удобно;
        в проде бакет обычно создаётся инфраструктурой заранее."""

        def _ensure() -> None:
            from botocore.exceptions import ClientError

            try:
                self._client.head_bucket(Bucket=self._bucket)
            except ClientError as error:
                if not self._is_not_found(error):
                    raise
                self._client.create_bucket(Bucket=self._bucket)

        await asyncio.to_thread(_ensure)

    # ------------------------------------------------------------ запись
    async def put(self, key: str, source: Path, *, content_type: str | None = None) -> str:
        extra: dict[str, Any] = {"ContentType": content_type} if content_type else {}

        def _upload() -> None:
            # upload_file сам разбивает большие файлы на multipart-части
            # и грузит их параллельно — вручную это делать не нужно.
            self._client.upload_file(str(source), self._bucket, key, ExtraArgs=extra or None)

        try:
            await asyncio.to_thread(_upload)
        except Exception as error:
            raise self._wrap(error, key) from error

        log.debug("storage.put", key=key, backend="s3")
        return key

    # ------------------------------------------------------------ чтение
    async def get(self, key: str, destination: Path) -> Path:
        def _download() -> None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._client.download_file(self._bucket, key, str(destination))

        try:
            await asyncio.to_thread(_download)
        except Exception as error:
            raise self._wrap(error, key) from error
        return destination

    async def _stream(self, key: str, chunk_size: int) -> AsyncIterator[bytes]:
        try:
            response = await asyncio.to_thread(
                self._client.get_object, Bucket=self._bucket, Key=key
            )
        except Exception as error:
            raise self._wrap(error, key) from error

        body = response["Body"]
        try:
            while chunk := await asyncio.to_thread(body.read, chunk_size):
                yield chunk
        finally:
            await asyncio.to_thread(body.close)

    def open_stream(self, key: str, *, chunk_size: int = 1 << 20) -> AsyncIterator[bytes]:
        return self._stream(key, chunk_size)

    # ------------------------------------------------------- метаданные
    async def exists(self, key: str) -> bool:
        try:
            await asyncio.to_thread(self._client.head_object, Bucket=self._bucket, Key=key)
        except Exception as error:
            if self._is_not_found(error):
                return False
            raise self._wrap(error, key) from error
        return True

    async def size(self, key: str) -> int:
        try:
            head = await asyncio.to_thread(self._client.head_object, Bucket=self._bucket, Key=key)
        except Exception as error:
            raise self._wrap(error, key) from error
        return int(head["ContentLength"])

    async def delete(self, key: str) -> None:
        def _delete() -> None:
            # delete_object в S3 идемпотентен: удаление отсутствующего ключа
            # возвращает успех. Отдельная проверка не нужна.
            self._client.delete_object(Bucket=self._bucket, Key=key)

        try:
            await asyncio.to_thread(_delete)
        except Exception as error:
            raise self._wrap(error, key) from error

    async def delete_prefix(self, prefix: str) -> int:
        def _delete_all() -> int:
            paginator = self._client.get_paginator("list_objects_v2")
            removed = 0
            for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                objects: list[ObjectIdentifierTypeDef] = [
                    {"Key": item["Key"]} for item in page.get("Contents", [])
                ]
                if not objects:
                    continue
                # Пакетное удаление: до 1000 ключей за запрос вместо 1000 запросов.
                self._client.delete_objects(Bucket=self._bucket, Delete={"Objects": objects})
                removed += len(objects)
            return removed

        try:
            return await asyncio.to_thread(_delete_all)
        except Exception as error:
            raise self._wrap(error, prefix) from error

    async def signed_url(self, key: str, *, ttl_sec: int | None = None) -> str | None:
        def _sign() -> str:
            return self._signing_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=ttl_sec or self._signed_url_ttl,
            )

        try:
            return await asyncio.to_thread(_sign)
        except Exception as error:
            raise self._wrap(error, key) from error

    async def healthcheck(self) -> None:
        try:
            await asyncio.to_thread(self._client.head_bucket, Bucket=self._bucket)
        except Exception as error:
            raise ExternalServiceError(f"Бакет {self._bucket} недоступен: {error}") from error
