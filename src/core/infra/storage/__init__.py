"""Реализации порта ``ObjectStorage``.

``LocalStorage`` — разработка и однохостовая установка.
``S3Storage`` — прод: воркеры без общего диска, presigned-ссылки.

Обе проходят один и тот же набор интеграционных тестов
(``tests/integration/test_storage.py``) — это и есть проверка того,
что абстракция не протекает.
"""

from core.infra.storage.local import LocalStorage
from core.infra.storage.s3 import S3Storage

__all__ = ["LocalStorage", "S3Storage"]
