"""Совместимость: правила допуска медиа переехали в домен.

Изначально валидация лежала здесь, но выяснилось, что она — доменная политика:
теми же правилами пользуются сценарий загрузки, регистрация локального файла
и будущий импорт из YouTube. Модуль в ``core/domain`` не тянет зависимостей
и доступен всем слоям, а ``video_loader`` — адаптер, который другие адаптеры
импортировать не вправе.

Файл оставлен реэкспортом, чтобы не ломать привычное место импорта.
"""

from core.domain.validation import (
    CONTAINER_FAMILIES,
    container_family,
    validate_media,
    validate_upload_name,
)

__all__ = [
    "CONTAINER_FAMILIES",
    "container_family",
    "validate_media",
    "validate_upload_name",
]
