"""Иерархия доменных ошибок.

Смысл иерархии не в «красоте», а в том, что по типу исключения два разных
слоя принимают решения, не зная деталей:

* ``api/`` — какой HTTP-код вернуть;
* ``workers/`` — ретраить задачу или сразу пометить провалившейся.

Поэтому у каждой ошибки есть ``http_status``, ``code`` и ``retriable``.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ConfigurationError",
    "ConflictError",
    "DependencyResolutionError",
    "DuplicateRegistrationError",
    "ExternalServiceError",
    "FFmpegError",
    "FileTooLargeError",
    "HighlightError",
    "JobCancelledError",
    "ModelLoadError",
    "NotFoundError",
    "ObjectNotFoundError",
    "ProcessingError",
    "QuotaExceededError",
    "SourceError",
    "StorageError",
    "TransientError",
    "UnsupportedMediaError",
    "ValidationError",
]


class HighlightError(Exception):
    """Базовая ошибка приложения. Всё, что мы бросаем осознанно, наследует её."""

    http_status: int = 500
    code: str = "INTERNAL_ERROR"
    retriable: bool = False

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def to_problem(self) -> dict[str, Any]:
        """Представление в формате RFC 7807 (без ``instance`` и ``trace_id`` —
        их подставляет обработчик в ``api/``)."""
        problem: dict[str, Any] = {
            "title": self.__class__.__name__,
            "status": self.http_status,
            "detail": self.message,
            "code": self.code,
        }
        if self.context:
            problem["context"] = self.context
        return problem

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r}, {self.context!r})"


# --------------------------------------------------------------- ввод
class ValidationError(HighlightError):
    """Некорректные входные данные. Ретраить бессмысленно — вход не изменится."""

    http_status = 422
    code = "VALIDATION_ERROR"


class SourceError(HighlightError):
    """Источник недоступен, повреждён или не читается."""

    http_status = 400
    code = "INVALID_SOURCE"


class UnsupportedMediaError(SourceError):
    """Формат/контейнер не поддерживается."""

    http_status = 415
    code = "UNSUPPORTED_MEDIA"


class NotFoundError(HighlightError):
    http_status = 404
    code = "NOT_FOUND"


class QuotaExceededError(HighlightError):
    http_status = 429
    code = "RATE_LIMITED"


# --------------------------------------------------------- обработка
class ProcessingError(HighlightError):
    """Сбой на этапе конвейера."""

    http_status = 500
    code = "PROCESSING_ERROR"


class FFmpegError(ProcessingError):
    code = "FFMPEG_ERROR"


class ModelLoadError(ProcessingError):
    """Не удалось загрузить ML-модель: нет файла, не хватило VRAM."""

    code = "MODEL_LOAD_ERROR"


class TransientError(ProcessingError):
    """Временный сбой: диск занят, файл заблокирован, сеть моргнула.

    Единственная ошибка обработки, которую Celery ретраит агрессивно.
    """

    code = "TRANSIENT_ERROR"
    retriable = True


class ExternalServiceError(HighlightError):
    """Внешний сервис (LLM, S3) недоступен."""

    http_status = 503
    code = "DEPENDENCY_UNAVAILABLE"
    retriable = True


class StorageError(ProcessingError):
    """Сбой файлового хранилища: нет объекта, нет прав, кончилось место."""

    code = "STORAGE_ERROR"


class ObjectNotFoundError(StorageError):
    """Объекта нет в хранилище. Отдельный класс, потому что это часто
    не ошибка приложения, а рассинхрон БД и хранилища — его надо видеть."""

    http_status = 404
    code = "OBJECT_NOT_FOUND"


class FileTooLargeError(ValidationError):
    http_status = 413
    code = "FILE_TOO_LARGE"


class JobCancelledError(HighlightError):
    """Джоб отменён пользователем, пока этап выполнялся.

    Не ошибка в обычном смысле: никто ничего не сломал, просто работу
    больше делать не надо. Отдельный тип нужен, чтобы задача Celery
    отличила отмену от настоящего сбоя и не пыталась ретраить.
    """

    http_status = 409
    code = "JOB_CANCELLED"
    retriable = False


class ConflictError(HighlightError):
    """Операция конфликтует с текущим состоянием ресурса."""

    http_status = 409
    code = "CONFLICT"


# ------------------------------------------------------ конфигурация
class ConfigurationError(HighlightError):
    """Ошибка настройки приложения: дубликат плагина, циклическая зависимость."""

    http_status = 500
    code = "CONFIGURATION_ERROR"


class DuplicateRegistrationError(ConfigurationError):
    code = "DUPLICATE_REGISTRATION"


class DependencyResolutionError(ConfigurationError):
    """Невозможно построить план анализа: цикл или недостающая зависимость."""

    code = "DEPENDENCY_RESOLUTION_ERROR"
