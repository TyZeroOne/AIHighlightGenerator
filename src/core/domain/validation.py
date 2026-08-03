"""Правила допуска медиа: формат, длительность, размер.

Это **доменная политика**, а не деталь загрузчика: «какие контейнеры мы
принимаем» — решение о продукте, одинаковое для загрузки через HTTP,
регистрации локального файла и будущего импорта из YouTube.

Модуль чистый: ни ffprobe, ни файловой системы. На вход — уже разобранный
``MediaInfo``, на выход — исключение или молчание.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Final

from core.domain.errors import FileTooLargeError, UnsupportedMediaError, ValidationError
from core.domain.media import MediaInfo

__all__ = [
    "CONTAINER_FAMILIES",
    "container_family",
    "validate_media",
    "validate_upload_name",
]

#: Расширение → семейство контейнеров.
#: ``.mp4`` и ``.mov`` — это один и тот же формат ISO BMFF, и ffprobe
#: не различает их: для обоих ``format_name`` равен
#: ``"mov,mp4,m4a,3gp,3g2,mj2"``. Поэтому сравнивать нужно семейства,
#: а не расширения, иначе честный ``.mov`` был бы отвергнут как «не mp4».
CONTAINER_FAMILIES: Final[dict[str, str]] = {
    "mp4": "mp4",
    "mov": "mp4",
    "m4v": "mp4",
    "mkv": "mkv",
    "webm": "mkv",
}

MAX_FILENAME_LENGTH: Final = 255


def container_family(extension_or_name: str) -> str | None:
    """``"movie.MOV"`` или ``"mov"`` → ``"mp4"``."""
    token = extension_or_name.rsplit(".", 1)[-1].strip().lower()
    return CONTAINER_FAMILIES.get(token)


def validate_upload_name(filename: str, allowed: frozenset[str]) -> str:
    """Предварительная проверка по имени — до чтения байтов.

    Не заменяет проверку содержимого, но экономит 20 ГБ трафика и места
    на диске, если пользователь ошибся файлом. Дешёвая проверка первой,
    дорогая — второй.
    """
    # PurePosixPath, а не Path: имя приходит от клиента, и разбирать его
    # правилами ФС сервера неверно. На Windows Path("a/b.mp4").name дал бы
    # "b.mp4", а Path("a\\b.mp4").name — то же самое, скрыв попытку обхода.
    name = PurePosixPath(filename.replace("\\", "/")).name.strip()
    if not name or name in {".", ".."}:
        raise ValidationError(f"Некорректное имя файла: {filename!r}", filename=filename[:80])
    if len(name) > MAX_FILENAME_LENGTH:
        raise ValidationError(
            f"Имя файла длиннее {MAX_FILENAME_LENGTH} символов", filename=name[:80]
        )

    suffix = PurePosixPath(name).suffix.lstrip(".").lower()
    if not suffix:
        raise UnsupportedMediaError(
            f"У файла нет расширения. Разрешены: {', '.join(sorted(allowed))}",
            filename=name,
        )
    if suffix not in allowed:
        raise UnsupportedMediaError(
            f"Контейнер {suffix!r} не поддерживается. Разрешены: {', '.join(sorted(allowed))}",
            container=suffix,
            allowed=sorted(allowed),
        )
    return suffix


def validate_media(
    info: MediaInfo,
    *,
    allowed_containers: frozenset[str],
    max_duration_sec: int,
    max_bytes: int,
    filename: str | None = None,
) -> None:
    """Проверка после ffprobe: формат подтверждён содержимым, лимиты соблюдены."""
    if not info.has_video and not info.has_audio:
        raise UnsupportedMediaError("В файле нет ни видео-, ни аудиопотока")

    allowed_families = {
        family
        for extension in allowed_containers
        if (family := CONTAINER_FAMILIES.get(extension)) is not None
    }
    if info.container is not None and info.container not in allowed_families:
        raise UnsupportedMediaError(
            f"Содержимое файла — контейнер {info.container!r}, "
            f"а разрешены: {', '.join(sorted(allowed_containers))}",
            detected=info.container,
            allowed=sorted(allowed_containers),
            filename=filename,
        )

    if info.duration_sec > max_duration_sec:
        raise ValidationError(
            f"Длительность {info.duration_sec / 3600:.2f} ч превышает лимит "
            f"{max_duration_sec / 3600:.2f} ч",
            duration_sec=round(info.duration_sec, 3),
            max_duration_sec=max_duration_sec,
        )

    if info.size_bytes is not None and info.size_bytes > max_bytes:
        raise FileTooLargeError(
            f"Размер {info.size_bytes / 1024**3:.2f} ГиБ превышает лимит "
            f"{max_bytes / 1024**3:.2f} ГиБ",
            size_bytes=info.size_bytes,
            max_bytes=max_bytes,
        )
