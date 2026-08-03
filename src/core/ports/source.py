"""Порт источника видео — главная точка расширения проекта.

Реализовав этот протокол, можно подключить YouTube, Twitch, Kick, RTMP-эфир
или сетевую папку, не тронув ни одной строчки в конвейере, скоринге и API.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import ClassVar, Protocol

from core.domain.media import LocalMedia, MediaInfo, SourceDescriptor
from core.domain.values import MediaWindow, TimeRange

__all__ = ["MediaProbe", "SourceProvider"]


class MediaProbe(Protocol):
    """Чтение технических параметров медиафайла.

    Отдельный порт, потому что этим занимается внешний бинарник (``ffprobe``),
    а сценариям нужен только результат. В тестах подставляется функция,
    возвращающая заранее подготовленный ``MediaInfo``, — и весь сценарий
    загрузки проверяется без единого видеофайла.
    """

    async def probe(self, path: Path) -> MediaInfo: ...


class SourceProvider(Protocol):
    """Поставщик медиа.

    Жизненный цикл: ``supports`` → ``describe`` → ``windows`` → ``acquire``
    (по окну) → ... → ``release``.
    """

    #: Схема URI, которую обслуживает провайдер: "file", "upload", "youtube", ...
    scheme: ClassVar[str]

    @classmethod
    def supports(cls, uri: str) -> bool:
        """Быстрая проверка без сети — резолвер выбирает провайдер по ней.

        Именно ``classmethod``, а не обычный метод: резолвер должен уметь
        спросить «ты умеешь это?» до того, как создаст экземпляр (создание
        может требовать конфига, сетевого клиента, ключей).
        """
        ...

    async def describe(self, uri: str) -> SourceDescriptor:
        """Метаданные ДО скачивания: длительность, live или нет, можно ли перематывать."""
        ...

    def windows(self, uri: str) -> AsyncIterator[MediaWindow]:
        """Окна, которые нужно обработать.

        Для файла — один элемент на всю длительность.
        Для live-эфира — бесконечный генератор окон по мере поступления данных.
        Именно здесь проходит граница между «файлом» и «стримом»; выше по
        стеку разницы уже нет.
        """
        ...

    async def acquire(self, uri: str, window: TimeRange | None = None) -> LocalMedia:
        """Сделать медиа доступным на диске воркера.

        Для локального файла — вернуть путь как есть (``owned=False``).
        Для сети — скачать сегмент во временный файл (``owned=True``).
        """
        ...

    async def release(self, media: LocalMedia) -> None:
        """Убрать за собой. Для ``owned=False`` — ничего не делать."""
        ...
