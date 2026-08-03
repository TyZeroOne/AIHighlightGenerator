"""Порт распознавания речи."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from core.domain.media import AudioTrack, Transcript

__all__ = ["ProgressCallback", "Transcriber"]

#: Доля выполненного, 0..1. Вызывается из потока, в котором считает модель.
type ProgressCallback = Callable[[float], None]


class Transcriber(Protocol):
    async def transcribe(
        self,
        audio: AudioTrack,
        *,
        language: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> Transcript:
        """Распознать речь.

        Контракт:
        * пословные тайминги обязательны — на них держатся субтитры и снап границ;
        * ``language=None`` означает автоопределение;
        * аудио без речи даёт пустой ``Transcript``, а не исключение;
        * ``progress`` необязателен, но объявлен в контракте: этап занимает
          четверть времени джоба, и реализация, не умеющая сообщать о ходе
          работы, оставляет пользователя перед двадцатью минутами тишины.
        """
        ...
