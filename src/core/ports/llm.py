"""Порт языковой модели.

Абстракция намеренно узкая — ровно две операции. Не «отправь произвольный
промпт», а «оцени сцены» и «сгенерируй метаданные». Так подмена провайдера
(другая модель, локальная LLM, мок в тестах) не требует переписывать промпты
в вызывающем коде: они — деталь реализации адаптера.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from core.domain.analysis import LLMVerdict, SceneSummary, VideoContext
from core.domain.entities import ClipMetadata, Highlight
from core.domain.media import Transcript

__all__ = ["LLMProvider"]


class LLMProvider(Protocol):
    async def rank_scenes(
        self,
        scenes: Sequence[SceneSummary],
        context: VideoContext,
    ) -> Sequence[LLMVerdict]:
        """Оценить интересность сцен.

        Контракт:
        * возвращает вердикты только по тем сценам, которые смогла оценить —
          пустой список является допустимым ответом (например, при отказе модели);
        * ``LLMVerdict.scene_index`` обязан соответствовать ``SceneSummary.index``;
        * недоступность сервиса выражается через ``ExternalServiceError``,
          а не через пустой список.
        """
        ...

    async def generate_metadata(
        self,
        highlight: Highlight,
        transcript: Transcript,
        language: str | None = None,
    ) -> ClipMetadata:
        """Заголовок, описание и хештеги для клипа на языке исходной речи."""
        ...
