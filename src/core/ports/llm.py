"""Порт языковой модели.

Абстракция намеренно узкая — ровно две операции. Не «отправь произвольный
промпт», а «оцени сцены» и «сгенерируй метаданные». Так подмена провайдера
(другая модель, локальная LLM, мок в тестах) не требует переписывать промпты
в вызывающем коде: они — деталь реализации адаптера.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from core.domain.analysis import LLMVerdict, RankingOutcome, SceneSummary, VideoContext
from core.domain.entities import ClipMetadata, Highlight, Scene
from core.domain.media import Transcript

__all__ = ["LLMProvider", "SceneRanker"]


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


class SceneRanker(Protocol):
    """Опрос модели по всем сценам джоба целиком.

    Отдельный порт над :class:`LLMProvider`, а не метод в нём. Разделение
    проходит по границе ответственности: провайдер знает, как разговаривать
    с конкретным API; ранжировщик знает, как превратить сцены в сводки,
    разбить их на батчи и собрать результат. Первое меняется при смене
    поставщика, второе — нет.

    Сценарию этапа виден только этот порт: он получает готовый
    ``RankingOutcome`` и не подозревает ни про батчи, ни про промпты.
    """

    async def rank(
        self,
        scenes: Sequence[Scene],
        context: VideoContext,
    ) -> RankingOutcome:
        """Оценить сцены. Недоступность сервиса — исключение, а не пустой список."""
        ...
