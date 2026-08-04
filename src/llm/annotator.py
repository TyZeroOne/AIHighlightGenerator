"""Реализация порта ``ClipAnnotator``: клипы → заголовки и хештеги.

Работы здесь меньше, чем у ранжировщика: батчить нечего, каждый клип —
свой запрос. Зато появляется другое.

## Почему по одному клипу, а не пачкой

Ранжирование сцен идёт батчами, потому что сцен сотни и каждая описывается
парой строк. Клипов — единицы, и каждому нужен свой заголовок, своё
описание и свои хештеги. Батч из десяти таких заданий даёт ответ, в котором
модель начинает повторяться: заголовки становятся вариациями одного,
хештеги — общим набором на всех. Отдельные запросы дороже на системный
промпт (он один и тот же, поэтому кешируется) и заметно лучше по существу.

## Отказ по одному клипу не отменяет остальные

Клип без метаданных — это клип без заголовка, а не потерянный клип.
Ошибки собираются в список, результат отдаётся частичным, а решать,
что с этим делать, будет сценарий этапа.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from uuid import UUID

from core.domain.analysis import AnnotationOutcome
from core.domain.entities import Clip, ClipMetadata, Highlight
from core.domain.media import Transcript
from core.logging import get_logger
from core.ports.llm import LLMProvider

__all__ = ["DEFAULT_CONCURRENCY", "ClipMetadataAnnotator"]

log = get_logger(__name__)

#: Сколько клипов опрашивается одновременно. Три — компромисс между
#: скоростью и лимитами провайдера: у бесплатных моделей OpenRouter
#: ограничение по запросам в минуту достигается быстрее, чем кончается
#: терпение.
DEFAULT_CONCURRENCY = 3


class ClipMetadataAnnotator:
    """Аннотатор, опрашивающий модель по одному клипу."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        concurrency: int = DEFAULT_CONCURRENCY,
    ) -> None:
        self._provider = provider
        self._concurrency = max(concurrency, 1)

    async def annotate(
        self,
        clips: Sequence[Clip],
        transcript: Transcript,
        language: str | None = None,
    ) -> AnnotationOutcome:
        if not clips:
            return AnnotationOutcome()

        semaphore = asyncio.Semaphore(self._concurrency)
        results: dict[UUID, ClipMetadata] = {}
        failed: list[str] = []

        async def one(clip: Clip) -> None:
            async with semaphore:
                try:
                    metadata = await self._provider.generate_metadata(
                        _to_highlight(clip), transcript, language
                    )
                except Exception as error:
                    log.warning("llm.metadata_failed", rank=clip.rank, error=str(error))
                    failed.append(f"клип {clip.rank}: {type(error).__name__}: {error}")
                    return
            results[clip.id] = metadata

        await asyncio.gather(*(one(clip) for clip in clips))

        usage = getattr(self._provider, "usage", None)
        log.info("llm.metadata.done", clips=len(clips), got=len(results), failed=len(failed))
        return AnnotationOutcome(
            metadata=results,
            requested=len(clips),
            failed=tuple(failed),
            cost=dict(usage.as_dict()) if usage is not None else {},
        )


def _to_highlight(clip: Clip) -> Highlight:
    """Клип → то, что понимает провайдер.

    Провайдер описан в терминах ``Highlight`` — отобранного момента, —
    и это правильнее, чем в терминах клипа: метаданные можно захотеть
    ещё до того, как файл нарезан.
    """
    return Highlight(
        range=clip.range,
        final_score=clip.score,
        category=clip.category,
        reason=clip.reason,
        scene_ids=clip.scene_ids,
    )
