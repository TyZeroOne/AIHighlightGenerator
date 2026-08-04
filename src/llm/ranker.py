"""Реализация порта ``SceneRanker``: сцены → вердикты.

Здесь сходится всё, что этап ранжирования делает вокруг модели:

1. фильтр — какие сцены вообще стоит показывать (:mod:`llm.summarizer`);
2. сводка — во что превратить сцену, чтобы модель поняла её за 150 токенов;
3. батчи — как разложить сотни сцен по запросам (:mod:`llm.batching`);
4. расход — сколько это стоило.

Сценарий этапа ничего из этого не видит: он вызывает ``rank`` и получает
``RankingOutcome``. Смысл разделения в том, что все четыре пункта —
это знание про **конкретный способ разговаривать с моделью**, а сценарий
описывает **место этапа в конвейере**, и меняются они по разным поводам.
"""

from __future__ import annotations

from collections.abc import Sequence

from core.domain.analysis import RankingOutcome, VideoContext
from core.domain.entities import Scene
from core.logging import get_logger
from core.ports.llm import LLMProvider
from llm.batching import DEFAULT_CONCURRENCY, rank_in_batches
from llm.summarizer import DEFAULT_MIN_SIGNAL, prepare

__all__ = ["BatchRanker"]

log = get_logger(__name__)


class BatchRanker:
    """Ранжировщик, опрашивающий модель батчами."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        batch_size: int = 40,
        min_signal: float = DEFAULT_MIN_SIGNAL,
        concurrency: int = DEFAULT_CONCURRENCY,
    ) -> None:
        self._provider = provider
        self._batch_size = batch_size
        self._min_signal = min_signal
        self._concurrency = concurrency

    async def rank(
        self,
        scenes: Sequence[Scene],
        context: VideoContext,
    ) -> RankingOutcome:
        summaries = prepare(scenes, min_signal=self._min_signal)
        if not summaries:
            return RankingOutcome()

        log.info(
            "llm.rank.prepared",
            scenes=len(scenes),
            sent=len(summaries),
            saved=len(scenes) - len(summaries),
        )

        outcome = await rank_in_batches(
            self._provider,
            summaries,
            context,
            batch_size=self._batch_size,
            concurrency=self._concurrency,
        )

        usage = getattr(self._provider, "usage", None)
        return RankingOutcome(
            verdicts=tuple(outcome.verdicts),
            scenes_sent=len(summaries),
            batches=outcome.batches,
            failed=tuple(outcome.failed),
            cost=dict(usage.as_dict()) if usage is not None else {},
        )
