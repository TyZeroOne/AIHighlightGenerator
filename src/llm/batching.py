"""Разбиение сцен на батчи и параллельный опрос модели.

Два решения, ради которых существует этот модуль.

**Почему батчи вообще.** Двухчасовой фильм — это полторы тысячи сцен.
Запрос на сцену означал бы полторы тысячи системных промптов, полторы
тысячи round-trip'ов и лимит запросов в минуту, пробитый на второй минуте.
Сорок сцен в одном запросе делят накладные расходы на сорок.

**Почему не все сцены в одном запросе.** Соблазнительно, но ломается
дважды: контекст модели конечен, а главное — чем длиннее список,
тем чаще модель сбивается с нумерации и начинает выдавать вердикты
про сцены, которых в батче не было.

**Изоляция по батчам.** Упавший батч не роняет этап: остальные вердикты
сохраняются, а сцены без вердикта просто останутся с одной эвристикой.
Это то же правило, что у анализаторов на Этапе 5, и по той же причине —
частичный результат полезнее пустого.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

from core.domain.analysis import LLMVerdict, SceneSummary, VideoContext
from core.logging import get_logger
from core.ports.llm import LLMProvider

__all__ = ["BatchOutcome", "rank_in_batches", "split_batches"]

log = get_logger(__name__)

#: Сколько батчей опрашивать одновременно. Три — компромисс: у бесплатных
#: моделей лимит запросов в минуту низкий, и десяток параллельных запросов
#: гарантированно упрётся в 429, после которого ретраи растянутся дольше,
#: чем занял бы последовательный проход.
DEFAULT_CONCURRENCY: Final = 3


def split_batches(
    summaries: Sequence[SceneSummary],
    size: int,
) -> list[list[SceneSummary]]:
    """Ровные куски по ``size`` сцен, порядок сохраняется.

    Порядок важен: соседние сцены дают модели контекст («предыдущая была
    спокойным диалогом»), и перемешивание его бы уничтожило.
    """
    if size <= 0:
        raise ValueError(f"Размер батча должен быть положительным: {size}")
    return [list(summaries[index : index + size]) for index in range(0, len(summaries), size)]


@dataclass(slots=True)
class BatchOutcome:
    """Что получилось у всех батчей вместе."""

    verdicts: list[LLMVerdict] = field(default_factory=list)
    batches: int = 0
    failed: list[str] = field(default_factory=list)

    @property
    def all_failed(self) -> bool:
        """Ни один батч не дал результата.

        Отличается от «нет вердиктов»: модель могла честно оценить сцены
        и не найти ни одной интересной. Здесь же речь о том, что провайдер
        не ответил ни разу — и это повод пометить джоб ``partial``.
        """
        return self.batches > 0 and len(self.failed) == self.batches


async def rank_in_batches(
    provider: LLMProvider,
    summaries: Sequence[SceneSummary],
    context: VideoContext,
    *,
    batch_size: int,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> BatchOutcome:
    """Опросить модель по всем батчам и собрать вердикты.

    Ошибка одного батча записывается и не мешает остальным: ``gather``
    вызывается с ``return_exceptions=True`` — ровно как в сценарии анализа
    признаков.
    """
    batches = split_batches(summaries, batch_size)
    if not batches:
        return BatchOutcome()

    semaphore = asyncio.Semaphore(max(concurrency, 1))

    async def run(index: int, batch: list[SceneSummary]) -> Sequence[LLMVerdict]:
        async with semaphore:
            log.debug("llm.batch.start", batch=index, scenes=len(batch))
            return await provider.rank_scenes(batch, context)

    results = await asyncio.gather(
        *(run(index, batch) for index, batch in enumerate(batches)),
        return_exceptions=True,
    )

    outcome = BatchOutcome(batches=len(batches))
    for index, result in enumerate(results):
        if isinstance(result, BaseException):
            message = f"{type(result).__name__}: {result}"
            outcome.failed.append(message)
            log.warning("llm.batch.failed", batch=index, error=message)
            continue
        outcome.verdicts.extend(result)

    log.info(
        "llm.batches.done",
        batches=len(batches),
        verdicts=len(outcome.verdicts),
        failed=len(outcome.failed),
    )
    return outcome
