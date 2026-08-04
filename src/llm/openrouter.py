"""Реализация порта ``LLMProvider`` поверх OpenRouter.

OpenRouter — шлюз к сотням моделей по одному OpenAI-совместимому протоколу,
и среди них есть бесплатные. Для проекта это означает возможность разрабатывать
и прогонять этап целиком, не тратя денег, — а перейти на Claude потом заменой
одной строки в конфиге.

## Чем это отличается от Claude — и почему это важно знать

**Схему соблюдает модель, а не API.** У Claude `output_config.format`
гарантирует структуру ответа: невалидный JSON физически невозможен.
Здесь `response_format` с флагом `strict` — это просьба, которую модель
может проигнорировать. Замер на пяти бесплатных моделях: одна из пяти
вернула markdown-список вместо JSON и категории, которых нет в перечислении.

Отсюда три следствия, определившие устройство этого модуля:

1. ответ **всегда** валидируется на нашей стороне (Pydantic);
2. невалидный ответ — не исключение, а повод повторить запрос;
3. вердикт с несуществующим индексом сцены или чужими тайм-кодами
   отбрасывается поштучно, а не роняет весь батч.

**Кеширования промптов нет.** У Claude системный префикс кешируется и со
второго батча стоит десятую часть цены. Здесь его нет вовсе, поэтому
порядок блоков (инструкция → контекст → батч) сохранён не ради экономии,
а чтобы код не пришлось переписывать при переезде.

**Batch API нет.** Батчи здесь — наши собственные: несколько сцен в одном
запросе. Асинхронной пакетной обработки со скидкой, как у Claude, нет.

## Почему httpx напрямую, а не SDK

OpenRouter совместим с протоколом OpenAI, и его можно звать SDK от OpenAI.
Но нам нужны ровно два эндпоинта и полный контроль над заголовками —
тащить ради этого чужой клиент с его моделью ретраев и своими типами
означает больше кода, а не меньше.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from core.domain.analysis import LLMVerdict, SceneSummary, VideoContext
from core.domain.entities import ClipMetadata, Highlight
from core.domain.errors import ExternalServiceError
from core.domain.media import Transcript
from core.logging import get_logger
from llm.prompts import METADATA_SYSTEM_PROMPT, RANKING_SYSTEM_PROMPT
from llm.schemas import ClipMetadataResponse, RankingResponse
from llm.summarizer import format_batch, format_video_context

__all__ = ["LLMUsage", "OpenRouterProvider"]

log = get_logger(__name__)

#: Коды, при которых имеет смысл повторить запрос. 429 — лимит запросов
#: (у бесплатных моделей он низкий), 5xx — временные сбои шлюза.
_RETRYABLE: Final = frozenset({408, 409, 429, 500, 502, 503, 504})

#: Допуск при сверке тайм-кодов ответа с нашей сценой (док 06, §6.10).
_TIMECODE_TOLERANCE: Final = 2.0


@dataclass(slots=True)
class LLMUsage:
    """Сколько израсходовано за джоб.

    Копится в провайдере, а не возвращается из каждого вызова: порт
    ``LLMProvider`` описывает смысл операции («оцени сцены»), и навешивать
    на него счётчик токенов значит смешивать два разных вопроса.
    """

    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: Стоимость в долларах. У бесплатных моделей ноль — но поле есть,
    #: потому что оно понадобится ровно в тот день, когда бесплатная
    #: модель кончится, а джобы уже будут идти.
    cost_usd: float = 0.0
    #: Сколько ответов пришлось перезапросить из-за нарушенной схемы.
    #: Это метрика провайдера, а не ошибка: см. заголовок модуля.
    schema_retries: int = 0
    failed: list[str] = field(default_factory=list)

    def merge(self, payload: Mapping[str, Any]) -> None:
        usage = payload.get("usage") or {}
        self.requests += 1
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        self.cost_usd += float(usage.get("cost") or 0.0)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "requests": self.requests,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "schema_retries": self.schema_retries,
        }


class OpenRouterProvider:
    """Ранжирование сцен и метаданные клипов через OpenRouter."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        max_tokens: int = 16000,
        timeout_sec: float = 180.0,
        max_retries: int = 3,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._max_tokens = max_tokens
        self._timeout = timeout_sec
        self._max_retries = max_retries
        self.usage = LLMUsage()

    # ------------------------------------------------------------ порт
    async def rank_scenes(
        self,
        scenes: Sequence[SceneSummary],
        context: VideoContext,
    ) -> Sequence[LLMVerdict]:
        """Оценить интересность сцен.

        Пустой список — допустимый ответ (порт, док 06 §6.7): модель могла
        отказаться или вернуть мусор. Недоступность сервиса, наоборот,
        выражается исключением — иначе «сервис лежит» и «сцены скучные»
        стали бы неразличимы.
        """
        if not scenes:
            return []

        payload = await self._complete(
            system=[RANKING_SYSTEM_PROMPT, format_video_context(context)],
            user=format_batch(scenes),
            schema=RankingResponse,
        )
        if payload is None:
            return []

        return self._validate(payload, scenes)

    async def generate_metadata(
        self,
        highlight: Highlight,
        transcript: Transcript,
        language: str | None = None,
    ) -> ClipMetadata:
        """Заголовок, описание и хештеги для клипа."""
        excerpt = transcript.text_within(highlight.range)
        block = "\n".join(
            [
                f"Категория: {highlight.category}",
                f"Чем интересен: {highlight.reason}",
                f"Длительность: {highlight.range.duration:.0f} с",
                f"Язык: {language or 'по речи в клипе'}",
                "",
                f"Расшифровка клипа:\n{excerpt or '(речи нет)'}",
            ]
        )

        payload = await self._complete(
            system=[METADATA_SYSTEM_PROMPT],
            user=block,
            schema=ClipMetadataResponse,
        )
        if payload is None:
            raise ExternalServiceError(
                "Модель не вернула метаданные клипа в ожидаемом формате",
                service="openrouter",
            )

        return ClipMetadata(
            title=payload.title,
            description=payload.description,
            hashtags=tuple(payload.hashtags),
            hook=payload.hook,
        )

    # ------------------------------------------------------------ внутреннее
    def _validate(
        self,
        response: RankingResponse,
        scenes: Sequence[SceneSummary],
    ) -> list[LLMVerdict]:
        """Отсев вердиктов, которым нельзя верить.

        Проверок две, и обе про одно: модель могла сбиться с нумерации.
        Индекс, которого мы не присылали, — выдумка. Тайм-коды, не похожие
        на нашу сцену, означают, что вердикт относится к другой сцене,
        и засчитать его как «неточность» нельзя.
        """
        by_index = {summary.index: summary for summary in scenes}
        verdicts: list[LLMVerdict] = []

        for item in response.verdicts:
            summary = by_index.get(item.scene_index)
            if summary is None:
                log.warning("llm.unknown_scene", scene_index=item.scene_index)
                continue
            if not item.timecodes_match(summary.range, tolerance=_TIMECODE_TOLERANCE):
                log.warning(
                    "llm.timecode_mismatch",
                    scene_index=item.scene_index,
                    expected=str(summary.range),
                    got=f"{item.start}–{item.end}",
                )
                continue
            verdicts.append(item.to_domain(summary.range))

        # Дубликаты по индексу: оставляем первый. Модель иногда оценивает
        # сцену дважды, и второй вердикт обычно копия первого.
        seen: set[int] = set()
        unique: list[LLMVerdict] = []
        for verdict in verdicts:
            if verdict.scene_index in seen:
                continue
            seen.add(verdict.scene_index)
            unique.append(verdict)

        log.debug("llm.verdicts", got=len(response.verdicts), kept=len(unique))
        return unique

    async def _complete(
        self,
        *,
        system: Sequence[str],
        user: str,
        schema: type[RankingResponse] | type[ClipMetadataResponse],
    ) -> Any:
        """Запрос со строгой схемой, ретраями и разбором ответа.

        Возвращает ``None``, если после всех попыток ответ так и не сошёлся
        со схемой: для ранжирования это означает «нет вердиктов», и джоб
        продолжится по эвристике.
        """
        body = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [
                # Порядок от самого стабильного к самому изменчивому.
                # Здесь это ничего не экономит, но у Claude ровно этот
                # порядок включает кеширование префикса.
                *({"role": "system", "content": block} for block in system),
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__.lower(),
                    "strict": True,
                    "schema": schema.model_json_schema(),
                },
            },
        }

        last_error: str | None = None
        for attempt in range(1, self._max_retries + 1):
            payload = await self._post("/chat/completions", body, attempt=attempt)
            self.usage.merge(payload)

            text = _extract_text(payload)
            if text is None:
                last_error = "пустой ответ"
            else:
                try:
                    return schema.model_validate_json(text)
                except Exception as error:
                    last_error = f"{type(error).__name__}: {error}"

            # Схема не сошлась — это не сбой сети, повторять сразу можно.
            self.usage.schema_retries += 1
            log.warning(
                "llm.schema_mismatch",
                model=self._model,
                attempt=attempt,
                error=last_error,
            )

        self.usage.failed.append(last_error or "неизвестная ошибка")
        return None

    async def _post(self, path: str, body: Mapping[str, Any], *, attempt: int) -> Mapping[str, Any]:
        """POST с повторами по кодам, которые имеет смысл повторять."""
        try:
            import httpx
        except ImportError as error:  # pragma: no cover — зависит от окружения
            raise ExternalServiceError(
                "httpx не установлен. Поставьте extra 'worker'.",
                service="openrouter",
            ) from error

        url = f"{self._base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            # OpenRouter просит представляться: по этим заголовкам он
            # показывает приложение в своей статистике.
            "HTTP-Referer": "https://github.com/ai-highlight-generator",
            "X-Title": "AI Highlight Generator",
        }

        delay = 1.0
        last: Exception | None = None
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for retry in range(self._max_retries):
                try:
                    response = await client.post(url, json=body, headers=headers)
                except httpx.HTTPError as error:
                    last = error
                else:
                    if response.status_code < 400:
                        return dict(response.json())
                    if response.status_code not in _RETRYABLE:
                        raise ExternalServiceError(
                            f"OpenRouter вернул {response.status_code}: {response.text[:300]}",
                            service="openrouter",
                            status_code=response.status_code,
                        )
                    last = ExternalServiceError(
                        f"OpenRouter вернул {response.status_code}",
                        service="openrouter",
                        status_code=response.status_code,
                    )

                # Джиттер обязателен: без него параллельные батчи после
                # общего 429 проснутся одновременно и получат его снова.
                await asyncio.sleep(delay + random.uniform(0, 0.5))
                delay *= 2
                log.warning("llm.retry", attempt=attempt, retry=retry + 1, error=str(last))

        raise ExternalServiceError(
            f"OpenRouter недоступен после {self._max_retries} попыток: {last}",
            service="openrouter",
        ) from last


def _extract_text(payload: Mapping[str, Any]) -> str | None:
    """Текст ответа из OpenAI-совместимого конверта.

    Аккуратно, а не ``payload["choices"][0]["message"]["content"]``: при
    ошибке провайдера или отказе модели любого из этих ключей может не быть,
    и тогда вместо понятного «схема не сошлась» получился бы ``KeyError``
    в середине джоба.
    """
    choices = payload.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    # Некоторые модели отдают content списком блоков, как в Anthropic-формате.
    if isinstance(content, list):
        parts = [str(part.get("text", "")) for part in content if isinstance(part, dict)]
        joined = "".join(parts).strip()
        return joined or None
    return None
