"""Схемы ответов модели.

Это не «валидация на всякий случай», а **контракт, который модель физически
не может нарушить**: схема уходит в запрос параметром `output_config.format`,
и API гарантирует структуру ответа. Pydantic здесь второй рубеж — он ловит
не «модель придумала другое поле», а расхождения между нашим представлением
и тем, что реально пришло по сети.

> **Почему схема живёт в адаптере, а не в домене**
>
> `LLMVerdict` в `core/domain` — это то, что нужно системе: индекс сцены,
> оценка, категория, обоснование. `SceneVerdict` здесь — то, в чём удобно
> разговаривать с конкретным API: строковые тайм-коды, ограничения длины,
> `Literal` вместо перечисления домена.
>
> Разница не косметическая. Сменится провайдер — изменится этот файл;
> домен и скоринг не заметят. Обратное тоже верно: добавим в домен поле,
> которое модель не выдаёт, — здесь не поменяется ничего.

## Тайм-коды строками

В исходном ТЗ они записаны как `"start": 00:15:21` — это невалидный JSON,
такой документ не разберёт ни один парсер. Здесь строка `"HH:MM:SS[.mmm]"`
с проверкой по регулярному выражению; в секунды она превращается на границе
слоя (:meth:`SceneVerdict.to_domain`).
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, Field, field_validator

from core.domain.analysis import LLMVerdict
from core.domain.timecode import parse_timecode
from core.domain.values import TimeRange

__all__ = [
    "CATEGORIES",
    "ClipMetadataResponse",
    "RankingResponse",
    "SceneVerdict",
]

#: Категории момента. Закрытый список, а не свободная строка: по категории
#: работает MMR на Этапе 7 («не больше четырёх клипов одной категории
#: в топ-10»), и синонимы вроде «comedy»/«funny» сломали бы это молча.
CATEGORIES: Final = (
    "action",
    "funny",
    "emotional",
    "plot_twist",
    "dialogue",
    "tension",
    "romantic",
    "informative",
    "other",
)

Category = Literal[
    "action",
    "funny",
    "emotional",
    "plot_twist",
    "dialogue",
    "tension",
    "romantic",
    "informative",
    "other",
]

#: Формат тайм-кода в ответе. Часы — ровно две цифры: столько же, сколько
#: модель видит во входных данных, и лишняя свобода здесь ни к чему.
_TIMECODE_PATTERN: Final = r"^\d{2}:\d{2}:\d{2}(\.\d{1,3})?$"


class SceneVerdict(BaseModel):
    """Оценка одной сцены."""

    #: Ключ надёжности всего этапа.
    #:
    #: В ТЗ этого поля нет — там сцена опознаётся по тайм-кодам. Так делать
    #: нельзя: модель регулярно смещает границы на секунду-другую, и «найти
    #: сцену по времени» превращается в поиск ближайшей с порогом, то есть
    #: в источник тихих ошибок. Индекс мы дали сами, модель его возвращает,
    #: сопоставление становится точным.
    scene_index: int = Field(ge=0)
    start: str = Field(pattern=_TIMECODE_PATTERN)
    end: str = Field(pattern=_TIMECODE_PATTERN)
    score: int = Field(ge=0, le=100)
    category: Category
    #: 300 символов — не эстетика, а деньги: обоснование идёт в выходных
    #: токенах, а их у нас по одному на сцену на всём фильме (док 06, §6.9).
    reason: str = Field(max_length=300)

    @field_validator("reason")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    def to_domain(self, scene_range: TimeRange) -> LLMVerdict:
        """Перевод в доменный вердикт.

        Границы берутся **из нашей сцены**, а не из ответа модели: тайм-коды
        в ответе — подсказка для уточнения на Этапе 7, но не источник правды.
        Доверять им как границам клипа значит доверять модели арифметику,
        которую мы уже посчитали точно.
        """
        return LLMVerdict(
            scene_index=self.scene_index,
            range=scene_range,
            score=float(self.score),
            category=self.category,
            reason=self.reason,
        )

    def timecodes_match(self, scene_range: TimeRange, *, tolerance: float = 2.0) -> bool:
        """Совпадают ли тайм-коды ответа с нашей сценой (± допуск).

        Отдельная проверка, а не отбрасывание расхождений молча: если модель
        назвала сцену 47, а тайм-коды дала от сцены 12, это не «неточность» —
        это сбитая нумерация, и такой вердикт нельзя засчитывать вовсе.
        """
        try:
            start = parse_timecode(self.start)
            end = parse_timecode(self.end)
        except ValueError:
            return False
        return (
            abs(start - scene_range.start) <= tolerance and abs(end - scene_range.end) <= tolerance
        )


class RankingResponse(BaseModel):
    """Ответ на батч сцен."""

    verdicts: list[SceneVerdict] = Field(default_factory=list)


class ClipMetadataResponse(BaseModel):
    """Метаданные готового клипа."""

    title: str = Field(max_length=100)
    description: str = Field(max_length=500)
    #: Минимум три хештега — меньше бессмысленно для площадок; максимум
    #: десять — дальше начинается спам, который те же площадки штрафуют.
    hashtags: list[str] = Field(min_length=3, max_length=10)
    #: Первая фраза для превью или обложки.
    hook: str = Field(max_length=120)

    @field_validator("hashtags")
    @classmethod
    def _normalise(cls, values: list[str]) -> list[str]:
        """Хештеги приводятся к виду ``#слово``.

        Модель возвращает их то с решёткой, то без, то с пробелами внутри.
        Нормализация здесь, а не в шаблоне промпта: просить модель о формате
        можно, полагаться на просьбу — нет.
        """
        normalised: list[str] = []
        for value in values:
            tag = value.strip().lstrip("#").replace(" ", "")
            if tag:
                normalised.append(f"#{tag}")
        return normalised
