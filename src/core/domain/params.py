"""Параметры анализа и их отпечаток.

## Зачем отдельный тип, если есть словарь

``Job.params`` — это ``jsonb``, и хранить туда можно что угодно. Но на этих
параметрах держится **идемпотентность**: повторный запрос на то же видео
с теми же настройками обязан вернуть уже посчитанный джоб, а не запустить
сорок минут работы заново.

Сравнивать словари для этого нельзя. ``{"profile": "movie", "max_clips": 10}``
и ``{"max_clips": 10, "profile": "movie"}`` — одно и то же по смыслу и разные
строки по факту; отсутствующее поле и поле со значением по умолчанию — тоже.
Отпечаток должен зависеть от **смысла**, а не от того, в каком порядке
клиент собрал JSON и сколько полей не поленился передать.

Отсюда конструкция: типизированный объект со значениями по умолчанию →
канонический словарь (все поля, сортированные ключи) → sha256.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

__all__ = ["AnalysisParams"]

#: Верхняя граница на число клипов в одном джобе. Не вкус: каждый клип —
#: это рендер, и запрос на тысячу клипов означает часы работы воркера
#: по одной строчке в JSON.
MAX_CLIPS_LIMIT: Final = 50


@dataclass(frozen=True, slots=True)
class AnalysisParams:
    """Что именно просили сделать с видео."""

    profile: str = "default"
    max_clips: int = 10
    min_duration_sec: float = 15.0
    max_duration_sec: float = 90.0
    #: Собирать ли вертикальную версию 1080×1920.
    vertical: bool = True
    burn_subtitles: bool = False
    #: Язык распознавания; ``None`` — определить по речи.
    language: str | None = None
    use_llm: bool = True
    analyzers_disabled: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not 1 <= self.max_clips <= MAX_CLIPS_LIMIT:
            raise ValueError(f"max_clips должен быть в 1..{MAX_CLIPS_LIMIT}: {self.max_clips}")
        if self.min_duration_sec <= 0:
            raise ValueError(f"min_duration_sec должен быть положительным: {self.min_duration_sec}")
        if self.max_duration_sec <= self.min_duration_sec:
            raise ValueError(
                "max_duration_sec должен быть больше min_duration_sec: "
                f"{self.max_duration_sec} <= {self.min_duration_sec}"
            )

    def as_dict(self) -> dict[str, Any]:
        """Канонический вид: все поля, ничего не пропущено.

        Именно все — включая те, что остались по умолчанию. Иначе два
        запроса, отличающиеся только явно переданным дефолтом, дали бы
        разные отпечатки и посчитались бы дважды.
        """
        return {
            "profile": self.profile,
            "max_clips": self.max_clips,
            "min_duration_sec": round(self.min_duration_sec, 3),
            "max_duration_sec": round(self.max_duration_sec, 3),
            "vertical": self.vertical,
            "burn_subtitles": self.burn_subtitles,
            "language": self.language,
            "use_llm": self.use_llm,
            # Порядок отключённых анализаторов ничего не значит, а на строку
            # влияет — сортируем, чтобы отпечаток от него не зависел.
            "analyzers_disabled": sorted(self.analyzers_disabled),
        }

    def fingerprint(self) -> str:
        """sha256 канонического представления.

        ``sort_keys`` и ``ensure_ascii=False`` — не косметика: без первого
        отпечаток зависит от порядка вставки в словарь, без второго —
        от того, попала ли в параметры кириллица.
        """
        payload = json.dumps(self.as_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> AnalysisParams:
        """Восстановить параметры из того, что лежит в базе.

        Незнакомые ключи молча игнорируются: джоб мог быть создан старой
        версией, и падать на чтении истории — худшее, что можно сделать
        с записью, которая уже отработала.
        """
        known = {name: raw[name] for name in _FIELD_NAMES if name in raw}
        disabled = known.get("analyzers_disabled")
        if isinstance(disabled, list | tuple):
            known["analyzers_disabled"] = tuple(str(item) for item in disabled)
        return cls(**known)


_FIELD_NAMES: Final = frozenset(AnalysisParams.__dataclass_fields__)
