"""Эвристический скор сцены и слияние с оценкой модели.

Два шага, разделённые намеренно.

**Эвристика** считается всегда: она не зависит ни от сети, ни от денег,
ни от чужой доступности. Это опорная оценка системы.

**Слияние** добавляет мнение модели, если оно есть. Если нет — итоговый
скор равен эвристике, и конвейер работает целиком (док 05, §5.6).

## Перераспределение весов — механизм деградации

```
heuristic = 100 · Σ(w_g · s_g) / Σ(w_g)   по группам g, которые доступны
```

Знаменатель считается **только по доступным группам**, и это не деталь
реализации, а то, ради чего вся конструкция с группами затевалась.

Упал видеоанализ — группы `action` и `visual` недоступны. Без перенормировки
каждая сцена потеряла бы 40 % возможного скора, все клипы просели бы
до уровня «ничего интересного», и порог `min_score` отсёк бы вообще всё.
С перенормировкой оставшиеся группы масштабируются до 100, и отбор
продолжается — по трём группам вместо пяти, но продолжается.

## Бонус за согласие

```
final = α·эвристика + β·LLM + γ·(100 − |эвристика − LLM|)
```

Третье слагаемое вознаграждает совпадение двух независимых оценок. Момент,
который считают удачным и сигналы, и модель, почти всегда действительно
удачный: они смотрят на разное — одна на форму, другая на смысл, — и их
согласие маловероятно случайно.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from core.domain.analysis import ScoringProfile
from core.domain.values import Score
from highlight_detector.profiles import GROUPS

__all__ = ["ALPHA", "BETA", "GAMMA", "fuse", "heuristic_score"]

#: Веса слияния по умолчанию (док 05, §5.6). В сумме единица — иначе
#: итоговый скор вышел бы за 100.
ALPHA: Final = 0.4
BETA: Final = 0.5
GAMMA: Final = 0.1


def heuristic_score(
    signals: Mapping[str, float],
    profile: ScoringProfile,
) -> Score:
    """Скор сцены 0..100 с разложением по группам.

    Группа считается доступной, если в сцене есть хотя бы один её сигнал
    с ненулевым весом. Разложение возвращается всегда: без него вопрос
    «почему эта сцена набрала 90» остаётся без ответа, а значит скоринг
    невозможно отлаживать.
    """
    breakdown: dict[str, float] = {}
    total_weight = 0.0
    total_value = 0.0

    for group, names in GROUPS.items():
        weight = profile.group_weights.get(group, 0.0)
        if weight <= 0:
            continue

        present = [signals[name] for name in names if name in signals]
        if not present:
            # Группа недоступна: её вес не попадает в знаменатель.
            continue

        # Максимум внутри группы, а не среднее: группа «экшен» описывает
        # одно явление тремя способами, и сцена не становится спокойнее
        # оттого, что подходит только под один из них.
        value = max(present)
        breakdown[group] = round(value, 4)
        total_weight += weight
        total_value += weight * value

    if total_weight <= 0:
        # Ни одной группы: признаков нет вовсе. Ноль — честный ответ,
        # а порог min_score отсеет такую сцену.
        return Score(value=0.0, breakdown={})

    return Score(value=round(100.0 * total_value / total_weight, 3), breakdown=breakdown)


def fuse(
    heuristic: float,
    llm_score: float | None,
    *,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
) -> float:
    """Эвристика и оценка модели → итоговый скор 0..100.

    Модель не оценивала сцену (не попала в батч, вердикт отброшен,
    сервис недоступен) — веса β и γ обнуляются, а α перенормируется
    до единицы. Иначе сцена без вердикта получила бы 40 % от своего
    скора и проиграла бы конкуренцию не по существу, а потому что
    её не показали модели.
    """
    if llm_score is None:
        return round(_clamp(heuristic), 3)

    agreement = 100.0 - abs(heuristic - llm_score)
    total = alpha + beta + gamma
    if total <= 0:  # pragma: no cover — защита от конфигурации из нулей
        return round(_clamp(heuristic), 3)

    value = (alpha * heuristic + beta * llm_score + gamma * agreement) / total
    return round(_clamp(value), 3)


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, value))
