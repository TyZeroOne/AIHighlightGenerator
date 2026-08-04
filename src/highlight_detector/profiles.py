"""Профили скоринга: что считать интересным в этом жанре.

Одни и те же признаки означают разное в разном материале. В боевике экшен
решает всё, в подкасте он вообще не встречается, а диалог, который в фильме
проходной, в интервью и есть содержание. Профиль — это набор весов,
переключающий приоритеты, не трогая код.

## Почему константами, а не YAML

Документ 05 предполагал `config/profiles/*.yaml`. Здесь профили — константы
в модуле, и это осознанная замена.

YAML нужен, когда значения правит **не разработчик**: продакт, редактор,
клиент — и правит на живой системе, без выката. Такого пользователя
у проекта сейчас нет: веса правит тот же человек, что пишет код,
и правка через git ничем не хуже правки файла на сервере, зато проходит
ревью и тесты.

Взамен исчезает целый класс проблем: парсинг, валидация схемы, «профиль
не найден» в проде, YAML-файл, забытый в образе. Появится внешний
редактор весов — переезд на YAML займёт один модуль: интерфейс
(``get_profile``) от формата хранения не зависит.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from core.domain.analysis import ScoringProfile
from core.domain.errors import ValidationError

__all__ = ["PROFILES", "get_profile"]

#: Группы сигналов. Скор считается по группам, а не по отдельным сигналам:
#: так отсутствие целой группы (упал видеоанализ) обрабатывается честно —
#: см. ``scoring.perceived_weights``.
GROUPS: Final[Mapping[str, tuple[str, ...]]] = {
    "emotion": ("emotional_speech", "laughter", "screaming"),
    "action": ("action", "fight", "chase"),
    "narrative": ("plot_twist", "tension"),
    "dialogue": ("long_dialogue",),
    "visual": ("romantic",),
}

_DEFAULT_WEIGHTS: Final[Mapping[str, float]] = {
    "emotion": 0.25,
    "action": 0.25,
    "narrative": 0.20,
    "dialogue": 0.15,
    "visual": 0.15,
}

#: Профили из док 05, §5.5. Длительности клипов различаются не для красоты:
#: у подкаста осмысленный фрагмент длиннее, чем у боевика, — там нужна
#: законченная мысль, здесь достаточно одного удара.
PROFILES: Final[Mapping[str, ScoringProfile]] = {
    "default": ScoringProfile(
        name="default",
        group_weights=_DEFAULT_WEIGHTS,
        min_duration_sec=15.0,
        max_duration_sec=90.0,
        min_score=40.0,
        max_clips=10,
    ),
    "movie": ScoringProfile(
        name="movie",
        group_weights={
            "emotion": 0.20,
            "action": 0.35,
            "narrative": 0.25,
            "dialogue": 0.10,
            "visual": 0.10,
        },
        min_duration_sec=20.0,
        max_duration_sec=90.0,
        min_score=45.0,
        max_clips=10,
    ),
    "podcast": ScoringProfile(
        name="podcast",
        group_weights={
            "emotion": 0.30,
            "action": 0.05,
            "narrative": 0.20,
            "dialogue": 0.35,
            "visual": 0.10,
        },
        min_duration_sec=30.0,
        max_duration_sec=120.0,
        min_score=35.0,
        max_clips=10,
    ),
    "lecture": ScoringProfile(
        name="lecture",
        group_weights={
            "emotion": 0.15,
            "action": 0.05,
            "narrative": 0.30,
            "dialogue": 0.40,
            "visual": 0.10,
        },
        min_duration_sec=30.0,
        max_duration_sec=120.0,
        min_score=35.0,
        max_clips=8,
    ),
    "gaming": ScoringProfile(
        name="gaming",
        group_weights={
            "emotion": 0.30,
            "action": 0.40,
            "narrative": 0.15,
            "dialogue": 0.05,
            "visual": 0.10,
        },
        min_duration_sec=15.0,
        max_duration_sec=60.0,
        min_score=45.0,
        max_clips=12,
    ),
}


def get_profile(name: str) -> ScoringProfile:
    """Профиль по имени.

    Неизвестное имя — ошибка, а не молчаливый откат на ``default``.
    Опечатка в конфиге иначе обернулась бы отбором по чужим весам,
    и заметить это было бы можно только по странному результату.
    """
    profile = PROFILES.get(name)
    if profile is None:
        raise ValidationError(
            f"Неизвестный профиль скоринга: {name!r}",
            profile=name,
            available=sorted(PROFILES),
        )
    return profile
