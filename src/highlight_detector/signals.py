"""Составные сигналы: от признаков к понятиям заказчика.

В ТЗ перечислены не признаки, а явления: «драка», «погоня», «смех»,
«эмоциональная речь», «неожиданный поворот». Ни одно из них не измеряется
напрямую — каждое собирается из нескольких нормализованных признаков
(док 05, §5.2).

## Зачем этот слой вообще

Можно было бы считать итоговый скор прямо из двадцати признаков одной
взвешенной суммой. Три причины так не делать.

**Отладка.** «Сцена набрала 90» ничего не говорит, а «action 0.85,
laughter 0.02, tension 0.40» — говорит. Разложение по сигналам попадает
в ``score_breakdown`` и отвечает на вопрос «почему».

**Настройка без релиза.** Веса внутри сигналов и веса групп живут
в YAML-профиле. Правка «в подкастах диалог важнее экшена» — это правка
файла, а не кода.

**Промпт короче.** В модель уходят сигналы, а не сорок сырых чисел
(док 06, §6.4): и токенов меньше, и понятнее.

## Почему формулы линейные

Соблазн обучить регрессию понятен, и для него нет данных: размеченного
корпуса «интересных моментов» у проекта нет, а собрать его — отдельный
проект. Линейные комбинации с весами в конфиге — честный компромисс:
их можно объяснить, отладить и поправить руками, а когда появится
golden-датасет (док 05, §5.8), они станут baseline для сравнения.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

__all__ = ["SIGNAL_NAMES", "compute_signals"]

#: Все сигналы, которые умеет считать этот модуль. Список закрытый:
#: по именам сигналов работают веса групп в профиле, и опечатка в имени
#: означала бы молча потерянный вклад.
SIGNAL_NAMES: Final = (
    "emotional_speech",
    "laughter",
    "screaming",
    "fight",
    "chase",
    "action",
    "plot_twist",
    "long_dialogue",
    "tension",
    "romantic",
)


def _get(features: Mapping[str, float], key: str, default: float = 0.0) -> float:
    """Признак или значение по умолчанию.

    Отсутствие признака — штатная ситуация: анализатор мог быть отключён
    через ``ANALYZERS_DISABLED`` или упасть. Ноль здесь означает «этот
    вклад не учитываем», а не «признак равен нулю»; на уровне групп
    отсутствие целой группы обрабатывается отдельно (см. ``scoring``).
    """
    value = features.get(key, default)
    return value if isinstance(value, (int, float)) else default


def compute_signals(
    features: Mapping[str, float],
    weights: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, float]:
    """Нормализованные признаки одной сцены → сигналы 0..1.

    ``features`` — уже нормализованные значения (см. :mod:`.normalize`),
    иначе слагаемые несопоставимы. ``weights`` позволяет профилю переопределить
    коэффициенты внутри сигнала; отсутствующие берутся из формул ниже.
    """
    custom = weights or {}

    signals: dict[str, float] = {}

    # Эмоциональная речь: возбуждение важнее знака эмоции — злость и восторг
    # одинаково годятся в клип, а спокойная радость нет.
    signals["emotional_speech"] = _combine(
        custom.get("emotional_speech"),
        {
            "speech.emotion.arousal": 0.45,
            "_abs_valence": 0.25,
            "audio.loudness.delta": 0.20,
            "speech.rate.delta": 0.10,
        },
        {**features, "_abs_valence": abs(_get(features, "speech.emotion.valence") - 0.5) * 2},
    )

    signals["laughter"] = _combine(
        custom.get("laughter"),
        {"audio.event.laughter": 0.80, "speech.emotion.hap": 0.20},
        features,
    )

    signals["screaming"] = _combine(
        custom.get("screaming"),
        {
            "audio.event.scream": 0.55,
            "audio.loudness.peak": 0.30,
            "speech.emotion.arousal": 0.15,
        },
        features,
    )

    signals["fight"] = _combine(
        custom.get("fight"),
        {
            "vision.motion.peak": 0.35,
            "vision.cuts.rate": 0.25,
            "audio.event.impact": 0.20,
            "vision.camera.shake": 0.20,
        },
        features,
    )

    signals["chase"] = _combine(
        custom.get("chase"),
        {
            "vision.camera.pan": 0.40,
            "vision.motion.mean": 0.25,
            "vision.cuts.rate": 0.20,
            "vision.camera.shake": 0.15,
        },
        features,
    )

    # Экшен — максимум из драки и погони, а не их сумма: сцена не становится
    # вдвое динамичнее оттого, что подходит под оба описания сразу.
    signals["action"] = min(
        1.0,
        max(signals["fight"], signals["chase"])
        + 0.2 * _get(features, "audio.loudness.dynamic_range"),
    )

    signals["plot_twist"] = _combine(
        custom.get("plot_twist"),
        {
            "struct.novelty": 0.40,
            "audio.pause.max": 0.25,
            "speech.emotion.arousal": 0.20,
            "vision.dynamics.brightness_delta": 0.15,
        },
        features,
    )

    signals["long_dialogue"] = _combine(
        custom.get("long_dialogue"),
        {
            "speech.density": 0.50,
            "speech.dialogue_turns": 0.30,
            "vision.faces.count_mean": 0.20,
        },
        features,
    )

    # Напряжение: тихо, но напряжённо. Отдельно считается «низкая громкость
    # при высоком возбуждении» — само по себе ни то, ни другое не сигнал.
    quiet_tension = _get(features, "speech.emotion.arousal") * (
        1.0 - _get(features, "audio.loudness.mean")
    )
    signals["tension"] = _combine(
        custom.get("tension"),
        {
            "struct.build_up": 0.35,
            "audio.pause.ratio": 0.25,
            "_quiet_tension": 0.20,
            "audio.event.music": 0.20,
        },
        {**features, "_quiet_tension": quiet_tension},
    )

    # Романтика: крупный план, спокойная позитивная эмоция, музыка, двое в кадре.
    two_faces = 1.0 if abs(_get(features, "vision.faces.count_max") - 0.4) < 0.25 else 0.0
    warm = _get(features, "speech.emotion.hap") * (1.0 - _get(features, "speech.emotion.arousal"))
    signals["romantic"] = _combine(
        custom.get("romantic"),
        {
            "vision.faces.closeup": 0.35,
            "_warm": 0.25,
            "audio.event.music": 0.20,
            "_two_faces": 0.20,
        },
        {**features, "_warm": warm, "_two_faces": two_faces},
    )

    return {name: round(min(max(value, 0.0), 1.0), 4) for name, value in signals.items()}


def _combine(
    override: Mapping[str, float] | None,
    default_weights: Mapping[str, float],
    features: Mapping[str, float],
) -> float:
    """Взвешенная сумма с перенормировкой по доступным слагаемым.

    Ключевая деталь — знаменатель. Если половина признаков отсутствует
    (видеогруппа отключена), сумма делится на вес **имеющихся**, а не на
    единицу. Иначе сигнал «драка» на видео без видеоанализа всегда был бы
    вдвое ниже, и такая сцена молча проигрывала бы конкуренцию — не потому
    что она спокойнее, а потому что мы её не досмотрели.
    """
    weights = dict(override) if override else dict(default_weights)

    total_weight = 0.0
    total_value = 0.0
    for key, weight in weights.items():
        if key not in features:
            continue
        total_weight += weight
        total_value += weight * _get(features, key)

    if total_weight <= 0:
        return 0.0
    return total_value / total_weight
