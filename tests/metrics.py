"""Метрики качества распознавания.

Живут в тестовом окружении, а не в ``src``: продукту WER не нужен, он нужен
проверке продукта. Своя реализация в двадцать строк вместо зависимости
``jiwer`` — метрика известна, кода в ней меньше, чем в описании пакета.
"""

from __future__ import annotations

import re
from typing import Final

__all__ = ["normalize", "tokens", "wer"]

#: Ё и Е в русском тексте взаимозаменяемы на письме, и разница между
#: «ещё» и «еще» — это орфография, а не ошибка распознавания.
_YO: Final = str.maketrans("ёЁ", "еЕ")
_PUNCTUATION: Final = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES: Final = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Привести текст к виду, в котором сравнение измеряет слух, а не оформление.

    Убираются регистр, пунктуация и различие ё/е. Что **не** убирается:
    цифры. «3 месяца» против «три месяца» — это настоящее расхождение,
    которое всплывёт в субтитрах, и прятать его от метрики нечестно.
    """
    lowered = text.translate(_YO).lower()
    return _SPACES.sub(" ", _PUNCTUATION.sub(" ", lowered)).strip()


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate: расстояние Левенштейна по словам, делённое на эталон.

    0.0 — совпадение слово в слово. 1.0 — не угадано ничего (или угадано
    столько же, сколько добавлено лишнего).
    """
    source, target = tokens(reference), tokens(hypothesis)
    if not source:
        return 0.0 if not target else 1.0

    previous = list(range(len(target) + 1))
    for index, ref_word in enumerate(source, start=1):
        current = [index]
        for position, hyp_word in enumerate(target, start=1):
            current.append(
                min(
                    previous[position] + 1,  # удаление
                    current[position - 1] + 1,  # вставка
                    previous[position - 1] + (ref_word != hyp_word),  # замена
                )
            )
        previous = current
    return previous[-1] / len(source)
