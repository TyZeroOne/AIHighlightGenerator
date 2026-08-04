"""Чтение кусков аудиодорожки по времени.

Двум анализаторам Этапа 5 — событиям и эмоции — нужен не весь файл,
а короткие окна: десять секунд вокруг интересного места, пять секунд речи
из сцены. Целиком двухчасовую дорожку они разворачивать не должны:
это 460 МБ на каждый вызов, и именно так VAD стал самым прожорливым
местом этапа.

Читаем через :mod:`wave` из стандартной библиотеки. Это возможно потому,
что дорожку делаем мы сами (Этап 4): несжатый PCM 16 бит, 16 кГц, моно.
Для такого формата чтение куска — это позиционирование по номеру отсчёта
и `readframes`, то есть арифметика без декодирования вообще.

> **Почему не `decode_audio` из faster-whisper**
>
> Он умеет любой формат, но только целиком: сначала разворачивает файл
> в память, потом отдаёт массив. Для окна в десять секунд это в тысячу
> раз больше работы, чем нужно, — и вся она в оперативной памяти.

Модуль лежит в ``core.infra``, а не в пакете ``audio``, хотя по смыслу
принадлежит скорее туда. Причина в правиле архитектуры: читать окна нужно
и анализатору звуковых событий (``audio``), и анализатору эмоции
(``speech``), а адаптеры не имеют права импортировать друг друга.
Общий инструмент переезжает в ядро — это дешевле, чем два одинаковых
парсера WAV в двух пакетах.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any, Final

from core.domain.errors import ProcessingError
from core.domain.values import TimeRange
from core.logging import get_logger

__all__ = ["MAX_INT16", "read_window", "track_duration"]

log = get_logger(__name__)

#: Делитель для перевода int16 в вещественный диапазон [-1, 1], в котором
#: работают все модели.
MAX_INT16: Final = 32768.0


def _open(path: Path) -> Any:
    # Контекстный менеджер здесь не подходит: файл нужен вызывающему,
    # а не этой функции — она только открывает и проверяет формат.
    # Закрывают его те, кто вызвал, в своём `finally`.
    try:
        handle = wave.open(str(path), "rb")  # noqa: SIM115
    except Exception as error:
        raise ProcessingError(
            f"Не удалось открыть WAV: {type(error).__name__}: {error}",
            path=path.name,
        ) from error

    if handle.getsampwidth() != 2:
        handle.close()
        raise ProcessingError(
            f"Ожидался PCM 16 бит, получено {handle.getsampwidth() * 8} бит",
            path=path.name,
        )
    return handle


def track_duration(path: Path) -> float:
    """Длительность дорожки в секундах."""
    handle = _open(path)
    try:
        return float(handle.getnframes()) / float(handle.getframerate())
    finally:
        handle.close()


def read_window(path: Path, window: TimeRange) -> Any:
    """Отсчёты дорожки на заданном отрезке — моно, float32 в [-1, 1].

    Многоканальный файл сводится в моно усреднением каналов: модели
    обучались на моно, и подавать им стерео значит либо потерять половину
    сигнала, либо получить непредсказуемое поведение препроцессора.

    Выход за границы файла — не ошибка: сцена может кончаться на последнем
    отсчёте, а окно вокруг неё — заходить за него. Возвращается то, что есть.
    """
    import numpy as np

    handle = _open(path)
    try:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        total = handle.getnframes()

        first = max(int(window.start * rate), 0)
        last = min(int(window.end * rate), total)
        if last <= first:
            return np.zeros(0, dtype=np.float32)

        handle.setpos(first)
        raw = handle.readframes(last - first)
    finally:
        handle.close()

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / MAX_INT16
    if channels > 1:
        # Хвост неполного кадра отбрасываем: reshape иначе не сойдётся,
        # а один отсчёт из ста тысяч ничего не решает.
        usable = (len(samples) // channels) * channels
        samples = samples[:usable].reshape(-1, channels).mean(axis=1)
    return samples
