"""Превращение точек склеек в сцены.

Детектор — что PySceneDetect, что ``scdet`` — отвечает на один вопрос:
«в каких местах картинка резко сменилась». Ответ сырой, и напрямую его
использовать нельзя:

* три склейки подряд за полсекунды (вспышка, удар, быстрый монтаж) дают
  сцены по 0.2 с — анализировать в них нечего, а строк в БД прибавляют тысячи;
* статичный план на 12 минут даёт одну сцену, внутри которой скоринг усреднит
  всё подряд и утопит любой всплеск;
* границы окна детектор не возвращает вовсе — он говорит только про склейки.

Поэтому здесь живёт постобработка, общая для всех реализаций порта.
Функции чистые: на входе числа, на выходе числа, никакого ввода-вывода.
Это позволяет проверять контракт «сцены покрывают окно целиком, без дыр
и пересечений» property-тестами на тысячах случайных наборов склеек.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence

from core.domain.values import EPSILON, TimeRange

__all__ = ["merge_short", "normalize_scenes", "scenes_from_cuts", "split_long"]


def normalize_scenes(
    cuts: Iterable[float],
    window: TimeRange,
    *,
    min_len_sec: float,
    max_len_sec: float,
) -> list[TimeRange]:
    """Точки склеек → сцены, покрывающие окно целиком.

    Порядок шагов важен. Сначала слияние коротких, потом дробление длинных:
    в обратном порядке слияние снова склеило бы то, что дробление только что
    разделило, и результат зависел бы от того, чему повезло оказаться раньше.

    Гарантии на выходе (проверяются тестами):

    * первая сцена начинается в ``window.start``, последняя кончается
      в ``window.end``, соседние стыкуются вплотную;
    * все сцены не короче ``min_len_sec`` — кроме единственного случая, когда
      само окно короче, и делить нечего;
    * все сцены не длиннее ``max_len_sec`` — кроме случая, когда дробление
      нарушило бы предыдущий пункт.
    """
    scenes = scenes_from_cuts(cuts, window)
    scenes = merge_short(scenes, min_len_sec)
    return split_long(scenes, max_len_sec, min_len_sec=min_len_sec)


def scenes_from_cuts(cuts: Iterable[float], window: TimeRange) -> list[TimeRange]:
    """Построить непрерывное покрытие окна по точкам склеек.

    Склейки вне окна и на самых его границах отбрасываются: первая создала бы
    сцену за пределами обрабатываемого отрезка, вторая — сцену нулевой длины.
    Совпадающие точки схлопываются — два детектора внутри PySceneDetect
    вполне могут указать на один и тот же кадр.
    """
    inner = sorted(
        {
            cut
            for cut in cuts
            if math.isfinite(cut) and window.start + EPSILON < cut < window.end - EPSILON
        }
    )

    boundaries = [window.start, *inner, window.end]
    return [
        TimeRange(start, end)
        for start, end in itertools.pairwise(boundaries)
        if end - start > EPSILON
    ]


def merge_short(scenes: Sequence[TimeRange], min_len_sec: float) -> list[TimeRange]:
    """Слить сцены короче ``min_len_sec`` со следующей.

    Направление выбрано осознанно. Слияние вперёд сохраняет **начало** сцены —
    а начало это и есть реальная склейка, замеченная детектором; конец же
    у сцены всегда искусственный, это просто «место, где началась следующая».
    Заодно естественно обрабатывается вспышка в самом начале окна: сливать
    её назад было бы некуда.

    Альтернатива «сливать с более коротким соседом» отвергнута: у быстрого
    монтажа короткие куски идут пачками, и результат такого слияния зависит
    от порядка обхода — то есть невоспроизводим.

    Последняя сцена — исключение: ей сливаться вперёд некуда, поэтому она
    уходит в предыдущую. Единственная сцена не удаляется никогда: если всё
    окно короче минимума, возвращается оно целиком. Пустой список — это
    отсутствие результата этапа, и получить его из непустого входа нельзя.
    """
    merged: list[TimeRange] = []
    for scene in scenes:
        if merged and merged[-1].duration < min_len_sec - EPSILON:
            merged[-1] = TimeRange(merged[-1].start, scene.end)
        else:
            merged.append(scene)

    # Хвост: последняя сцена могла остаться короткой — сливать вперёд некуда.
    while len(merged) > 1 and merged[-1].duration < min_len_sec - EPSILON:
        tail = merged.pop()
        merged[-1] = TimeRange(merged[-1].start, tail.end)

    return merged


def split_long(
    scenes: Sequence[TimeRange],
    max_len_sec: float,
    *,
    min_len_sec: float = 0.0,
) -> list[TimeRange]:
    """Раздробить сцены длиннее ``max_len_sec`` на равные части.

    На равные, а не «по ``max_len_sec`` плюс остаток»: интервью на 185 секунд
    при пределе 90 дало бы 90 + 90 + 5, и этот пятисекундный огрызок пришлось
    бы всё равно сливать обратно. Три части по 61.7 с полезнее.

    Число частей ограничено так, чтобы ни одна не оказалась короче
    ``min_len_sec``: инвариант «сцена не короче минимума» важнее, чем
    «сцена не длиннее максимума». Первый защищает от мусорных строк в БД,
    второй — лишь от слишком крупного шага анализа.
    """
    result: list[TimeRange] = []
    for scene in scenes:
        parts = _part_count(scene.duration, max_len_sec, min_len_sec)
        if parts == 1:
            result.append(scene)
            continue

        step = scene.duration / parts
        for index in range(parts):
            start = scene.start + step * index
            # Последнюю границу берём из исходной сцены, а не считаем:
            # накопленная ошибка сложения дала бы дыру в покрытии.
            end = scene.end if index == parts - 1 else scene.start + step * (index + 1)
            result.append(TimeRange(start, end))

    return result


def _part_count(duration: float, max_len_sec: float, min_len_sec: float) -> int:
    if max_len_sec <= 0 or duration <= max_len_sec + EPSILON:
        return 1

    parts = math.ceil(duration / max_len_sec)
    if min_len_sec > 0:
        parts = min(parts, math.floor(duration / min_len_sec))
    return max(parts, 1)
