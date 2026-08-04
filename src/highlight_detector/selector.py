"""Отбор финальных моментов: от оценённых сцен к списку клипов.

Скор есть у каждой сцены — но список клипов это не «десять сцен с верхушки
рейтинга». Между рейтингом и результатом шесть шагов, и каждый решает
конкретную проблему, которая иначе видна в выдаче невооружённым глазом.

| Шаг | Что было бы без него |
|---|---|
| Расширение | клипы по 3 секунды: сцена редко равна законченному моменту |
| Снап границ | клип начинается с середины слова |
| Фильтр длительности | ролики на 4 секунды и на 6 минут вперемешку |
| NMS | десять клипов, перекрывающих один и тот же кусок |
| MMR | пять почти одинаковых сцен драки подряд |
| Бюджет | час клипов там, где просили десять минут |

Модуль чистый: ни базы, ни сети, ни ввода-вывода. На входе сцены и паузы,
на выходе список ``Highlight``. Поэтому его целиком можно проверить
на выдуманных данных, а каждый шаг — отдельно.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from core.domain.analysis import ScoringProfile
from core.domain.entities import Highlight, Scene
from core.domain.values import TimeRange
from core.logging import get_logger

__all__ = ["Candidate", "expand", "select_highlights", "snap_to_pauses"]

log = get_logger(__name__)

#: Доля от пика, начиная с которой соседняя сцена приклеивается к кандидату.
#: 0.7 — компромисс: ниже склеивалось бы всё подряд в один длинный кусок,
#: выше кандидат остался бы одинокой сценой.
NEIGHBOUR_THRESHOLD: Final = 0.7

#: Сколько сцен максимум может войти в один кандидат. Ограничение появилось
#: после живого прогона: на однородном материале доля от пика не отсекает
#: никого, и кандидат съедал всё видео целиком.
MAX_MEMBERS: Final = 6

#: И не больше этой доли хронометража. Второй потолок нужен потому, что
#: сцены бывают разной длины: шесть длинных сцен — это тоже «всё видео».
MAX_VIDEO_FRACTION: Final = 0.5

#: На сколько секунд разрешено двигать границу ради паузы в речи.
#: Полторы секунды — заметно для монтажа и незаметно для смысла.
SNAP_WINDOW: Final = 1.5

#: Порог пересечения, выше которого кандидат считается дублем (док 05, §5.7).
#:
#: Пробовали ужесточить до 0.25, чтобы убрать два клипа, покрывавших один
#: размеченный момент с разных сторон (их IoU был 0.27 — на грани). Замер
#: показал, что это плохой размен: Precision подросла на три пункта,
#: а Recall упал с 0.83 до 0.67, потому что под нож попал и более точный
#: из двух клипов. Оставлено 0.3.
IOU_THRESHOLD: Final = 0.3

#: Баланс «качество против разнообразия» в MMR. 0.7 — выдача остаётся
#: рейтингом, но однотипные клипы уступают место непохожим.
MMR_LAMBDA: Final = 0.7

#: Штраф за совпадение категории с уже выбранным клипом. Не запрет:
#: если в видео всё смешное, десять смешных клипов — правильный ответ.
CATEGORY_PENALTY: Final = 0.35


@dataclass(frozen=True, slots=True)
class Candidate:
    """Кандидат в клипы: отрезок, собранный из одной или нескольких сцен."""

    range: TimeRange
    score: float
    category: str
    reason: str
    scene_indices: tuple[int, ...]
    signals: Mapping[str, float]


def expand(
    scenes: Sequence[Scene],
    *,
    min_duration_sec: float,
    max_duration_sec: float,
    threshold: float = NEIGHBOUR_THRESHOLD,
) -> list[Candidate]:
    """Сцены → кандидаты, склеенные с достойными соседями.

    Идём от каждой сцены как от пика и прирастаем соседями. Растём
    поочерёдно в обе стороны, выбирая на каждом шаге более сильного соседа:
    иначе кандидат уезжал бы в одну сторону просто потому, что её
    просматривали первой.

    Рост идёт в две фазы, и обе появились после живого прогона.

    **Фаза «по достоинству»** — сосед принимается, пока его скор держится
    выше доли от пика. Одного этого правила мало: на однородном материале
    (говорящая голова, все сцены похожи) любой сосед проходит порог,
    и кандидат растёт до упора, покрывая половину видео. Дальше NMS
    схлопывает такие кандидаты в один — вместо отбора получается
    «взять всё подряд». Поэтому рост ограничен ещё и сверху: не более
    ``MAX_MEMBERS`` сцен и не более половины длительности видео.

    **Фаза «до минимума»** — если кандидат короче ``min_duration_sec``,
    он дорастает соседями **вопреки порогу**, пока не дотянет. Без этой
    фазы расширение и фильтр длительности работали друг против друга:
    сильная короткая сцена в конце ролика набирала 13.6 с при минимуме 15,
    и лучший момент видео отбрасывался фильтром, не доехав до NMS.
    """
    if not scenes:
        return []

    ordered = sorted(scenes, key=lambda scene: scene.index)
    by_position = {scene.index: position for position, scene in enumerate(ordered)}
    video_span = ordered[-1].range.end - ordered[0].range.start
    # Кандидат не может занимать больше половины видео: то, что «интересно»
    # на половине хронометража, интересным уже не является.
    span_limit = (
        min(max_duration_sec, video_span * MAX_VIDEO_FRACTION)
        if video_span > 0
        else (max_duration_sec)
    )
    candidates: list[Candidate] = []

    for scene in ordered:
        peak = scene.final_score if scene.final_score is not None else 0.0
        if peak <= 0:
            continue

        position = by_position[scene.index]
        left = right = position
        floor = peak * threshold

        # Фаза 1: рост по достоинству, с потолками.
        while right - left + 1 < MAX_MEMBERS:
            duration = ordered[right].range.end - ordered[left].range.start
            if duration >= span_limit:
                break

            left_score = _score_at(ordered, left - 1)
            right_score = _score_at(ordered, right + 1)
            if max(left_score, right_score) < floor:
                break

            step = _grow(
                ordered, left, right, left_score, right_score, limit=span_limit, peak=position
            )
            if step is None:
                break
            left, right = step

        # Фаза 2: дотянуть до минимальной длительности любой ценой.
        # Потолок здесь — max_duration_sec, а не span_limit: если минимум
        # профиля больше половины видео, выбор между «клип длиннее половины»
        # и «клипов нет вовсе» решается в пользу клипа.
        while ordered[right].range.end - ordered[left].range.start < min_duration_sec:
            step = _grow(
                ordered,
                left,
                right,
                _score_at(ordered, left - 1),
                _score_at(ordered, right + 1),
                limit=max_duration_sec,
                peak=position,
            )
            if step is None:
                break
            left, right = step

        members = ordered[left : right + 1]
        candidates.append(
            Candidate(
                range=TimeRange(members[0].range.start, members[-1].range.end),
                score=peak,
                category=scene.category or "other",
                reason=scene.reason or "",
                scene_indices=tuple(member.index for member in members),
                signals=_signals_of(scene),
            )
        )

    return candidates


def snap_to_pauses(
    candidate: Candidate,
    pauses: Sequence[TimeRange],
    *,
    window: float = SNAP_WINDOW,
) -> Candidate:
    """Сдвинуть границы к ближайшей паузе в речи.

    Клип, начинающийся с середины слова, выглядит браком независимо от того,
    насколько удачен момент внутри. Ищем паузу в пределах окна: начало
    двигаем к её концу (чтобы не захватить хвост чужой фразы), конец —
    к её началу.

    Пауз нет — границы остаются на сцене. Это уже приемлемо: границы сцен
    проходят по монтажным склейкам.
    """
    if not pauses:
        return candidate

    start = _nearest_edge(
        candidate.range.start,
        [pause.end for pause in pauses],
        window=window,
    )
    end = _nearest_edge(
        candidate.range.end,
        [pause.start for pause in pauses],
        window=window,
    )

    if end - start <= 0:  # pragma: no cover — вырожденный случай
        return candidate
    return Candidate(
        range=TimeRange(start, end),
        score=candidate.score,
        category=candidate.category,
        reason=candidate.reason,
        scene_indices=candidate.scene_indices,
        signals=candidate.signals,
    )


def select_highlights(
    scenes: Sequence[Scene],
    profile: ScoringProfile,
    *,
    pauses: Sequence[TimeRange] = (),
    total_duration_sec: float | None = None,
) -> list[Highlight]:
    """Полный конвейер отбора: расширение → снап → фильтр → NMS → MMR."""
    candidates = expand(
        scenes,
        min_duration_sec=profile.min_duration_sec,
        max_duration_sec=profile.max_duration_sec,
    )
    if not candidates:
        return []

    snapped = [snap_to_pauses(candidate, pauses) for candidate in candidates]

    fitting = [
        candidate
        for candidate in snapped
        if profile.min_duration_sec <= candidate.range.duration <= profile.max_duration_sec
        and candidate.score >= profile.min_score
    ]
    log.debug(
        "select.filtered",
        candidates=len(snapped),
        fitting=len(fitting),
        min_score=profile.min_score,
    )
    if not fitting:
        return []

    deduped = _suppress_overlaps(fitting)
    diverse = _diversify(deduped, limit=profile.max_clips)

    if total_duration_sec is not None:
        diverse = _fit_budget(diverse, total_duration_sec)

    by_index = {scene.index: scene for scene in scenes}
    return [
        Highlight(
            range=candidate.range,
            final_score=candidate.score,
            category=candidate.category,
            reason=candidate.reason,
            scene_ids=tuple(
                by_index[index].id for index in candidate.scene_indices if index in by_index
            ),
            breakdown=dict(candidate.signals),
        )
        for candidate in diverse
    ]


# ------------------------------------------------------------------ шаги
def _suppress_overlaps(
    candidates: Sequence[Candidate],
    *,
    threshold: float = IOU_THRESHOLD,
) -> list[Candidate]:
    """Non-maximum suppression по пересечению отрезков.

    Кандидаты рождаются от каждой сцены, поэтому соседние почти всегда
    описывают один и тот же кусок видео. Оставляем сильнейшего из группы.
    """
    ordered = sorted(candidates, key=lambda item: (-item.score, item.range.start))
    kept: list[Candidate] = []

    for candidate in ordered:
        if any(_iou(candidate.range, other.range) > threshold for other in kept):
            continue
        kept.append(candidate)

    return kept


def _diversify(
    candidates: Sequence[Candidate],
    *,
    limit: int,
    lambda_: float = MMR_LAMBDA,
) -> list[Candidate]:
    """Maximal Marginal Relevance: качество минус похожесть на выбранное.

    Без этого шага выдача сползает в одну ноту: если в видео есть пять
    сцен драки, они займут весь топ, и зритель увидит один и тот же приём
    пять раз. MMR на каждом шаге берёт кандидата, максимизирующего
    ``λ·score − (1−λ)·похожесть на уже выбранное``.
    """
    if not candidates:
        return []

    pool = sorted(candidates, key=lambda item: -item.score)
    selected: list[Candidate] = [pool[0]]
    rest = pool[1:]

    while rest and len(selected) < limit:
        best_index = 0
        best_value = -math.inf
        for index, candidate in enumerate(rest):
            similarity = max(_similarity(candidate, chosen) for chosen in selected)
            value = lambda_ * (candidate.score / 100.0) - (1 - lambda_) * similarity
            if value > best_value:
                best_value = value
                best_index = index
        selected.append(rest.pop(best_index))

    return selected


def _fit_budget(candidates: Sequence[Candidate], budget: float) -> list[Candidate]:
    """Жадный набор, пока укладываемся в суммарную длительность."""
    total = 0.0
    kept: list[Candidate] = []
    for candidate in candidates:
        if total + candidate.range.duration > budget:
            continue
        kept.append(candidate)
        total += candidate.range.duration
    return kept


# ------------------------------------------------------------- вспомогательное
def _grow(
    scenes: Sequence[Scene],
    left: int,
    right: int,
    left_score: float,
    right_score: float,
    *,
    limit: float,
    peak: int,
) -> tuple[int, int] | None:
    """Один шаг расширения в сторону более сильного соседа.

    Возвращает ``None``, если расти некуда: соседей нет или любой из них
    выводит кандидата за ``limit``. Общий код двух фаз — иначе они бы
    разошлись в поведении на границах, а именно там и живут ошибки.
    """
    options: list[tuple[float, float, int, int]] = []
    if right + 1 < len(scenes):
        span = scenes[right + 1].range.end - scenes[left].range.start
        if span <= limit:
            # Второй ключ — «насколько эта сторона отстаёт»: чем меньше
            # уже прирос правый край, тем охотнее растём вправо.
            options.append((right_score, -(right - peak), left, right + 1))
    if left - 1 >= 0:
        span = scenes[right].range.end - scenes[left - 1].range.start
        if span <= limit:
            options.append((left_score, -(peak - left), left - 1, right))

    if not options:
        return None

    # Сортировка по паре: сначала сильнейший сосед, при равенстве —
    # та сторона, которая ближе к пику.
    #
    # Замер на трейлере показал, зачем нужен второй ключ: при простом
    # «предпочитаем вправо» три клипа из четырёх начинались на 5–13 секунд
    # позже размеченного экспертом момента. Пик оказывался в начале клипа,
    # а не в середине: эксперт берёт завязку, система брала развязку.
    best = max(options, key=lambda option: (option[0], option[1]))
    return best[2], best[3]


def _score_at(scenes: Sequence[Scene], position: int) -> float:
    if position < 0 or position >= len(scenes):
        return -math.inf
    score = scenes[position].final_score
    return score if score is not None else -math.inf


def _signals_of(scene: Scene) -> dict[str, float]:
    return {
        key.removeprefix("signal."): value.value
        for key, value in scene.features.items()
        if key.startswith("signal.")
    }


def _nearest_edge(value: float, edges: Sequence[float], *, window: float) -> float:
    """Ближайшая граница паузы в пределах окна — или исходное значение."""
    best = value
    best_distance = window
    for edge in edges:
        distance = abs(edge - value)
        if distance < best_distance:
            best_distance = distance
            best = edge
    return best


def _iou(left: TimeRange, right: TimeRange) -> float:
    """Пересечение к объединению для двух отрезков времени."""
    overlap = min(left.end, right.end) - max(left.start, right.start)
    if overlap <= 0:
        return 0.0
    union = max(left.end, right.end) - min(left.start, right.start)
    return overlap / union if union > 0 else 0.0


def _similarity(left: Candidate, right: Candidate) -> float:
    """Похожесть кандидатов: косинус по сигналам плюс штраф за категорию.

    Категория — самая заметная для зрителя ось однообразия: пять клипов
    с меткой ``fight`` выглядят одинаково, даже если их сигналы различаются.
    Поэтому она учитывается отдельно от косинуса, а не растворяется в нём.
    """
    keys = set(left.signals) | set(right.signals)
    if keys:
        a = [left.signals.get(key, 0.0) for key in sorted(keys)]
        b = [right.signals.get(key, 0.0) for key in sorted(keys)]
        norm_a = math.sqrt(sum(value * value for value in a))
        norm_b = math.sqrt(sum(value * value for value in b))
        cosine = (
            sum(x * y for x, y in zip(a, b, strict=True)) / (norm_a * norm_b)
            if norm_a > 0 and norm_b > 0
            else 0.0
        )
    else:
        cosine = 0.0

    penalty = CATEGORY_PENALTY if left.category == right.category else 0.0
    return min(1.0, cosine + penalty)
