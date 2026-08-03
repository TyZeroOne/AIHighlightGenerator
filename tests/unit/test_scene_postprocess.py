"""Постобработка сцен: слияние, дробление, покрытие окна.

Здесь проверяется контракт, ради которого постобработка вообще существует:
что бы ни вернул детектор — хоть пустой список, хоть склейку на каждом кадре —
на выходе получаются сцены, покрывающие окно целиком, без дыр и пересечений.

Часть проверок — property-based: контракт формулируется для *любого* набора
склеек, а не для трёх примеров, которые придумал автор. Hypothesis подбирает
входы сам и, найдя контрпример, ужимает его до минимального.
"""

from __future__ import annotations

import itertools

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from core.domain.values import EPSILON, TimeRange
from video_loader.scenes.postprocess import (
    merge_short,
    normalize_scenes,
    scenes_from_cuts,
    split_long,
)

pytestmark = pytest.mark.unit


def assert_covers(scenes: list[TimeRange], window: TimeRange) -> None:
    """Сцены покрывают окно целиком, без дыр и пересечений."""
    assert scenes, "покрытие не может быть пустым"
    assert scenes[0].start == pytest.approx(window.start, abs=1e-6)
    assert scenes[-1].end == pytest.approx(window.end, abs=1e-6)
    for previous, current in itertools.pairwise(scenes):
        assert current.start == pytest.approx(previous.end, abs=1e-6)


class TestScenesFromCuts:
    def test_cuts_become_boundaries(self) -> None:
        window = TimeRange(0.0, 30.0)

        scenes = scenes_from_cuts([10.0, 20.0], window)

        assert scenes == [TimeRange(0.0, 10.0), TimeRange(10.0, 20.0), TimeRange(20.0, 30.0)]

    def test_no_cuts_gives_one_scene(self) -> None:
        """Статичное видео — это одна сцена, а не ноль.

        Ноль сцен означал бы «этап не дал результата», и весь конвейер дальше
        остался бы без входных данных.
        """
        assert scenes_from_cuts([], TimeRange(0.0, 30.0)) == [TimeRange(0.0, 30.0)]

    def test_cuts_outside_window_are_ignored(self) -> None:
        """Склейка за пределами окна создала бы сцену вне обрабатываемого отрезка."""
        scenes = scenes_from_cuts([-5.0, 12.0, 99.0], TimeRange(10.0, 20.0))

        assert scenes == [TimeRange(10.0, 12.0), TimeRange(12.0, 20.0)]

    def test_duplicate_and_boundary_cuts_do_not_create_empty_scenes(self) -> None:
        """Два детектора вполне могут указать на один и тот же кадр."""
        scenes = scenes_from_cuts([15.0, 15.0, 0.0, 30.0], TimeRange(0.0, 30.0))

        assert scenes == [TimeRange(0.0, 15.0), TimeRange(15.0, 30.0)]

    def test_unsorted_cuts(self) -> None:
        scenes = scenes_from_cuts([20.0, 5.0, 12.0], TimeRange(0.0, 30.0))

        assert [scene.start for scene in scenes] == [0.0, 5.0, 12.0, 20.0]


class TestMergeShort:
    def test_short_scene_is_merged_into_the_next_one(self) -> None:
        """Короткая сцена уходит вперёд, и граница на её начале сохраняется:
        это настоящая склейка, а конец у сцены всегда искусственный."""
        scenes = [TimeRange(0.0, 10.0), TimeRange(10.0, 10.4), TimeRange(10.4, 20.0)]

        merged = merge_short(scenes, 1.5)

        assert merged == [TimeRange(0.0, 10.0), TimeRange(10.0, 20.0)]

    def test_short_scene_at_the_very_start(self) -> None:
        """Вспышка на первом кадре: сливать назад некуда, и это работает."""
        scenes = [TimeRange(0.0, 0.4), TimeRange(0.4, 10.0)]

        assert merge_short(scenes, 1.5) == [TimeRange(0.0, 10.0)]

    def test_burst_of_short_scenes_collapses(self) -> None:
        """Быстрый монтаж: пять склеек за полторы секунды."""
        scenes = [
            TimeRange(0.0, 0.3),
            TimeRange(0.3, 0.6),
            TimeRange(0.6, 0.9),
            TimeRange(0.9, 1.2),
            TimeRange(1.2, 10.0),
        ]

        merged = merge_short(scenes, 1.5)

        assert merged == [TimeRange(0.0, 10.0)]

    def test_short_tail_merges_backwards(self) -> None:
        """Последней сцене сливаться вперёд некуда — сливаем назад."""
        scenes = [TimeRange(0.0, 10.0), TimeRange(10.0, 10.2)]

        assert merge_short(scenes, 1.5) == [TimeRange(0.0, 10.2)]

    def test_single_short_scene_survives(self) -> None:
        """Окно короче минимума. Выкинуть его нельзя: другого результата нет."""
        assert merge_short([TimeRange(0.0, 0.4)], 1.5) == [TimeRange(0.0, 0.4)]


class TestSplitLong:
    def test_long_scene_is_split_into_equal_parts(self) -> None:
        parts = split_long([TimeRange(0.0, 185.0)], 90.0)

        assert len(parts) == 3
        assert [round(part.duration, 3) for part in parts] == [61.667, 61.667, 61.667]
        assert parts[-1].end == 185.0

    def test_short_scene_is_untouched(self) -> None:
        assert split_long([TimeRange(0.0, 30.0)], 90.0) == [TimeRange(0.0, 30.0)]

    def test_split_never_produces_scenes_shorter_than_minimum(self) -> None:
        """Инвариант «не короче минимума» важнее, чем «не длиннее максимума».

        При max=2 и min=2 честное дробление 5-секундной сцены дало бы три
        части по 1.67 — короче минимума. Берём две по 2.5.
        """
        parts = split_long([TimeRange(0.0, 5.0)], 2.0, min_len_sec=2.0)

        assert len(parts) == 2
        assert all(part.duration >= 2.0 - EPSILON for part in parts)


class TestNormalize:
    def test_pipeline_order_short_then_long(self) -> None:
        window = TimeRange(0.0, 300.0)
        # Пачка коротких в начале + огромный статичный кусок до конца.
        cuts = [0.2, 0.5, 0.9, 1.1]

        scenes = normalize_scenes(cuts, window, min_len_sec=1.5, max_len_sec=90.0)

        assert_covers(scenes, window)
        assert all(scene.duration <= 90.0 + EPSILON for scene in scenes)
        assert all(scene.duration >= 1.5 - EPSILON for scene in scenes)

    def test_window_not_starting_at_zero(self) -> None:
        """Окно стрима начинается не с нуля — покрытие всё равно полное."""
        window = TimeRange(120.0, 180.0)

        scenes = normalize_scenes([130.0, 150.0], window, min_len_sec=1.5, max_len_sec=90.0)

        assert_covers(scenes, window)

    def test_window_shorter_than_minimum(self) -> None:
        window = TimeRange(0.0, 0.8)

        scenes = normalize_scenes([0.4], window, min_len_sec=1.5, max_len_sec=90.0)

        assert scenes == [window]


class TestNormalizeProperties:
    """Свойства, которые обязаны выполняться для любого входа."""

    @settings(max_examples=300)
    @given(
        duration=st.floats(min_value=1.0, max_value=7200.0),
        cuts=st.lists(st.floats(min_value=0.0, max_value=7200.0), max_size=60),
        min_len=st.floats(min_value=0.5, max_value=5.0),
        max_len=st.floats(min_value=10.0, max_value=600.0),
    )
    def test_scenes_always_cover_window(
        self,
        duration: float,
        cuts: list[float],
        min_len: float,
        max_len: float,
    ) -> None:
        window = TimeRange(0.0, duration)

        scenes = normalize_scenes(cuts, window, min_len_sec=min_len, max_len_sec=max_len)

        assert_covers(scenes, window)

    @settings(max_examples=200)
    @given(
        duration=st.floats(min_value=10.0, max_value=7200.0),
        cuts=st.lists(st.floats(min_value=0.0, max_value=7200.0), max_size=60),
    )
    def test_length_limits_hold(self, duration: float, cuts: list[float]) -> None:
        window = TimeRange(0.0, duration)
        min_len, max_len = 1.5, 90.0

        scenes = normalize_scenes(cuts, window, min_len_sec=min_len, max_len_sec=max_len)

        # Единственное исключение из «не короче минимума» — окно, которое само
        # короче минимума: делить его не на что.
        if duration >= min_len:
            assert all(scene.duration >= min_len - EPSILON for scene in scenes)
        assert all(scene.duration <= max_len + EPSILON for scene in scenes)

    @settings(max_examples=100)
    @given(cuts=st.lists(st.floats(min_value=0.1, max_value=99.9), max_size=40))
    def test_result_is_deterministic(self, cuts: list[float]) -> None:
        """Один вход — один результат. Постобработка не имеет права зависеть
        ни от порядка склеек, ни от состояния процесса."""
        window = TimeRange(0.0, 100.0)
        kwargs = {"min_len_sec": 1.5, "max_len_sec": 90.0}

        first = normalize_scenes(cuts, window, **kwargs)  # type: ignore[arg-type]
        second = normalize_scenes(list(reversed(cuts)), window, **kwargs)  # type: ignore[arg-type]

        assert first == second
