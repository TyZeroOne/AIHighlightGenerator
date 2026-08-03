"""Тесты объектов-значений: ``TimeRange``, ``Score``, ``FeatureValue``, тайм-коды."""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from core.domain.timecode import format_timecode, parse_timecode
from core.domain.values import CropBox, FeatureValue, MediaWindow, Score, TimeRange

pytestmark = pytest.mark.unit


# ------------------------------------------------------------- TimeRange
class TestTimeRange:
    def test_basic_properties(self) -> None:
        span = TimeRange(10.0, 25.5)
        assert span.duration == pytest.approx(15.5)
        assert span.midpoint == pytest.approx(17.75)

    @pytest.mark.parametrize(
        ("start", "end"),
        [
            (5.0, 5.0),  # нулевая длительность
            (10.0, 5.0),  # конец раньше начала
            (-1.0, 5.0),  # отрицательное начало
            (0.0, math.inf),  # бесконечность
            (math.nan, 5.0),  # NaN
        ],
    )
    def test_rejects_invalid(self, start: float, end: float) -> None:
        with pytest.raises(ValueError):
            TimeRange(start, end)

    def test_is_immutable(self) -> None:
        span = TimeRange(0.0, 10.0)
        with pytest.raises(AttributeError):
            span.start = 5.0  # type: ignore[misc]

    def test_is_hashable(self) -> None:
        """Неизменяемость даёт хешируемость — отрезок можно класть в множество."""
        assert len({TimeRange(0.0, 1.0), TimeRange(0.0, 1.0)}) == 1

    def test_contains(self) -> None:
        span = TimeRange(10.0, 20.0)
        assert span.contains(10.0)
        assert span.contains(15.0)
        assert span.contains(19.999)
        assert not span.contains(9.0)
        assert not span.contains(21.0)

    def test_overlaps_and_intersection(self) -> None:
        a, b = TimeRange(0.0, 10.0), TimeRange(5.0, 15.0)
        assert a.overlaps(b) and b.overlaps(a)
        assert a.intersection(b) == TimeRange(5.0, 10.0)

    def test_touching_ranges_do_not_overlap(self) -> None:
        """Отрезки — полуинтервалы: сцена [0,10) и [10,20) не пересекаются.

        Иначе соседние сцены считались бы пересекающимися, и NMS выкидывал
        бы половину кандидатов.
        """
        a, b = TimeRange(0.0, 10.0), TimeRange(10.0, 20.0)
        assert not a.overlaps(b)
        assert a.intersection(b) is None

    def test_iou(self) -> None:
        a, b = TimeRange(0.0, 10.0), TimeRange(5.0, 15.0)
        assert a.iou(b) == pytest.approx(5.0 / 15.0)
        assert a.iou(a) == pytest.approx(1.0)
        assert a.iou(TimeRange(100.0, 110.0)) == 0.0

    def test_shifted_and_expanded(self) -> None:
        span = TimeRange(10.0, 20.0)
        assert span.shifted(-10.0) == TimeRange(0.0, 10.0)
        assert span.expanded(5.0, 5.0) == TimeRange(5.0, 25.0)
        # Ограничение рамками видео
        assert span.expanded(50.0, 50.0, limit=TimeRange(0.0, 30.0)) == TimeRange(0.0, 30.0)

    def test_ordering(self) -> None:
        """Сортировка по началу — нужна везде, где сцены идут по порядку."""
        spans = [TimeRange(20.0, 30.0), TimeRange(0.0, 10.0), TimeRange(10.0, 20.0)]
        assert sorted(spans)[0] == TimeRange(0.0, 10.0)

    # ---------------------------------------------------- property-based
    @given(
        start=st.floats(min_value=0, max_value=1e5, allow_nan=False),
        duration=st.floats(min_value=0.01, max_value=1e4, allow_nan=False),
    )
    def test_duration_always_positive(self, start: float, duration: float) -> None:
        assert TimeRange(start, start + duration).duration > 0

    @given(
        a_start=st.floats(min_value=0, max_value=1000, allow_nan=False),
        a_dur=st.floats(min_value=0.1, max_value=100, allow_nan=False),
        b_start=st.floats(min_value=0, max_value=1000, allow_nan=False),
        b_dur=st.floats(min_value=0.1, max_value=100, allow_nan=False),
    )
    def test_iou_is_symmetric_and_bounded(
        self, a_start: float, a_dur: float, b_start: float, b_dur: float
    ) -> None:
        a = TimeRange(a_start, a_start + a_dur)
        b = TimeRange(b_start, b_start + b_dur)
        assert a.iou(b) == pytest.approx(b.iou(a))
        assert 0.0 <= a.iou(b) <= 1.0 + 1e-9

    @given(
        a_start=st.floats(min_value=0, max_value=1000, allow_nan=False),
        a_dur=st.floats(min_value=1, max_value=100, allow_nan=False),
        # Второй отрезок строим со смещением от первого, а не независимо:
        # два случайных отрезка на [0, 1000] почти никогда не пересекаются,
        # и hypothesis отбраковал бы почти все примеры.
        offset=st.floats(min_value=-100, max_value=100, allow_nan=False),
        b_dur=st.floats(min_value=1, max_value=100, allow_nan=False),
    )
    def test_intersection_is_inside_both(
        self, a_start: float, a_dur: float, offset: float, b_dur: float
    ) -> None:
        a = TimeRange(a_start, a_start + a_dur)
        b_start = max(0.0, a_start + offset)
        b = TimeRange(b_start, b_start + b_dur)

        overlap = a.intersection(b)
        if overlap is None:
            assert not a.overlaps(b)
            return

        # Пересечение обязано лежать внутри ОБОИХ отрезков.
        assert overlap.start >= max(a.start, b.start) - 1e-9
        assert overlap.end <= min(a.end, b.end) + 1e-9
        assert overlap.duration <= min(a.duration, b.duration) + 1e-9


# -------------------------------------------------------------- timecode
class TestTimecode:
    @pytest.mark.parametrize(
        ("text", "seconds"),
        [
            ("00:00:00", 0.0),
            ("00:15:21", 921.0),
            ("00:15:21.480", 921.48),
            ("01:00:00.000", 3600.0),
            ("99:59:59.999", 359999.999),
            ("00:00:01.5", 1.5),  # "5" = 500 мс, а не 5 мс
            ("00:00:01.05", 1.05),
        ],
    )
    def test_parse(self, text: str, seconds: float) -> None:
        assert parse_timecode(text) == pytest.approx(seconds)

    @pytest.mark.parametrize(
        "text",
        ["15:21", "00:60:00", "00:00:60", "abc", "", "0:15:21", "00:15:21.4801"],
    )
    def test_parse_rejects_garbage(self, text: str) -> None:
        with pytest.raises(ValueError):
            parse_timecode(text)

    def test_format(self) -> None:
        assert format_timecode(921.48) == "00:15:21.480"
        assert format_timecode(921.48, with_millis=False) == "00:15:21"
        assert format_timecode(0.0) == "00:00:00.000"

    def test_format_rejects_negative(self) -> None:
        with pytest.raises(ValueError):
            format_timecode(-1.0)

    def test_rounding_does_not_produce_60_seconds(self) -> None:
        """59.9996 не должно превратиться в ``00:00:59.1000``."""
        assert format_timecode(59.9996) == "00:01:00.000"

    @given(seconds=st.floats(min_value=0, max_value=359999.9, allow_nan=False))
    def test_roundtrip(self, seconds: float) -> None:
        """``parse(format(x)) == x`` с точностью до миллисекунды."""
        assert parse_timecode(format_timecode(seconds)) == pytest.approx(seconds, abs=0.001)


# ----------------------------------------------------------------- Score
class TestScore:
    def test_valid(self) -> None:
        score = Score(96.2, {"action": 0.94, "emotion": 0.71})
        assert score.value == pytest.approx(96.2)
        assert score.breakdown["action"] == pytest.approx(0.94)

    @pytest.mark.parametrize("value", [-1.0, 100.1, 1000.0])
    def test_rejects_out_of_range(self, value: float) -> None:
        with pytest.raises(ValueError):
            Score(value)

    def test_breakdown_is_frozen(self) -> None:
        """Иначе «неизменяемый» Score давал бы изменяемый словарь."""
        score = Score(50.0, {"a": 1.0})
        with pytest.raises(TypeError):
            score.breakdown["b"] = 2.0  # type: ignore[index]

    def test_breakdown_is_copied(self) -> None:
        source = {"a": 1.0}
        score = Score(50.0, source)
        source["b"] = 2.0
        assert "b" not in score.breakdown


# ---------------------------------------------------------- прочие VO
class TestOtherValues:
    def test_feature_value_validates_confidence(self) -> None:
        FeatureValue("audio.loudness.peak", -3.2, 0.9)
        with pytest.raises(ValueError):
            FeatureValue("x", 1.0, confidence=1.5)
        with pytest.raises(ValueError):
            FeatureValue("", 1.0)
        with pytest.raises(ValueError):
            FeatureValue("x", math.nan)

    def test_media_window_defaults_to_final(self) -> None:
        window = MediaWindow("video-1", TimeRange(0.0, 120.0))
        assert window.is_final is True
        assert window.index == 0

    def test_media_window_for_stream(self) -> None:
        """Стриминговое окно отличается только флагом — больше ничем."""
        window = MediaWindow("twitch:chan", TimeRange(60.0, 120.0), index=1, is_final=False)
        assert not window.is_final

    def test_media_window_requires_source_id(self) -> None:
        with pytest.raises(ValueError):
            MediaWindow("", TimeRange(0.0, 1.0))

    def test_crop_box(self) -> None:
        box = CropBox(100, 0, 1080, 1920)
        assert box.center_x == pytest.approx(640.0)
        with pytest.raises(ValueError):
            CropBox(0, 0, 0, 100)
