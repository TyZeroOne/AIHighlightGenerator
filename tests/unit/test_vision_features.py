"""Видеопризнаки на синтетических кадрах.

Кадры рисуются массивами прямо здесь: чёрный, белый, шум, сдвинутая
картинка. На таких входах ответ известен точно, и видно, из чего он взялся —
в отличие от «прогнали на видео, получили 0.37».
"""

from __future__ import annotations

from typing import Any

import pytest

from core.domain.values import TimeRange
from vision.cuts import count_cuts
from vision.dynamics import brightness_series, entropy_of, histogram_delta
from vision.frames import sample_times

pytestmark = pytest.mark.unit

np = pytest.importorskip("numpy", reason="numpy ставится вместе с OpenCV")


def solid(value: int, size: int = 64) -> Any:
    return np.full((size, size), value, dtype=np.uint8)


def noise(seed: int, size: int = 64) -> Any:
    generator = np.random.default_rng(seed)
    return generator.integers(0, 256, size=(size, size), dtype=np.uint8)


def gradient(size: int = 64) -> Any:
    row = np.linspace(0, 255, size, dtype=np.uint8)
    return np.tile(row, (size, 1))


class TestEntropy:
    def test_uniform_frame_has_zero_entropy(self) -> None:
        """Чёрный экран или титры на однотонном фоне: ноль информации."""
        assert entropy_of(solid(0)) == pytest.approx(0.0)
        assert entropy_of(solid(255)) == pytest.approx(0.0)

    def test_noise_is_near_maximum(self) -> None:
        """Равномерный шум по 256 уровням — предел в 8 бит."""
        assert entropy_of(noise(42)) == pytest.approx(8.0, abs=0.1)

    def test_gradient_is_between(self) -> None:
        value = entropy_of(gradient())

        assert 5.0 < value < 8.0


class TestHistogramDelta:
    def test_identical_frames(self) -> None:
        frame = noise(1)

        assert histogram_delta(frame, frame) == pytest.approx(0.0)

    def test_black_to_white_is_total_difference(self) -> None:
        assert histogram_delta(solid(0), solid(255)) == pytest.approx(1.0)

    def test_shift_does_not_change_the_histogram(self) -> None:
        """Ключевое свойство: панорама сдвигает картинку, но не меняет
        распределение яркостей. Попиксельная разность приняла бы это
        за смену плана — гистограмма не принимает."""
        frame = gradient()
        shifted = np.roll(frame, 5, axis=1)

        assert histogram_delta(frame, shifted) < 0.05


class TestBrightness:
    def test_series_matches_the_frames(self) -> None:
        series = brightness_series([solid(0), solid(128), solid(255)])

        assert series == pytest.approx([0.0, 0.502, 1.0], abs=0.01)

    def test_empty(self) -> None:
        assert brightness_series([]) == []


class TestCuts:
    def test_counts_abrupt_changes(self) -> None:
        """Три плана подряд — две склейки."""
        frames = [solid(0), solid(0), solid(255), solid(255), solid(0)]

        assert count_cuts(frames) == 2

    def test_smooth_sequence_has_no_cuts(self) -> None:
        """Плавное движение не должно считаться монтажом."""
        base = gradient()
        frames = [np.roll(base, shift, axis=1) for shift in range(0, 10, 2)]

        assert count_cuts(frames) == 0

    def test_single_frame(self) -> None:
        assert count_cuts([solid(10)]) == 0

    def test_threshold_is_respected(self) -> None:
        frames = [solid(0), solid(255)]

        assert count_cuts(frames, threshold=0.9) == 1
        assert count_cuts(frames, threshold=1.1) == 0


class TestSampling:
    def test_times_are_evenly_spaced(self) -> None:
        times = sample_times(TimeRange(10.0, 12.0), sample_fps=2.0)

        assert times == pytest.approx([10.0, 10.5, 11.0, 11.5])

    def test_short_scene_still_gets_one_sample(self) -> None:
        """Сцена короче шага сэмплирования — всё равно один кадр:
        пустой результат означал бы «нет данных» там, где данные есть."""
        assert len(sample_times(TimeRange(0.0, 0.1), sample_fps=1.0)) == 1


class TestMotion:
    def test_static_frames_have_no_motion(self) -> None:
        from vision.motion import flow_magnitude

        frame = gradient()

        assert flow_magnitude(frame, frame) == pytest.approx(0.0, abs=1e-4)

    def test_shift_produces_motion(self) -> None:
        """Сдвиг картинки — это движение, и поток обязан его увидеть."""
        from vision.motion import flow_magnitude

        frame = noise(7)
        shifted = np.roll(frame, 8, axis=1)

        assert flow_magnitude(frame, shifted) > flow_magnitude(frame, frame)

    def test_camera_motion_needs_features(self) -> None:
        """На однотонном кадре особых точек нет — оценки не будет.
        None здесь честнее нуля, который скоринг примет за «камера стоит»."""
        from vision.motion import camera_motion

        assert camera_motion(solid(128), solid(128)) is None

    def test_camera_pan_is_detected(self) -> None:
        """Сдвинутый шум — модель панорамирования: сдвиг есть, масштаб тот же."""
        from vision.motion import camera_motion

        frame = noise(11, size=256)
        shifted = np.roll(frame, 20, axis=1)

        motion = camera_motion(frame, shifted)

        assert motion is not None
        assert motion.pan > 0.01
        assert motion.zoom < 0.05
