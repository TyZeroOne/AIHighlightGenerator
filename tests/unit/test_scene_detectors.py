"""Разбор вывода ``scdet`` и перевод порога между шкалами.

Обе вещи чистые: строка на входе, числа на выходе. Проверяются на записанном
выводе настоящего ffmpeg 7.1 — без запуска процессов и без видеофайлов.
Сами детекторы на настоящем видео проверяются в
``tests/integration/test_scene_detection_real.py``.
"""

from __future__ import annotations

import pytest

from video_loader.scenes.ffmpeg import parse_scdet_output, scdet_threshold

pytestmark = pytest.mark.unit


class TestThresholdScale:
    def test_threshold_is_rescaled_not_copied(self) -> None:
        """У PySceneDetect шкала 0..255, у scdet — проценты. Перенести число
        как есть значило бы дать двум «взаимозаменяемым» реализациям разную
        чувствительность при одном и том же ``SCENE_THRESHOLD``."""
        assert scdet_threshold(27.0) == pytest.approx(10.588, abs=0.001)
        assert scdet_threshold(255.0) == pytest.approx(100.0)
        assert scdet_threshold(0.0) == pytest.approx(0.0)

    def test_out_of_range_values_are_clamped(self) -> None:
        """Конфиг может содержать что угодно; ffmpeg на пороге 200 % откажется
        запускаться, и разбираться в этом придётся посреди этапа."""
        assert scdet_threshold(1000.0) == pytest.approx(100.0)
        assert scdet_threshold(-5.0) == pytest.approx(0.0)


class TestScdetParser:
    def test_reads_real_ffmpeg_output(self) -> None:
        """Формат — как его печатает `metadata=print` у ffmpeg 7.1."""
        raw = (
            "frame:50   pts:50050   pts_time:2.002\n"
            "lavfi.scd.time=2.002\n"
            "lavfi.scd.score=42.750\n"
            "frame:100  pts:100100  pts_time:4.004\n"
            "lavfi.scd.time=4.004\n"
            "lavfi.scd.score=38.120\n"
        )

        assert parse_scdet_output(raw) == [2.002, 4.004]

    def test_falls_back_to_pts_time(self) -> None:
        """Сборка без ``scd.time`` в метаданных: момент берётся из заголовка
        кадра. На кадр позже, чем хотелось бы, но это лучше, чем потерять
        склейку целиком."""
        raw = "frame:50 pts:50050 pts_time:2.002\nlavfi.scd.score=42.750\n"

        assert parse_scdet_output(raw) == [2.002]

    def test_empty_output_means_no_cuts(self) -> None:
        """Статичное видео: склеек нет, и это нормальный ответ, а не сбой."""
        assert parse_scdet_output("") == []

    def test_duplicates_are_collapsed_and_sorted(self) -> None:
        raw = "lavfi.scd.time=4.004\nlavfi.scd.time=2.002\nlavfi.scd.time=4.004\n"

        assert parse_scdet_output(raw) == [2.002, 4.004]

    def test_unrelated_lines_are_ignored(self) -> None:
        raw = "Stream #0:0 -> #0:0 (h264 -> wrapped_avframe)\nlavfi.scd.time=1.5\n"

        assert parse_scdet_output(raw) == [1.5]
