"""Аудиопризнаки: разбор вывода ffmpeg, агрегация по сценам, расчёт пауз.

Ни ffmpeg, ни модели — только чистые функции. Вывод ffmpeg взят с настоящего
прогона: тест на выдуманном формате проверял бы фантазию автора, а не
совместимость с инструментом.
"""

from __future__ import annotations

import math

import pytest

from audio.loudness import LoudnessSample, aggregate, parse_ebur128
from audio.pauses import PauseAnalyzer, pauses_in, silence_between
from core.domain.values import FeatureValue, TimeRange

pytestmark = pytest.mark.unit

#: Настоящий вывод ffmpeg 7.1: `ebur128=peak=true:metadata=1,ametadata=print`.
#: Обратите внимание на true_peak — он линейный, а не в децибелах.
REAL_OUTPUT = """\
frame:0    pts:0       pts_time:0
lavfi.r128.M=-22.500
lavfi.r128.S=-23.100
lavfi.r128.I=-70.000
lavfi.r128.LRA=0.000
lavfi.r128.true_peaks_ch0=0.125
lavfi.r128.true_peak=0.125
frame:1    pts:1600    pts_time:0.1
lavfi.r128.M=-20.100
lavfi.r128.S=-21.000
lavfi.r128.true_peak=0.250
frame:2    pts:3200    pts_time:0.2
lavfi.r128.M=-120.691
lavfi.r128.S=-120.691
lavfi.r128.true_peak=0.000
"""


def samples(*rows: tuple[float, float]) -> list[LoudnessSample]:
    """Отсчёты «время, громкость» с тем же значением short-term и пиком."""
    return [
        LoudnessSample(at_sec=at, momentary=level, short_term=level, true_peak=level + 5)
        for at, level in rows
    ]


def by_key(values: list[FeatureValue]) -> dict[str, float]:
    return {value.key: value.value for value in values}


class TestParser:
    def test_reads_real_ffmpeg_output(self) -> None:
        parsed = parse_ebur128(REAL_OUTPUT)

        assert len(parsed) == 3
        assert parsed[0].at_sec == pytest.approx(0.0)
        assert parsed[0].momentary == pytest.approx(-22.5)
        assert parsed[0].short_term == pytest.approx(-23.1)
        assert parsed[1].at_sec == pytest.approx(0.1)

    def test_true_peak_is_converted_to_decibels(self) -> None:
        """ffmpeg отдаёт пик линейной амплитудой (0.125), а рядом с LUFS
        нужны децибелы: иначе скоринг сравнивал бы величины в разных
        шкалах и «тихая» сцена оказалась бы громче взрыва."""
        parsed = parse_ebur128(REAL_OUTPUT)

        assert parsed[0].true_peak == pytest.approx(-18.06, abs=0.01)
        assert parsed[1].true_peak == pytest.approx(-12.04, abs=0.01)

    def test_zero_peak_is_not_minus_infinity_decibels(self) -> None:
        """Абсолютная тишина: логарифм нуля не существует, и признака
        просто нет — это честнее любого выдуманного числа."""
        parsed = parse_ebur128(REAL_OUTPUT)

        assert parsed[2].true_peak is None

    def test_frame_without_metrics_is_skipped(self) -> None:
        """Заголовок кадра без метрик встречается в конце потока."""
        assert parse_ebur128("frame:9 pts:0 pts_time:1.0\n") == []

    def test_unrelated_lines_are_ignored(self) -> None:
        noise = "Input #0, wav, from 'audio.wav':\n  Duration: 00:01:00.00\n"

        assert parse_ebur128(noise) == []

    def test_empty_input(self) -> None:
        assert parse_ebur128("") == []


class TestAggregate:
    def test_features_land_on_the_right_scenes(self) -> None:
        scenes = [TimeRange(0.0, 10.0), TimeRange(10.0, 20.0)]
        series = samples((1.0, -30.0), (5.0, -30.0), (12.0, -12.0), (18.0, -12.0))

        features = aggregate(series, scenes)

        assert by_key(features[0])["audio.loudness.mean"] == pytest.approx(-30.0, abs=0.1)
        assert by_key(features[1])["audio.loudness.mean"] == pytest.approx(-12.0, abs=0.1)

    def test_mean_is_energy_based_not_arithmetic(self) -> None:
        """Громкость логарифмическая: среднее −40 и −10 на слух ближе к −13,
        а не к −25. Арифметика по децибелам утопила бы всплеск, ради которого
        признак и считается."""
        features = aggregate(samples((1.0, -40.0), (2.0, -10.0)), [TimeRange(0.0, 5.0)])

        assert by_key(features[0])["audio.loudness.mean"] == pytest.approx(-13.0, abs=0.1)

    def test_delta_is_relative_to_the_video_median(self) -> None:
        """«Громко» — понятие относительное: в фильме ужасов и в лекции оно
        разное (док 05, §5.3)."""
        scenes = [TimeRange(0.0, 10.0), TimeRange(10.0, 20.0), TimeRange(20.0, 30.0)]
        series = samples((5.0, -30.0), (15.0, -30.0), (25.0, -10.0))

        features = aggregate(series, scenes)

        assert by_key(features[0])["audio.loudness.delta"] == pytest.approx(0.0, abs=0.1)
        assert by_key(features[2])["audio.loudness.delta"] == pytest.approx(20.0, abs=0.1)

    def test_peak_is_the_maximum_true_peak(self) -> None:
        features = aggregate(samples((1.0, -30.0), (2.0, -12.0)), [TimeRange(0.0, 5.0)])

        assert by_key(features[0])["audio.loudness.peak"] == pytest.approx(-7.0, abs=0.1)

    def test_silence_does_not_drag_the_mean_down(self) -> None:
        """Секунда тишины посреди сцены не должна уводить среднее на −70:
        отсчёты ниже порога тишины не участвуют в расчёте."""
        series = [
            *samples((1.0, -20.0)),
            LoudnessSample(at_sec=2.0, momentary=-math.inf, short_term=-math.inf),
            *samples((3.0, -20.0)),
        ]

        features = aggregate(series, [TimeRange(0.0, 5.0)])

        assert by_key(features[0])["audio.loudness.mean"] == pytest.approx(-20.0, abs=0.1)

    def test_dynamic_range_needs_enough_samples(self) -> None:
        """На двух отсчётах процентили бессмысленны — признака просто нет,
        и это честнее, чем выдать число из воздуха."""
        short = aggregate(samples((1.0, -20.0), (2.0, -10.0)), [TimeRange(0.0, 5.0)])

        assert "audio.loudness.dynamic_range" not in by_key(short[0])

    def test_dynamic_range_on_a_long_scene(self) -> None:
        series = samples(*[(float(i) * 0.1, -30.0 + i) for i in range(20)])

        features = aggregate(series, [TimeRange(0.0, 5.0)])

        assert by_key(features[0])["audio.loudness.dynamic_range"] > 0

    def test_scene_without_samples_gets_no_features(self) -> None:
        """Сцена целиком в тишине: признаков нет вовсе. Скоринг это переживёт
        (док 05, §5.4), а ноль вместо «нет данных» соврал бы."""
        features = aggregate(samples((1.0, -20.0)), [TimeRange(0.0, 5.0), TimeRange(5.0, 10.0)])

        assert 1 not in features

    def test_empty_input(self) -> None:
        assert aggregate([], [TimeRange(0.0, 5.0)]) == {}
        assert aggregate(samples((1.0, -20.0)), []) == {}


class TestSilence:
    def test_gaps_between_speech(self) -> None:
        speech = [TimeRange(1.0, 2.0), TimeRange(4.0, 5.0)]

        gaps = silence_between(speech, TimeRange(0.0, 6.0))

        assert gaps == [TimeRange(0.0, 1.0), TimeRange(2.0, 4.0), TimeRange(5.0, 6.0)]

    def test_scene_without_speech_is_all_silence(self) -> None:
        assert silence_between([], TimeRange(0.0, 6.0)) == [TimeRange(0.0, 6.0)]

    def test_speech_crossing_the_border_is_clipped(self) -> None:
        """Реплика может начаться до сцены и кончиться после неё — считать
        нужно только пересечение, иначе тишина уйдёт в минус."""
        speech = [TimeRange(0.0, 12.0)]

        assert silence_between(speech, TimeRange(5.0, 10.0)) == []

    def test_speech_outside_the_window_is_ignored(self) -> None:
        speech = [TimeRange(20.0, 25.0)]

        assert silence_between(speech, TimeRange(0.0, 5.0)) == [TimeRange(0.0, 5.0)]


class TestPauses:
    def test_longest_pause_matters_more_than_total(self) -> None:
        """Десять пауз по полсекунды — обычный ритм речи. Одна пауза
        на четыре секунды — событие."""
        speech = [TimeRange(0.0, 3.0), TimeRange(7.0, 10.0)]

        values = by_key(pauses_in(speech, TimeRange(0.0, 10.0)))

        assert values["audio.pause.max"] == pytest.approx(4.0)
        assert values["audio.pause.total"] == pytest.approx(4.0)
        assert values["audio.pause.ratio"] == pytest.approx(0.4)

    def test_full_silence(self) -> None:
        values = by_key(pauses_in([], TimeRange(0.0, 8.0)))

        assert values["audio.pause.ratio"] == pytest.approx(1.0)
        assert values["audio.pause.max"] == pytest.approx(8.0)

    def test_continuous_speech(self) -> None:
        values = by_key(pauses_in([TimeRange(0.0, 8.0)], TimeRange(0.0, 8.0)))

        assert values["audio.pause.total"] == pytest.approx(0.0)
        assert values["audio.pause.ratio"] == pytest.approx(0.0)

    def test_ratio_never_exceeds_one(self) -> None:
        """Защита от арифметики на границах: доля тишины — это доля."""
        values = by_key(pauses_in([], TimeRange(0.0, 0.001)))

        assert values["audio.pause.ratio"] <= 1.0


class TestVadSettings:
    def test_speech_is_not_padded(self) -> None:
        """Регрессия на живой дефект.

        Silero по умолчанию расширяет каждый интервал речи на 400 мс в обе
        стороны, чтобы модель распознавания не срезала начало слова. Для
        измерения пауз это губительно: из каждого промежутка исчезает
        по 800 мс, и на настоящем клипе пауз не находилось совсем —
        двенадцать сцен подряд с ratio 0.00.
        """
        assert PauseAnalyzer()._speech_pad_ms == 0

    def test_short_pauses_are_not_merged_away(self) -> None:
        """Порог тишины мягче, чем при распознавании: пауза в треть секунды
        здесь измеряемая величина, а не помеха."""
        assert PauseAnalyzer()._min_silence_ms <= 200
