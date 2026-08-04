"""Темп и плотность речи — чистая арифметика над транскриптом.

Ни модели, ни файлов: транскрипты собираются прямо здесь, поэтому в каждом
тесте видно, из чего получилось ожидаемое число.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from tests.fakes import make_transcript

from core.domain.analysis import AnalysisContext
from core.domain.media import Transcript, TranscriptSegment
from core.domain.values import MediaWindow, TimeRange
from speech.rate import SpeechRateAnalyzer, density_in, turns_in, words_per_minute

pytestmark = pytest.mark.unit


def segments(*spans: tuple[float, float, str]) -> list[TranscriptSegment]:
    return list(make_transcript(*spans).segments)


class TestWordsPerMinute:
    def test_counted_over_speech_time_not_scene_length(self) -> None:
        """Реплика-скороговорка посреди долгой паузы — быстрая речь.
        Делить на длину сцены значило бы утверждать обратное."""
        # Шесть слов за три секунды звучания = 120 слов в минуту,
        # независимо от того, что сцена длится полминуты.
        spoken = segments((10.0, 13.0, "раз два три четыре пять шесть"))

        assert words_per_minute(spoken, TimeRange(0.0, 30.0)) == pytest.approx(120.0, abs=1.0)

    def test_only_the_overlapping_part_counts(self) -> None:
        """Реплика через границу сцены принадлежит обеим — но каждой
        засчитывается только её часть, иначе темп удвоится."""
        spoken = segments((8.0, 12.0, "одно два три четыре пять шесть семь восемь"))

        first = words_per_minute(spoken, TimeRange(0.0, 10.0))
        second = words_per_minute(spoken, TimeRange(10.0, 20.0))

        assert first is not None and second is not None
        assert first == pytest.approx(second, rel=0.35)

    def test_too_few_words_gives_nothing(self) -> None:
        """«Слов в минуту» по двум словам — это шум, умноженный на шестьдесят.
        Отсутствие признака честнее выдуманного числа."""
        assert words_per_minute(segments((1.0, 2.0, "да нет")), TimeRange(0.0, 10.0)) is None

    def test_scene_without_speech(self) -> None:
        assert words_per_minute(segments((50.0, 55.0, "далеко")), TimeRange(0.0, 10.0)) is None


class TestDensity:
    def test_share_of_the_scene_occupied_by_speech(self) -> None:
        spoken = segments((0.0, 3.0, "первая"), (6.0, 9.0, "вторая"))

        assert density_in(spoken, TimeRange(0.0, 12.0)) == pytest.approx(0.5)

    def test_overlapping_segments_do_not_exceed_one(self) -> None:
        """У двух говорящих сегменты накладываются: наивная сумма дала бы
        плотность больше единицы, а это не доля."""
        spoken = segments((0.0, 8.0, "первый"), (2.0, 10.0, "второй"))

        assert density_in(spoken, TimeRange(0.0, 10.0)) == pytest.approx(1.0)

    def test_speech_outside_the_scene_is_ignored(self) -> None:
        assert density_in(segments((20.0, 25.0, "мимо")), TimeRange(0.0, 10.0)) == 0.0

    def test_no_speech(self) -> None:
        assert density_in([], TimeRange(0.0, 10.0)) == 0.0


class TestTurns:
    def test_pauses_between_replies_are_counted(self) -> None:
        """Перепалка звучит иначе, чем монолог той же длины: реплики
        разделены заметными паузами."""
        dialogue = segments(
            (0.0, 2.0, "а ты уверен"),
            (3.0, 5.0, "абсолютно точно уверен"),
            (6.0, 8.0, "ну хорошо тогда"),
        )

        assert turns_in(dialogue, TimeRange(0.0, 10.0)) == 2

    def test_continuous_speech_has_no_turns(self) -> None:
        """Монолог, нарезанный на сегменты без пауз, — это один говорящий."""
        monologue = segments(
            (0.0, 3.0, "и вот тогда я подумал"),
            (3.05, 6.0, "что всё это может закончиться"),
        )

        assert turns_in(monologue, TimeRange(0.0, 10.0)) == 0

    def test_single_segment(self) -> None:
        assert turns_in(segments((0.0, 5.0, "один")), TimeRange(0.0, 10.0)) == 0

    def test_empty(self) -> None:
        assert turns_in([], TimeRange(0.0, 10.0)) == 0


class TestWithoutWordTimings:
    def test_falls_back_to_proportional_estimate(self) -> None:
        """Пословные тайминги — контракт нашего транскрайбера, но не всякой
        чужой реализации. Без них темп оценивается по доле пересечения:
        грубо, зато не ноль."""
        plain = [
            TranscriptSegment(
                text="раз два три четыре пять шесть семь восемь",
                range=TimeRange(0.0, 4.0),
            )
        ]

        assert words_per_minute(plain, TimeRange(0.0, 4.0)) == pytest.approx(120.0, abs=1.0)


class TestAnalyzer:
    @pytest.mark.asyncio
    async def test_delta_is_relative_to_the_video_baseline(self, tmp_path: Path) -> None:
        """Сто восемьдесят слов в минуту у скорого рассказчика — норма,
        а у лектора — событие. Сравнивать нужно с самим видео."""
        transcript = Transcript(
            segments=(
                *make_transcript((0.0, 6.0, "раз два три четыре пять шесть")).segments,
                *make_transcript((10.0, 16.0, "раз два три четыре пять шесть")).segments,
                *make_transcript(
                    (20.0, 23.0, "раз два три четыре пять шесть семь восемь девять")
                ).segments,
            ),
            language="ru",
        )
        context = AnalysisContext(
            job_id=uuid4(),
            video_id=uuid4(),
            window=MediaWindow(source_id="x", range=TimeRange(0.0, 30.0)),
            workdir=tmp_path,
            transcript=transcript,
        )
        scenes = [TimeRange(0.0, 10.0), TimeRange(10.0, 20.0), TimeRange(20.0, 30.0)]

        features = await SpeechRateAnalyzer().analyze(context, scenes)

        deltas = {
            index: value.value
            for index, values in features.items()
            for value in values
            if value.key == "speech.rate.delta"
        }
        assert deltas[0] == pytest.approx(0.0, abs=1.0)
        assert deltas[2] > 100.0
