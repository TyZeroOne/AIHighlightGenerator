"""Обработка результата распознавания: куски, мусор, тайминги.

Ни модели, ни аудио — только чистые функции. Именно поэтому свойства вроде
«тайминги монотонны» проверяются здесь на сотнях случайных входов, а не
на одном записанном ответе Whisper.
"""

from __future__ import annotations

import itertools

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.fakes import make_transcript

from core.domain.media import Transcript, TranscriptSegment, TranscriptWord
from core.domain.values import TimeRange
from speech.postprocess import (
    Chunk,
    chunk_windows,
    drop_non_speech,
    merge_chunks,
    normalize_words,
    shift,
)

pytestmark = pytest.mark.unit


def segment(
    start: float,
    end: float,
    text: str,
    *,
    no_speech: float | None = 0.05,
    logprob: float | None = -0.3,
) -> TranscriptSegment:
    return TranscriptSegment(
        text=text,
        range=TimeRange(start, end),
        language="ru",
        no_speech_prob=no_speech,
        avg_logprob=logprob,
    )


class TestChunkWindows:
    def test_short_file_is_one_window(self) -> None:
        assert chunk_windows(300.0, chunk_sec=600.0, overlap_sec=2.0) == [TimeRange(0.0, 300.0)]

    def test_zero_chunk_means_no_splitting(self) -> None:
        """0 — это «отдать модели файл целиком», а не «резать по нулю»."""
        assert chunk_windows(5000.0, chunk_sec=0.0, overlap_sec=2.0) == [TimeRange(0.0, 5000.0)]

    def test_windows_overlap_and_cover_everything(self) -> None:
        windows = chunk_windows(1500.0, chunk_sec=600.0, overlap_sec=10.0)

        assert windows[0].start == 0.0
        assert windows[-1].end == 1500.0
        for previous, current in itertools.pairwise(windows):
            # Перекрытие обязательно: без него на стыке теряются слова.
            assert current.start < previous.end
            assert previous.end - current.start == pytest.approx(10.0)

    def test_empty_audio(self) -> None:
        assert chunk_windows(0.0, chunk_sec=600.0, overlap_sec=2.0) == []


class TestShift:
    def test_segments_and_words_move_together(self) -> None:
        transcript = make_transcript((0.0, 2.0, "привет мир"))

        moved = shift(transcript, 600.0)

        assert moved.segments[0].range == TimeRange(600.0, 602.0)
        assert moved.segments[0].words[0].start == pytest.approx(600.0)
        assert moved.segments[0].words[-1].end == pytest.approx(602.0)

    def test_zero_shift_returns_same_object(self) -> None:
        transcript = make_transcript((0.0, 2.0, "раз"))

        assert shift(transcript, 0.0) is transcript


class TestMergeChunks:
    def test_single_chunk_passes_through(self) -> None:
        transcript = make_transcript((1.0, 2.0, "раз"))

        merged = merge_chunks([Chunk(TimeRange(0.0, 600.0), transcript)])

        assert merged.segments == transcript.segments

    def test_overlap_is_cut_in_the_middle(self) -> None:
        """Фраза из зоны перекрытия попадает ровно в один кусок.

        Куски [0, 100] и [90, 190], перекрытие [90, 100], середина — 95.
        Реплика на 92 с принадлежит первому, на 97 — второму.
        """
        first = Chunk(
            TimeRange(0.0, 100.0),
            Transcript(segments=(segment(91.0, 93.0, "раньше"), segment(96.0, 98.0, "позже"))),
        )
        second = Chunk(
            TimeRange(90.0, 190.0),
            Transcript(segments=(segment(91.0, 93.0, "раньше"), segment(96.0, 98.0, "позже"))),
        )

        merged = merge_chunks([first, second])

        assert [seg.text for seg in merged.segments] == ["раньше", "позже"]

    def test_no_duplicates_on_three_chunks(self) -> None:
        chunks = [
            Chunk(
                TimeRange(0.0, 100.0),
                Transcript(segments=(segment(10.0, 12.0, "а"), segment(95.0, 97.0, "стык1"))),
            ),
            Chunk(
                TimeRange(90.0, 190.0),
                Transcript(segments=(segment(95.0, 97.0, "стык1"), segment(185.0, 187.0, "стык2"))),
            ),
            Chunk(
                TimeRange(180.0, 280.0),
                Transcript(segments=(segment(185.0, 187.0, "стык2"), segment(200.0, 202.0, "б"))),
            ),
        ]

        merged = merge_chunks(chunks)

        assert [seg.text for seg in merged.segments] == ["а", "стык1", "стык2", "б"]

    def test_result_is_sorted_by_time(self) -> None:
        chunks = [
            Chunk(TimeRange(90.0, 190.0), Transcript(segments=(segment(150.0, 152.0, "второй"),))),
            Chunk(TimeRange(0.0, 100.0), Transcript(segments=(segment(10.0, 12.0, "первый"),))),
        ]

        merged = merge_chunks(chunks)

        assert [seg.text for seg in merged.segments] == ["первый", "второй"]

    def test_language_is_taken_from_the_first_chunk_that_knows_it(self) -> None:
        chunks = [
            Chunk(TimeRange(0.0, 100.0), Transcript(segments=(), language=None)),
            Chunk(TimeRange(90.0, 190.0), Transcript(segments=(), language="en")),
        ]

        assert merge_chunks(chunks).language == "en"

    def test_empty_input(self) -> None:
        assert merge_chunks([]).is_empty


class TestDropNonSpeech:
    def test_high_no_speech_probability_with_low_confidence_is_dropped(self) -> None:
        """Оба признака вместе: окно «молчит» и текст неуверенный."""
        transcript = Transcript(
            segments=(
                segment(0.0, 2.0, "настоящая реплика", no_speech=0.05),
                segment(2.0, 4.0, "Продолжение следует...", no_speech=0.95, logprob=-1.4),
            )
        )

        cleaned = drop_non_speech(transcript, no_speech_threshold=0.6)

        assert [seg.text for seg in cleaned.segments] == ["настоящая реплика"]

    def test_confident_text_survives_a_noisy_window(self) -> None:
        """Регрессия на реальный дефект.

        ``no_speech_prob`` относится к **тридцатисекундному окну**, а не
        к сегменту: все сегменты внутри окна получают одно значение.
        Музыкальная отбивка в начале окна поднимает его до 0.74 — и наивный
        порог выбрасывал вместе с ней всю следующую за ней речь.

        Данные взяты с живого прогона: шесть сегментов подряд с ``0.74``
        и ``logprob=-0.17`` — тридцать секунд текста, в котором модель
        была уверена почти полностью.
        """
        transcript = Transcript(
            segments=tuple(
                segment(
                    23.0 + index * 5,
                    28.0 + index * 5,
                    f"реплика {index}",
                    no_speech=0.74,
                    logprob=-0.17,
                )
                for index in range(6)
            )
        )

        cleaned = drop_non_speech(transcript, no_speech_threshold=0.6)

        assert len(cleaned.segments) == 6, "уверенный текст нельзя выбрасывать"

    def test_unknown_confidence_falls_back_to_the_window(self) -> None:
        """Реализация без ``avg_logprob`` — оценка окна остаётся единственным
        признаком, и доверять приходится ей."""
        transcript = Transcript(segments=(segment(0.0, 2.0, "шум", no_speech=0.95, logprob=None),))

        assert not drop_non_speech(transcript).segments

    def test_empty_and_punctuation_only_segments_are_dropped(self) -> None:
        transcript = Transcript(
            segments=(segment(0.0, 1.0, "   "), segment(1.0, 2.0, "..."), segment(2.0, 3.0, "да"))
        )

        cleaned = drop_non_speech(transcript)

        assert [seg.text for seg in cleaned.segments] == ["да"]

    def test_decoder_loop_is_removed_entirely(self) -> None:
        """Зацикливание на музыке: одна и та же строка десять раз подряд.

        Удаляется вся серия, включая первые вхождения: они такой же артефакт,
        просто по ним одним этого ещё не видно.
        """
        transcript = Transcript(
            segments=(
                *(segment(float(i), float(i + 1), "Субтитры сделал DimaTorzok") for i in range(10)),
                segment(20.0, 22.0, "живая речь"),
            )
        )

        cleaned = drop_non_speech(transcript)

        assert [seg.text for seg in cleaned.segments] == ["живая речь"]

    def test_two_repeats_survive(self) -> None:
        """«Да. Да.» — это речь, а не петля. Порог начинается с третьего повтора."""
        transcript = Transcript(segments=(segment(0.0, 1.0, "Да."), segment(1.0, 2.0, "Да.")))

        cleaned = drop_non_speech(transcript)

        assert len(cleaned.segments) == 2

    def test_quiet_speech_in_a_speaking_window_survives(self) -> None:
        """Шёпот в фильме — это содержание, а не помеха. Низкая уверенность
        сама по себе не повод выбрасывать: окно-то говорит."""
        quiet = TranscriptSegment(
            text="еле слышно",
            range=TimeRange(0.0, 2.0),
            avg_logprob=-1.8,
            no_speech_prob=0.1,
        )

        assert len(drop_non_speech(Transcript(segments=(quiet,))).segments) == 1


class TestNormalizeWords:
    def test_reversed_word_is_fixed(self) -> None:
        words = [TranscriptWord(text="ой", start=5.0, end=4.0)]

        fixed = normalize_words(words)

        assert fixed[0].start <= fixed[0].end

    def test_words_are_clamped_to_the_segment(self) -> None:
        span = TimeRange(10.0, 12.0)
        words = [
            TranscriptWord(text="до", start=9.0, end=10.5),
            TranscriptWord(text="после", start=11.5, end=13.0),
        ]

        fixed = normalize_words(words, bounds=span)

        assert fixed[0].start >= 10.0
        assert fixed[-1].end <= 12.0

    def test_nothing_is_lost(self) -> None:
        """Потерянное слово хуже слова с подправленным на 10 мс таймингом."""
        words = [TranscriptWord(text=str(index), start=1.0, end=1.0) for index in range(5)]

        assert len(normalize_words(words)) == 5

    @settings(max_examples=200)
    @given(
        raw=st.lists(
            st.tuples(
                st.floats(min_value=0.0, max_value=100.0),
                st.floats(min_value=0.0, max_value=100.0),
            ),
            max_size=30,
        )
    )
    def test_result_is_always_monotonic(self, raw: list[tuple[float, float]]) -> None:
        """Свойство из DoD этапа: пословные тайминги обязаны быть монотонны.

        На них держатся караоке-субтитры (Этап 9) и снап границ клипа
        к паузе (Этап 7); слово с отрицательной длительностью ломает и то,
        и другое — а плеер просто не покажет такой субтитр.
        """
        words = [
            TranscriptWord(text=f"w{index}", start=start, end=end)
            for index, (start, end) in enumerate(raw)
        ]

        fixed = normalize_words(words)

        assert all(word.start <= word.end for word in fixed)
        for previous, current in itertools.pairwise(fixed):
            assert previous.end <= current.start + 1e-9
