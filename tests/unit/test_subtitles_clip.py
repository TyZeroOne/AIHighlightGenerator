"""Субтитры клипа: разбиение на реплики, SRT, ASS, караоке.

Всё здесь — чистые функции над выдуманным транскриптом. Это не удобство,
а следствие устройства: субтитры не зависят ни от файлов, ни от ffmpeg,
и требования к ним формулируются как проверяемые свойства текста —
«строк не больше двух», «тайминги монотонны», «время внутри клипа».

Отдельно от ``test_subtitle_formats.py``: там сериализуется транскрипт
целого видео (Этап 4), здесь — готовые реплики клипа.
"""

from __future__ import annotations

import itertools
import re

import pytest

from core.domain.enums import SubtitleFormat
from core.domain.media import Transcript, TranscriptSegment, TranscriptWord
from core.domain.values import TimeRange
from subtitles.ass import chars_that_fit, font_size_for, render_ass
from subtitles.cues import Cue, CueSettings, build_cues, wrap_lines
from subtitles.srt import render_srt, srt_timestamp
from subtitles.writer import ClipSubtitleWriter

pytestmark = pytest.mark.unit


def speech(
    text: str, *, start: float = 100.0, word_sec: float = 0.4, gap: float = 0.05
) -> Transcript:
    """Транскрипт с равномерной речью и известными таймингами."""
    words: list[TranscriptWord] = []
    moment = start
    for token in text.split():
        words.append(TranscriptWord(text=token, start=moment, end=moment + word_sec))
        moment += word_sec + gap
    segment = TranscriptSegment(text=text, range=TimeRange(start, moment), words=tuple(words))
    return Transcript(segments=(segment,), language="ru")


SAMPLE = (
    "Это была самая длинная ночь в моей жизни. "
    "Мы шли через лес и не понимали, куда нас ведут. "
    "Потом всё изменилось."
)


# ------------------------------------------------------------- разбиение
class TestCueLimits:
    """Требования DoD, сформулированные как свойства."""

    def test_lines_are_within_the_broadcast_limit(self) -> None:
        cues = build_cues(speech(SAMPLE), TimeRange(99.0, 130.0))
        assert cues
        for cue in cues:
            assert len(cue.lines) <= 2
            for line in cue.lines:
                assert len(line) <= 42, line

    def test_timings_are_monotonic_and_inside_the_clip(self) -> None:
        span = TimeRange(99.0, 130.0)
        cues = build_cues(speech(SAMPLE), span)

        for cue in cues:
            assert cue.range.start >= 0.0
            assert cue.range.end <= span.duration + 1e-6
        for previous, current in itertools.pairwise(cues):
            assert previous.range.end <= current.range.start + 1e-9

    def test_timings_are_shifted_into_clip_coordinates(self) -> None:
        """Слово на 100.0 в клипе, начатом с 99.0, показывается на 1.0."""
        cues = build_cues(speech(SAMPLE, start=100.0), TimeRange(99.0, 130.0))
        assert cues[0].range.start == pytest.approx(1.0, abs=0.01)

    def test_short_cue_is_stretched_to_be_readable(self) -> None:
        """Одно слово на 0.4 секунды — вспышка, которую не прочитать."""
        cues = build_cues(speech("Стой", word_sec=0.4), TimeRange(99.0, 110.0))
        assert cues[0].range.duration >= 1.0

    def test_long_speech_is_not_held_forever(self) -> None:
        settings = CueSettings(max_duration_sec=3.0)
        cues = build_cues(speech(SAMPLE), TimeRange(99.0, 140.0), settings)
        assert all(cue.range.duration <= 3.0 + 1e-6 for cue in cues)


class TestCueSplitting:
    def test_sentence_end_starts_a_new_cue(self) -> None:
        cues = build_cues(speech("Первое предложение. Второе предложение."), TimeRange(99.0, 120.0))
        assert len(cues) == 2
        assert cues[0].text.endswith("предложение.")

    def test_pause_starts_a_new_cue(self) -> None:
        """Пауза в речи — естественная граница, даже без точки."""
        words = (
            TranscriptWord(text="раз", start=100.0, end=100.4),
            TranscriptWord(text="два", start=102.0, end=102.4),
        )
        transcript = Transcript(
            segments=(
                TranscriptSegment(text="раз два", range=TimeRange(100.0, 102.4), words=words),
            )
        )
        assert len(build_cues(transcript, TimeRange(99.0, 110.0))) == 2

    def test_words_are_never_split(self) -> None:
        cues = build_cues(speech(SAMPLE), TimeRange(99.0, 130.0))
        joined = " ".join(word.text for cue in cues for word in cue.words)
        assert joined == SAMPLE.replace(".", ".").strip()

    def test_segments_are_used_when_word_timings_are_absent(self) -> None:
        """Не всякая модель делит речь на слова — и внешние субтитры тем более."""
        transcript = Transcript(
            segments=(
                TranscriptSegment(text="без пословных таймингов", range=TimeRange(100.0, 103.0)),
            )
        )
        cues = build_cues(transcript, TimeRange(99.0, 110.0))
        assert len(cues) == 1
        assert "без пословных" in cues[0].text


class TestWrapping:
    def test_lines_are_balanced_not_greedy(self) -> None:
        """Жадная набивка даёт «сорок символов и одно слово во второй строке»."""
        lines = wrap_lines("а" * 20 + " " + "б" * 20 + " короткое", 42, 2)
        assert len(lines) == 2
        assert abs(len(lines[0]) - len(lines[1])) < 20

    def test_short_text_stays_on_one_line(self) -> None:
        assert wrap_lines("коротко", 42, 2) == ("коротко",)

    def test_word_longer_than_the_limit_is_not_broken(self) -> None:
        """Разрезанное посередине слово — заметная ошибка; длинная строка — нет."""
        long_word = "а" * 60
        lines = wrap_lines(long_word, 42, 2)
        assert lines == (long_word,)


# ------------------------------------------------------------------ SRT
class TestSrt:
    def test_output_parses_as_srt(self) -> None:
        """Проверка парсером, а не глазами: формат строг к мелочам."""
        body = render_srt(build_cues(speech(SAMPLE), TimeRange(99.0, 130.0)))
        blocks = [block for block in body.split("\n\n") if block.strip()]
        assert blocks

        pattern = re.compile(
            r"^(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.+)$",
            re.DOTALL,
        )
        for index, block in enumerate(blocks, start=1):
            match = pattern.match(block.strip())
            assert match is not None, block
            assert int(match.group(1)) == index

    def test_timestamp_uses_a_comma(self) -> None:
        """С точкой вместо запятой файл открывается не везде, и молча."""
        assert srt_timestamp(4.29) == "00:00:04,290"

    def test_empty_input_gives_empty_file(self) -> None:
        assert render_srt([]) == ""


# ------------------------------------------------------------------ ASS
class TestAss:
    def _cues(self) -> list[Cue]:
        return build_cues(speech(SAMPLE), TimeRange(99.0, 130.0))

    def test_declares_the_frame_it_was_built_for(self) -> None:
        body = render_ass(self._cues(), width=1080, height=1920)
        assert "PlayResX: 1080" in body
        assert "PlayResY: 1920" in body

    def test_font_grows_with_the_frame(self) -> None:
        assert font_size_for(1080) < font_size_for(1920)

    def test_vertical_frame_fits_fewer_characters(self) -> None:
        """Ровно поэтому реплики для вертикали пересобираются короче."""
        assert chars_that_fit(1080, 1920) < chars_that_fit(1920, 1080)

    def test_karaoke_marks_every_word(self) -> None:
        body = render_ass(self._cues(), width=1920, height=1080, karaoke=True)
        events = [line for line in body.splitlines() if line.startswith("Dialogue:")]
        assert events
        for event in events:
            assert "{\\k" in event

    def test_karaoke_can_be_switched_off(self) -> None:
        body = render_ass(self._cues(), width=1920, height=1080, karaoke=False)
        assert "{\\k" not in body

    def test_line_break_uses_the_ass_marker(self) -> None:
        body = render_ass(self._cues(), width=1920, height=1080)
        assert "\\N" in body
        # Настоящего перевода строки внутри события быть не должно —
        # он оборвал бы разбор файла на середине реплики.
        for line in body.splitlines():
            assert not line.startswith(" ")

    def test_timestamps_use_centiseconds(self) -> None:
        """У ASS своя форма времени: часы без нуля и сотые доли секунды."""
        body = render_ass(self._cues(), width=1920, height=1080)
        assert re.search(r"Dialogue: 0,\d:\d{2}:\d{2}\.\d{2},", body)

    def test_markup_characters_are_escaped(self) -> None:
        """Фигурная скобка в речи иначе съела бы половину реплики."""
        transcript = speech("скобка { и } конец")
        body = render_ass(build_cues(transcript, TimeRange(99.0, 110.0)), width=1920, height=1080)
        events = [line for line in body.splitlines() if line.startswith("Dialogue:")]
        text = "".join(events).split(",,", 1)[1]
        assert "{" not in text.replace("{\\k", "")


# --------------------------------------------------------------- writer
class TestWriter:
    def test_writes_requested_format(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        writer = ClipSubtitleWriter()
        path = writer.write(
            speech(SAMPLE), TimeRange(99.0, 130.0), SubtitleFormat.SRT, tmp_path / "clip.srt"
        )
        assert path.read_text(encoding="utf-8").startswith("1\n")

    def test_vertical_ass_shortens_the_lines(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Прямое следствие §«PlayRes»: в вертикали 42 символа не помещаются."""
        writer = ClipSubtitleWriter()
        horizontal = writer.write(
            speech(SAMPLE),
            TimeRange(99.0, 130.0),
            SubtitleFormat.ASS,
            tmp_path / "h.ass",
            width=1920,
            height=1080,
        ).read_text(encoding="utf-8")
        vertical = writer.write(
            speech(SAMPLE),
            TimeRange(99.0, 130.0),
            SubtitleFormat.ASS,
            tmp_path / "v.ass",
            width=1080,
            height=1920,
        ).read_text(encoding="utf-8")

        assert _longest_line(vertical) < _longest_line(horizontal)

    def test_no_speech_gives_a_file_without_events(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        writer = ClipSubtitleWriter()
        path = writer.write(
            Transcript(), TimeRange(0.0, 10.0), SubtitleFormat.ASS, tmp_path / "empty.ass"
        )
        body = path.read_text(encoding="utf-8")
        assert "[Events]" in body
        assert "Dialogue:" not in body


def _longest_line(ass_body: str) -> int:
    """Самая длинная строка текста среди событий, без разметки."""
    longest = 0
    for line in ass_body.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        text = line.split(",,", 1)[1]
        text = re.sub(r"\{\\k\d+\}", "", text)
        longest = max(longest, *(len(part) for part in text.split("\\N")))
    return longest
