"""Сериализация транскрипта в SRT, WebVTT и текст.

Форматы простые, но строгие: плееры не прощают ни точки вместо запятой
в SRT, ни отсутствия заголовка в WebVTT. Проверяем именно эти детали,
а не «строка получилась непустой».
"""

from __future__ import annotations

import pytest
from tests.fakes import make_transcript

from core.domain.media import Transcript, TranscriptSegment
from core.domain.values import TimeRange
from subtitles.formats import to_srt, to_text, to_vtt

pytestmark = pytest.mark.unit


@pytest.fixture
def transcript() -> Transcript:
    return make_transcript(
        (921.48, 924.0, "Держись, они прямо за нами"),
        (925.0, 927.5, "Я вижу их"),
    )


class TestSrt:
    def test_structure(self, transcript: Transcript) -> None:
        srt = to_srt(transcript)

        assert srt.startswith("1\n00:15:21,480 --> 00:15:24,000\nДержись")
        assert "\n2\n00:15:25,000 --> 00:15:27,500\nЯ вижу их\n" in srt

    def test_comma_separator_is_mandatory(self, transcript: Transcript) -> None:
        """Точка вместо запятой — самый частый способ получить файл,
        который VLC откроет, а половина остальных плееров молча проигнорирует."""
        stamps = [line for line in to_srt(transcript).splitlines() if "-->" in line]

        assert stamps and all("," in stamp and "." not in stamp for stamp in stamps)

    def test_empty_transcript_gives_empty_file(self) -> None:
        assert to_srt(Transcript()) == ""

    def test_blank_segments_are_skipped_and_numbering_stays_dense(self) -> None:
        """Пустой сегмент между репликами не должен оставлять дыру в нумерации:
        парсеры SRT на этом спотыкаются."""
        transcript = Transcript(
            segments=(
                TranscriptSegment(text="раз", range=TimeRange(0.0, 1.0)),
                TranscriptSegment(text="   ", range=TimeRange(1.0, 2.0)),
                TranscriptSegment(text="два", range=TimeRange(2.0, 3.0)),
            )
        )

        numbers = [line for line in to_srt(transcript).splitlines() if line.isdigit()]

        assert numbers == ["1", "2"]


class TestVtt:
    def test_header_and_dot_separator(self, transcript: Transcript) -> None:
        vtt = to_vtt(transcript)

        assert vtt.startswith("WEBVTT\n\n")
        assert "00:15:21.480 --> 00:15:24.000" in vtt

    def test_no_numbering(self, transcript: Transcript) -> None:
        """В WebVTT номера необязательны, а лишние строки только мешают
        браузеру сопоставлять кью."""
        assert "\n1\n" not in to_vtt(transcript)

    def test_empty_transcript_still_has_header(self) -> None:
        """Файл без заголовка браузер отвергнет целиком, даже пустой."""
        assert to_vtt(Transcript()).startswith("WEBVTT")


class TestText:
    def test_plain_lines_without_timecodes(self, transcript: Transcript) -> None:
        assert to_text(transcript) == "Держись, они прямо за нами\nЯ вижу их"

    def test_empty(self) -> None:
        assert to_text(Transcript()) == ""
