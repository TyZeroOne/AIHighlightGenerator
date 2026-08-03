"""Тесты разбора вывода ffprobe и правил допуска медиа.

``parse_probe`` — чистая функция, поэтому тестируется на записанных JSON'ах
без единого видеофайла и без запуска ffprobe. Это позволяет держать здесь
случаи, которые вручную не воспроизвести: файл без длительности в контейнере,
обложка альбома вместо видеопотока, переменный fps.
"""

from __future__ import annotations

import pytest

from core.domain.errors import (
    FileTooLargeError,
    SourceError,
    UnsupportedMediaError,
    ValidationError,
)
from core.domain.media import MediaInfo
from core.domain.validation import container_family, validate_media, validate_upload_name
from video_loader.probe import parse_probe

pytestmark = pytest.mark.unit

ALLOWED = frozenset({"mp4", "mkv", "mov"})


def probe_json(
    *,
    streams: list[dict[str, object]] | None = None,
    fmt: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "streams": streams if streams is not None else [_video_stream(), _audio_stream()],
        "format": fmt if fmt is not None else {"format_name": "mov,mp4,m4a", "duration": "120.5"},
    }


def _video_stream(**overrides: object) -> dict[str, object]:
    stream: dict[str, object] = {
        "codec_type": "video",
        "codec_name": "h264",
        "width": 1920,
        "height": 1080,
        "avg_frame_rate": "24000/1001",
    }
    stream.update(overrides)
    return stream


def _audio_stream(**overrides: object) -> dict[str, object]:
    stream: dict[str, object] = {"codec_type": "audio", "codec_name": "aac"}
    stream.update(overrides)
    return stream


class TestParseProbe:
    def test_typical_file(self) -> None:
        info = parse_probe(probe_json(), size_bytes=12345)

        assert info.duration_sec == pytest.approx(120.5)
        assert (info.width, info.height) == (1920, 1080)
        assert info.fps == pytest.approx(23.976, abs=0.001)
        assert info.video_codec == "h264"
        assert info.audio_codec == "aac"
        assert info.has_audio and info.has_video
        assert info.size_bytes == 12345

    def test_mp4_and_mov_share_one_container_family(self) -> None:
        """ffprobe не различает .mp4 и .mov — это один формат ISO BMFF.

        Если бы мы сравнивали расширения, честный .mov отвергался бы
        как «не mp4». Поэтому в MediaInfo лежит семейство.
        """
        info = parse_probe(
            probe_json(fmt={"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "10"})
        )
        assert info.container == "mp4"

    def test_matroska_normalized_to_mkv(self) -> None:
        info = parse_probe(probe_json(fmt={"format_name": "matroska,webm", "duration": "10"}))
        assert info.container == "mkv"

    def test_duration_falls_back_to_streams(self) -> None:
        """У части mkv длительности в заголовке контейнера нет — только у дорожек."""
        info = parse_probe(
            probe_json(
                streams=[_video_stream(duration="95.0"), _audio_stream(duration="97.5")],
                fmt={"format_name": "matroska,webm"},
            )
        )
        # Максимум из дорожек: аудио может тянуться дольше видео.
        assert info.duration_sec == pytest.approx(97.5)

    def test_audio_only_file(self) -> None:
        info = parse_probe(
            probe_json(streams=[_audio_stream()], fmt={"format_name": "mov,mp4", "duration": "60"})
        )
        assert info.has_audio and not info.has_video
        assert info.width is None and info.fps is None

    def test_album_art_is_not_video(self) -> None:
        """Обложка лежит как видеопоток из одного кадра.

        Без фильтра по ``attached_pic`` mp3 с картинкой опознавался бы
        как видеофайл 1920×1080 — и дальше конвейер пытался бы искать
        в нём сцены.
        """
        info = parse_probe(
            probe_json(
                streams=[
                    _video_stream(disposition={"attached_pic": 1}),
                    _audio_stream(),
                ],
                fmt={"format_name": "mov,mp4", "duration": "180"},
            )
        )
        assert not info.has_video
        assert info.has_audio

    @pytest.mark.parametrize(
        "raw_fps",
        ["0/0", "N/A", None, "1/0"],
    )
    def test_broken_fps_gives_none_not_crash(self, raw_fps: str | None) -> None:
        stream = _video_stream()
        stream["avg_frame_rate"] = raw_fps
        info = parse_probe(probe_json(streams=[stream]))
        assert info.fps is None

    def test_no_streams_is_rejected(self) -> None:
        with pytest.raises(SourceError, match="не медиафайл"):
            parse_probe({"streams": [], "format": {"format_name": "data"}})

    def test_zero_duration_is_rejected(self) -> None:
        with pytest.raises(SourceError, match="длительность"):
            parse_probe(probe_json(fmt={"format_name": "mov,mp4", "duration": "0"}))

    def test_missing_optional_fields_do_not_crash(self) -> None:
        """Битые и неполные метаданные — норма для файлов из интернета."""
        info = parse_probe(
            {
                "streams": [{"codec_type": "video", "width": "N/A", "height": None}],
                "format": {"format_name": "matroska", "duration": "42"},
            }
        )
        assert info.width is None and info.height is None
        assert info.duration_sec == pytest.approx(42.0)


class TestContainerFamily:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("movie.mp4", "mp4"),
            ("movie.MOV", "mp4"),
            ("mkv", "mkv"),
            ("clip.webm", "mkv"),
            ("archive.zip", None),
            ("noext", None),
        ],
    )
    def test_mapping(self, value: str, expected: str | None) -> None:
        assert container_family(value) == expected


class TestValidateUploadName:
    def test_accepts_allowed(self) -> None:
        assert validate_upload_name("Фильм 2024.MKV", ALLOWED) == "mkv"

    @pytest.mark.parametrize(
        "filename",
        ["notes.txt", "archive.zip", "video.avi", "video.exe"],
    )
    def test_rejects_disallowed_extension(self, filename: str) -> None:
        with pytest.raises(UnsupportedMediaError):
            validate_upload_name(filename, ALLOWED)

    def test_rejects_missing_extension(self) -> None:
        with pytest.raises(UnsupportedMediaError, match="расширения"):
            validate_upload_name("movie", ALLOWED)

    @pytest.mark.parametrize("filename", ["", "   ", ".", ".."])
    def test_rejects_garbage(self, filename: str) -> None:
        with pytest.raises(ValidationError):
            validate_upload_name(filename, ALLOWED)

    def test_strips_directory_components(self) -> None:
        """Путь в имени файла — попытка обхода каталога.

        Клиент управляет полем filename целиком, поэтому оттуда берётся
        только базовое имя, и оба вида разделителей обрабатываются
        одинаково независимо от ОС сервера.
        """
        assert validate_upload_name("../../etc/passwd.mp4", ALLOWED) == "mp4"
        assert validate_upload_name(r"C:\Windows\evil.mkv", ALLOWED) == "mkv"

    def test_rejects_too_long_name(self) -> None:
        with pytest.raises(ValidationError, match="длиннее"):
            validate_upload_name("x" * 300 + ".mp4", ALLOWED)


class TestValidateMedia:
    def _info(self, **overrides: object) -> MediaInfo:
        base: dict[str, object] = {
            "duration_sec": 600.0,
            "video_codec": "h264",
            "audio_codec": "aac",
            "container": "mp4",
            "size_bytes": 1024,
        }
        base.update(overrides)
        return MediaInfo(**base)  # type: ignore[arg-type]

    def test_accepts_valid(self) -> None:
        validate_media(
            self._info(),
            allowed_containers=ALLOWED,
            max_duration_sec=21600,
            max_bytes=1024**3,
        )

    def test_mov_passes_because_family_matches(self) -> None:
        """Ключевой случай: .mov имеет семейство mp4 и обязан проходить."""
        validate_media(
            self._info(container="mp4"),
            allowed_containers=frozenset({"mov"}),
            max_duration_sec=21600,
            max_bytes=1024**3,
        )

    def test_rejects_foreign_container(self) -> None:
        with pytest.raises(UnsupportedMediaError, match="avi"):
            validate_media(
                self._info(container="avi"),
                allowed_containers=ALLOWED,
                max_duration_sec=21600,
                max_bytes=1024**3,
            )

    def test_rejects_too_long(self) -> None:
        with pytest.raises(ValidationError, match="Длительность"):
            validate_media(
                self._info(duration_sec=30000.0),
                allowed_containers=ALLOWED,
                max_duration_sec=21600,
                max_bytes=1024**3,
            )

    def test_rejects_too_large(self) -> None:
        with pytest.raises(FileTooLargeError):
            validate_media(
                self._info(size_bytes=10 * 1024**3),
                allowed_containers=ALLOWED,
                max_duration_sec=21600,
                max_bytes=1024**3,
            )

    def test_rejects_file_without_streams(self) -> None:
        with pytest.raises(UnsupportedMediaError, match="потока"):
            validate_media(
                self._info(video_codec=None, audio_codec=None),
                allowed_containers=ALLOWED,
                max_duration_sec=21600,
                max_bytes=1024**3,
            )
