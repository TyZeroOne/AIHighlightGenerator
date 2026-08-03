"""Проверка настоящего ffprobe на настоящих файлах.

Файлы генерируются FFmpeg прямо здесь, с **заранее известными** свойствами:
длительность 3 с, разрешение 320×240, 25 fps. Проверяем не «примерно похоже»,
а точные значения — потому что знаем, что записывали.

Хранить готовые видео в репозитории не нужно и вредно: они весят,
не читаются в diff и устаревают. Генерация занимает доли секунды.

Тесты пропускаются, если ffmpeg не установлен на машине разработчика:
в Docker он есть всегда, локально — не у всех.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from core.domain.errors import SourceError, UnsupportedMediaError
from video_loader.probe import FFprobe

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

pytestmark.append(
    pytest.mark.skipif(
        FFMPEG is None or FFPROBE is None,
        reason="ffmpeg/ffprobe не установлены (в Docker-образе они есть)",
    )
)


def _generate(path: Path, *, seconds: int = 3, with_audio: bool = True, fps: int = 25) -> Path:
    """Сгенерировать тестовое видео синтезаторами FFmpeg.

    ``testsrc2`` — цветная тестовая таблица с движением, ``sine`` — тон 440 Гц.
    Никаких внешних файлов, полностью воспроизводимо.
    """
    args = [
        str(FFMPEG),
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=320x240:rate={fps}:duration={seconds}",
    ]
    if with_audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if with_audio:
        args += ["-c:a", "aac", "-shortest"]
    args += [str(path)]

    subprocess.run(args, check=True, capture_output=True, timeout=120)
    return path


@pytest.fixture(scope="module")
def media_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("media")


class TestRealProbe:
    async def test_reads_mp4(self, media_dir: Path) -> None:
        path = _generate(media_dir / "sample.mp4")

        info = await FFprobe(str(FFPROBE)).probe(path)

        # Точные значения: мы сами их и задали при генерации.
        assert info.duration_sec == pytest.approx(3.0, abs=0.15)
        assert (info.width, info.height) == (320, 240)
        assert info.fps == pytest.approx(25.0, abs=0.01)
        assert info.video_codec == "h264"
        assert info.audio_codec == "aac"
        assert info.container == "mp4"
        assert info.has_audio and info.has_video
        assert info.size_bytes == path.stat().st_size

    async def test_reads_mkv(self, media_dir: Path) -> None:
        """mkv должен нормализоваться в семейство ``mkv``."""
        path = _generate(media_dir / "sample.mkv")

        info = await FFprobe(str(FFPROBE)).probe(path)

        assert info.container == "mkv"
        assert info.duration_sec == pytest.approx(3.0, abs=0.15)

    async def test_reads_mov(self, media_dir: Path) -> None:
        """Ключевой случай: .mov опознаётся как семейство mp4.

        Если бы мы сравнивали расширения, честный .mov был бы отвергнут.
        """
        path = _generate(media_dir / "sample.mov")

        info = await FFprobe(str(FFPROBE)).probe(path)

        assert info.container == "mp4"
        assert info.has_video

    async def test_video_without_audio(self, media_dir: Path) -> None:
        """Видео без звука — штатный случай, а не ошибка.

        Конвейер должен пойти по нему только видеопризнаками (док 04, §4.5).
        """
        path = _generate(media_dir / "silent.mp4", with_audio=False)

        info = await FFprobe(str(FFPROBE)).probe(path)

        assert info.has_video
        assert not info.has_audio
        assert info.audio_codec is None

    async def test_variable_fps_is_read(self, media_dir: Path) -> None:
        path = _generate(media_dir / "fps60.mp4", fps=60)

        info = await FFprobe(str(FFPROBE)).probe(path)

        assert info.fps == pytest.approx(60.0, abs=0.01)


class TestRealProbeFailures:
    async def test_text_file_renamed_to_mp4_is_rejected_as_415(self, media_dir: Path) -> None:
        """Тот самый случай из DoD этапа.

        Файл называется .mp4, но внутри текст. Расширение здесь ничего
        не значит — отличить может только попытка прочитать содержимое.

        Код именно 415, а не 400: запрос корректен, не подходит **тип
        содержимого**. Первая реализация отдавала 400, и это заметила
        сквозная проверка на живом стеке, а не тест с фейковым ffprobe.
        """
        fake = media_dir / "definitely_video.mp4"
        fake.write_text("Это обычный текстовый файл, а не видео.", encoding="utf-8")

        with pytest.raises(UnsupportedMediaError) as exc_info:
            await FFprobe(str(FFPROBE)).probe(fake)

        assert exc_info.value.http_status == 415

    async def test_error_message_does_not_leak_server_paths(self, media_dir: Path) -> None:
        """Диагностика ffprobe содержит полный путь к файлу на сервере.

        Отдавать его клиенту нельзя: это раскрывает раскладку файловой
        системы и имена внутренних каталогов.
        """
        fake = media_dir / "leaky.mp4"
        fake.write_text("not a video", encoding="utf-8")

        with pytest.raises(UnsupportedMediaError) as exc_info:
            await FFprobe(str(FFPROBE)).probe(fake)

        message = exc_info.value.message
        assert str(media_dir) not in message
        assert "leaky.mp4" not in message
        # Содержательная причина в сообщении быть обязана: без неё разбор
        # инцидента начинается с нуля. А вот маркер «<файл>» проверять нельзя:
        # он появляется, только если ffprobe вообще упомянул путь, а разные
        # версии ведут себя по-разному — ffmpeg 7.1 на этом файле пишет
        # «moov atom not found» без всякого имени.
        assert len(message) > len("Файл не удалось прочитать как медиа: ")

    async def test_truncated_file_is_rejected(self, media_dir: Path) -> None:
        """Оборванная загрузка даёт файл с валидным началом и без конца."""
        source = _generate(media_dir / "full.mp4")
        broken = media_dir / "truncated.mp4"
        broken.write_bytes(source.read_bytes()[:200])

        with pytest.raises(UnsupportedMediaError):
            await FFprobe(str(FFPROBE)).probe(broken)

    async def test_missing_file(self, media_dir: Path) -> None:
        with pytest.raises(SourceError, match="не найден"):
            await FFprobe(str(FFPROBE)).probe(media_dir / "nope.mp4")

    async def test_missing_binary_reports_clearly(self, media_dir: Path) -> None:
        """Отсутствие ffprobe — ошибка окружения, а не файла.

        Она должна быть retriable-классом с внятным текстом, иначе разбор
        инцидента начнётся с подозрений на битое видео.
        """
        from core.domain.errors import TransientError

        path = _generate(media_dir / "ok.mp4")

        with pytest.raises(TransientError, match="ffprobe не найден"):
            await FFprobe("definitely-not-a-real-binary").probe(path)

    async def test_timeout_kills_the_process(self, media_dir: Path) -> None:
        """Зависший ffprobe должен быть убит, а не оставлен висеть.

        Иначе процессы копятся, удерживая дескрипторы и место на диске.
        """
        path = _generate(media_dir / "slow.mp4")
        probe = FFprobe(str(FFPROBE), timeout_sec=0.001)

        with pytest.raises(SourceError, match="не уложился"):
            await probe.probe(path)

        # Даём циклу событий шанс дожать убитый процесс: если бы он остался,
        # сюда прилетело бы предупреждение о брошенном subprocess.
        await asyncio.sleep(0.05)
