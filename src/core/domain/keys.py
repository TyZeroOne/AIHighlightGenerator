"""Раскладка ключей в объектном хранилище.

Ключ — это не «путь к файлу», а адрес объекта. Раскладка вынесена в отдельный
модуль по двум причинам:

1. **Детерминированность.** Один и тот же артефакт всегда получает один и тот же
   ключ. Повторный запуск этапа перезаписывает объект, а не плодит копии —
   на этом держится идемпотентность конвейера (док 04, §4.6).
2. **Массовые операции.** Общий префикс у всех артефактов джоба позволяет
   удалить их одним запросом к хранилищу, не перечисляя по одному.

Функции чистые: строят строку и ничего не знают ни про диск, ни про S3.
"""

from __future__ import annotations

from uuid import UUID

__all__ = [
    "audio_key",
    "clip_key",
    "job_prefix",
    "keyframe_key",
    "manifest_key",
    "source_key",
    "subtitle_key",
    "thumbnail_key",
    "upload_part_key",
    "upload_prefix",
    "video_prefix",
]


def video_prefix(video_id: UUID) -> str:
    return f"videos/{video_id}"


def job_prefix(job_id: UUID) -> str:
    return f"jobs/{job_id}"


def source_key(video_id: UUID, extension: str) -> str:
    """Исходное видео. Расширение сохраняем: по нему ffmpeg выбирает демуксер."""
    ext = extension.lstrip(".").lower()
    return f"{video_prefix(video_id)}/source.{ext}"


def audio_key(job_id: UUID) -> str:
    return f"{job_prefix(job_id)}/audio.wav"


def keyframe_key(job_id: UUID, scene_index: int) -> str:
    # Нули в номере — чтобы лексикографическая сортировка в консоли хранилища
    # совпадала с числовой. Иначе кадр 10 окажется между 1 и 2.
    return f"{job_prefix(job_id)}/keyframes/{scene_index:06d}.jpg"


def clip_key(job_id: UUID, rank: int, *, vertical: bool = False) -> str:
    suffix = "v" if vertical else "h"
    return f"{job_prefix(job_id)}/clips/{rank:03d}_{suffix}.mp4"


def subtitle_key(job_id: UUID, rank: int, fmt: str) -> str:
    return f"{job_prefix(job_id)}/clips/{rank:03d}.{fmt}"


def thumbnail_key(job_id: UUID, rank: int) -> str:
    return f"{job_prefix(job_id)}/clips/{rank:03d}.jpg"


def manifest_key(job_id: UUID) -> str:
    return f"{job_prefix(job_id)}/manifest.json"


def upload_prefix(upload_id: UUID) -> str:
    """Незавершённые чанковые загрузки живут отдельно от готовых видео,
    чтобы их можно было чистить по TTL, не задевая ничего полезного."""
    return f"uploads/{upload_id}"


def upload_part_key(upload_id: UUID, part_number: int) -> str:
    return f"{upload_prefix(upload_id)}/{part_number:06d}.part"
