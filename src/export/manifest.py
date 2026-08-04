"""Манифест джоба: один файл, описывающий весь результат.

## Зачем он нужен, если есть API

API отвечает на вопросы «покажи клипы», «дай файл». Манифест отвечает
на другой вопрос: «что вообще получилось». Разница практическая.

Результат джоба живёт дольше самого джоба. Клипы скачивают, кладут
в папку, возвращаются к ним через месяц — и к этому моменту база может
быть очищена по retention, а сервис переехать. Манифест едет вместе
с файлами и остаётся самодостаточным описанием: что это за видео, какие
моменты выбраны, почему, с какими заголовками и где чей файл.

Второе применение — передача наружу. Скрипту, заливающему клипы
на площадку, нужен один JSON, а не десять запросов к API.

## Почему это чистая функция

Сборка манифеста ничего не читает и никуда не пишет: на вход — доменные
объекты, на выход — словарь. Отсюда тестируемость без базы и хранилища,
а главное — невозможность случайно превратить «описание результата»
в ещё один способ ходить в БД.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Final

from core.domain.entities import Asset, Clip, Job, Video
from core.domain.timecode import format_timecode

__all__ = ["MANIFEST_VERSION", "JsonManifestBuilder", "build_manifest"]

#: Версия формата. Первое, что читает чужой парсер, и единственное, что
#: позволит через год отличить старый манифест от нового, не угадывая.
MANIFEST_VERSION: Final = 1


def build_manifest(
    job: Job,
    video: Video,
    clips: Sequence[Clip],
    assets: Mapping[Any, Sequence[Asset]],
    *,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Собрать описание результата джоба.

    ``assets`` — артефакты по идентификатору клипа. Передаются снаружи,
    а не читаются здесь: функция остаётся чистой, а вызывающий сам решает,
    одним запросом он их достал или десятью.
    """
    return {
        "manifest_version": MANIFEST_VERSION,
        "generated_at": (generated_at or datetime.now(UTC)).isoformat(),
        "job": {
            "id": str(job.id),
            "status": job.status.value,
            "progress": job.progress,
            "params": dict(job.params),
            "cost": dict(job.cost),
        },
        "video": {
            "id": str(video.id),
            "source_uri": video.source_uri,
            "filename": video.original_filename,
            "duration_sec": round(video.info.duration_sec, 3) if video.info else None,
            "width": video.info.width if video.info else None,
            "height": video.info.height if video.info else None,
        },
        "clips": [_clip_entry(clip, assets.get(clip.id, ())) for clip in clips],
    }


class JsonManifestBuilder:
    """Реализация порта ``ManifestBuilder``.

    Класс поверх функции нужен только затем, чтобы контейнер связал порт
    с реализацией: сама сборка остаётся чистой функцией, и тесты зовут
    именно её.
    """

    def build(
        self,
        job: Job,
        video: Video,
        clips: Sequence[Clip],
        assets: Mapping[Any, Sequence[Asset]],
    ) -> dict[str, Any]:
        return build_manifest(job, video, clips, assets)


def _clip_entry(clip: Clip, assets: Sequence[Asset]) -> dict[str, Any]:
    metadata = clip.metadata
    return {
        "id": str(clip.id),
        "rank": clip.rank,
        "status": clip.status.value,
        # Время и строкой, и числом — по той же причине, что и в API:
        # человеку удобен тайм-код, коду секунды, и заставлять любую
        # из сторон парсить чужой формат незачем.
        "start": format_timecode(clip.range.start),
        "end": format_timecode(clip.range.end),
        "start_sec": round(clip.range.start, 3),
        "end_sec": round(clip.range.end, 3),
        "duration_sec": round(clip.range.duration, 3),
        "score": clip.score,
        "category": clip.category,
        "reason": clip.reason,
        "title": metadata.title if metadata else None,
        "description": metadata.description if metadata else None,
        "hashtags": list(metadata.hashtags) if metadata else [],
        "hook": metadata.hook if metadata else None,
        "error": clip.error,
        "assets": {
            asset.kind.value: {
                "id": str(asset.id),
                # Ключ хранилища, а не URL: манифест переживает переезд
                # сервиса, а подписанная ссылка живёт пятнадцать минут.
                "storage_key": asset.storage_key,
                "mime": asset.mime,
                "size_bytes": asset.size_bytes,
                "download_url": f"/api/v1/download/{asset.id}",
            }
            for asset in sorted(assets, key=lambda item: item.kind.value)
        },
    }
