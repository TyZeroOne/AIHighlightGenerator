"""Сценарии приложения — то, что система умеет делать.

Слой между доменом и доставкой. Знает домен и порты; не знает ни HTTP,
ни Celery, ни SQLAlchemy, ни ffmpeg. Благодаря этому один и тот же сценарий
вызывается из FastAPI-роутера, Celery-задачи и CLI без дублирования логики.
"""

from core.usecases.chunked_upload import (
    ChunkedUploadLimits,
    CompleteChunkedUpload,
    InitChunkedUpload,
    UploadPart,
)
from core.usecases.upload_video import (
    DeleteVideo,
    GetVideo,
    ListVideos,
    RegisterLocalVideo,
    StoredUpload,
    UploadVideo,
    VideoLimits,
    stream_to_file,
)

__all__ = [
    "ChunkedUploadLimits",
    "CompleteChunkedUpload",
    "DeleteVideo",
    "GetVideo",
    "InitChunkedUpload",
    "ListVideos",
    "RegisterLocalVideo",
    "StoredUpload",
    "UploadPart",
    "UploadVideo",
    "VideoLimits",
    "stream_to_file",
]
