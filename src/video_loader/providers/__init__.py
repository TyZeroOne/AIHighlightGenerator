"""Провайдеры источников видео.

Импорт этого модуля исполняет декораторы ``@register_source`` и наполняет
реестр — поэтому имя пакета перечислено в ``core.registry._BUILTIN_MODULES``.

Здесь же видно, куда добавятся YouTube, Twitch и Kick: рядом, отдельными
файлами, без единой правки в ядре и конвейере.
"""

from video_loader.providers.local import LocalFileProvider
from video_loader.providers.upload import UploadProvider, parse_upload_uri

__all__ = ["LocalFileProvider", "UploadProvider", "parse_upload_uri"]
