"""Как видеоанализатор получает кадры и настройки.

Анализаторы создаёт реестр вызовом ``cls()`` — без аргументов. Это цена
плагинной модели: сторонний плагин из чужого пакета невозможно провести
через наш композиционный корень, а значит, и настройки, и общие ресурсы
он должен уметь получить сам.

Отсюда две функции этого модуля. Они же — единственное место в пакете
``vision``, которое знает про конфиг; сами анализаторы остаются чистыми
и принимают всё через конструктор, что и позволяет тестировать их
на выдуманных кадрах.
"""

from __future__ import annotations

from core.config import get_settings
from core.domain.analysis import AnalysisContext
from vision.frames import DEFAULT_MAX_SIDE, DEFAULT_SAMPLE_FPS, FrameSource, shared_source

__all__ = ["sample_settings", "source_for"]


def sample_settings() -> tuple[float, int]:
    """Частота сэмплирования и размер кадра из конфига."""
    settings = get_settings()
    return (
        getattr(settings, "vision_sample_fps", DEFAULT_SAMPLE_FPS),
        getattr(settings, "vision_max_side", DEFAULT_MAX_SIDE),
    )


def source_for(context: AnalysisContext) -> FrameSource | None:
    """Общий источник кадров для видеофайла этого джоба.

    ``None`` означает, что видео не скачивалось: у этапа аудио его нет,
    и анализатор просто не выдаст признаков.
    """
    if context.media is None:
        return None

    sample_fps, max_side = sample_settings()
    return shared_source(context.media.path, sample_fps=sample_fps, max_side=max_side)
