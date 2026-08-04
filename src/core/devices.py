"""На чём считать: процессор или видеокарта.

Отдельный модуль в ``core``, а не функция в одном из адаптеров, по прозаичной
причине: этот вопрос задают и `audio.events`, и `speech.emotion`, и
`vision.faces`, а правило архитектуры запрещает адаптерам импортировать
друг друга. Общее знание живёт в ``core`` — там ему и место.

Импорт torch отложен внутрь функции: ``core`` обязан оставаться пригодным
для процесса API, где никакого torch нет и не будет.

Своя функция для CTranslate2 (`speech.whisper.resolve_device`) существует
рядом и это не дублирование по недосмотру: у неё другой источник правды —
про наличие CUDA спрашивается сам CTranslate2, потому что видеокарта
может быть видна torch и не поддерживаться его сборкой, и наоборот.
"""

from __future__ import annotations

from core.logging import get_logger

__all__ = ["resolve_torch_device"]

log = get_logger(__name__)


def resolve_torch_device(requested: str) -> str:
    """``auto`` → ``cuda``, если видеокарта действительно доступна torch.

    Явные ``cpu``/``cuda`` возвращаются как есть: если человек указал
    устройство руками, спорить с ним и молча откатываться — худшее,
    что можно сделать. Пусть падает громко.
    """
    if requested in {"cpu", "cuda"}:
        return requested

    try:
        import torch
    except ImportError:  # pragma: no cover — зависит от окружения
        return "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"
