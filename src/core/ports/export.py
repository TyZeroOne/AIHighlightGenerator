"""Порт сборки манифеста.

Отдельный порт ради одной функции выглядит избыточно — но правило
зависимостей не знает исключений: сценарий этапа не может импортировать
пакет ``export`` напрямую, иначе ядро начнёт зависеть от адаптера.

Выгода не только формальная. Манифест — это **формат обмена наружу**,
и он будет меняться от требований площадок и интеграций, а не от
устройства конвейера. Порт фиксирует ровно то, что нужно сценарию
(«собери описание результата»), и оставляет формату свободу.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol
from uuid import UUID

from core.domain.entities import Asset, Clip, Job, Video

__all__ = ["ManifestBuilder"]


class ManifestBuilder(Protocol):
    def build(
        self,
        job: Job,
        video: Video,
        clips: Sequence[Clip],
        assets: Mapping[UUID, Sequence[Asset]],
    ) -> Mapping[str, Any]:
        """Описание результата джоба — сериализуемый словарь.

        Артефакты передаются готовыми, а не читаются внутри: сборка
        манифеста обязана оставаться чистой функцией, иначе она незаметно
        превратится в ещё один способ ходить в базу.
        """
        ...
