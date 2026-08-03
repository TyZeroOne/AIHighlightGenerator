"""Порты — интерфейсы, через которые ядро говорит с внешним миром.

Все объявлены как ``typing.Protocol``, а не как абстрактные базовые классы.
Разница принципиальная: при ABC адаптер обязан наследоваться от нашего класса,
то есть импортировать ядро; при Protocol соответствие проверяется структурно,
и адаптер может вообще не знать о существовании порта.

Именно поэтому сторонний плагин из чужого пакета сможет реализовать
``FeatureAnalyzer``, ничего от нас не наследуя.
"""

from core.ports.analysis import FeatureAnalyzer, ScoringStrategy
from core.ports.llm import LLMProvider
from core.ports.media import AudioExtractor, ClipRenderer, SceneDetector, SubtitleWriter
from core.ports.repository import (
    AssetRepositoryPort,
    ClipRepository,
    JobRepository,
    SceneRepository,
    UnitOfWork,
    UnitOfWorkFactory,
    UploadSessionRepository,
    UserRepository,
    VideoRepository,
)
from core.ports.source import MediaProbe, SourceProvider
from core.ports.speech import Transcriber
from core.ports.storage import ObjectStorage

__all__ = [
    "AssetRepositoryPort",
    "AudioExtractor",
    "ClipRenderer",
    "ClipRepository",
    "FeatureAnalyzer",
    "JobRepository",
    "LLMProvider",
    "MediaProbe",
    "ObjectStorage",
    "SceneDetector",
    "SceneRepository",
    "ScoringStrategy",
    "SourceProvider",
    "SubtitleWriter",
    "Transcriber",
    "UnitOfWork",
    "UnitOfWorkFactory",
    "UploadSessionRepository",
    "UserRepository",
    "VideoRepository",
]
