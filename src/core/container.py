"""Композиционный корень.

Единственное место в проекте, где конкретные классы связываются с портами.
Всё остальное работает с интерфейсами и не знает, что за ними стоит.

Контейнер растёт вместе с проектом: на Этапе 1 здесь была только
инфраструктура, на Этапе 2 добавились хранилище, ffprobe, провайдеры
источников и сценарии загрузки. Важно, что добавление происходит здесь
и только здесь — сценарии и API не меняются.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from core import registry
from core.config import Settings, get_settings
from core.domain.analysis import ARTIFACT_AUDIO, ARTIFACT_TRANSCRIPT, ARTIFACT_VIDEO
from core.domain.enums import JobStage
from core.domain.errors import DependencyResolutionError
from core.domain.render import RenderSpec
from core.infra.db.session import create_engine, create_session_factory
from core.infra.db.uow import SqlAlchemyUnitOfWork
from core.logging import configure_logging, get_logger
from core.ports.export import ManifestBuilder
from core.ports.llm import ClipAnnotator, LLMProvider, SceneRanker
from core.ports.media import (
    AudioExtractor,
    ClipRenderer,
    CropPlanner,
    KeyframeExtractor,
    SceneDetector,
    SubtitleWriter,
)
from core.ports.repository import UnitOfWork
from core.ports.source import MediaProbe, SourceProvider
from core.ports.speech import Transcriber
from core.ports.storage import ObjectStorage
from core.usecases.analyze import AnalyzeFeatures
from core.usecases.chunked_upload import (
    ChunkedUploadLimits,
    CompleteChunkedUpload,
    InitChunkedUpload,
    UploadPart,
)
from core.usecases.detect_scenes import DetectScenes, SceneDetectionSettings
from core.usecases.finalize import FinalizeJob
from core.usecases.metadata import GenerateMetadata
from core.usecases.rank import RankScenes
from core.usecases.render import RenderClips, RenderSettings
from core.usecases.score import ScoreScenes
from core.usecases.select import SelectHighlights
from core.usecases.transcribe import ExtractAudio, Transcribe
from core.usecases.upload_video import (
    DeleteVideo,
    GetVideo,
    ListVideos,
    RegisterLocalVideo,
    UploadVideo,
    VideoLimits,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

    from highlight_detector.service import DetectorService

__all__ = ["Container", "get_container", "reset_container"]

log = get_logger(__name__)


class Container:
    """Ленивый контейнер зависимостей.

    ``cached_property`` вместо создания всего в ``__init__``: движок БД
    не нужен процессу, который выполняет только CPU-задачу, а модель Whisper
    незачем грузить в API-процессе. Ресурс создаётся при первом обращении.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        configure_logging(level=self.settings.log_level, fmt=self.settings.log_format)
        registry.discover()
        log.debug(
            "container.initialized",
            env=self.settings.app_env,
            storage=self.settings.storage_backend,
            analyzers=sorted(registry.get_analyzers()),
            sources=sorted(registry.get_source_providers()),
        )

    # ------------------------------------------------------------ БД
    @cached_property
    def engine(self) -> AsyncEngine:
        return create_engine(self.settings)

    @cached_property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        return create_session_factory(self.engine)

    def uow(self) -> UnitOfWork:
        """Фабрика Unit of Work.

        Метод, а не свойство: каждый вызов обязан давать **новый** UoW
        со своей сессией. Общий UoW на несколько запросов означал бы общую
        транзакцию — и откат одного запроса ронял бы чужие изменения.
        """
        return SqlAlchemyUnitOfWork(self.session_factory)

    # ------------------------------------------------------ хранилище
    @cached_property
    def storage(self) -> ObjectStorage:
        if self.settings.storage_backend == "s3":
            from core.infra.storage.s3 import S3Storage

            return S3Storage(
                bucket=self.settings.s3_bucket,
                endpoint_url=self.settings.s3_endpoint_url,
                public_endpoint_url=self.settings.s3_public_endpoint_url,
                access_key=self.settings.s3_access_key,
                secret_key=self.settings.s3_secret_key,
                signed_url_ttl=self.settings.s3_signed_url_ttl,
            )

        from core.infra.storage.local import LocalStorage

        return LocalStorage(self.settings.storage_local_root)

    @cached_property
    def workdir(self) -> Path:
        """Каталог для временных файлов: приём загрузок, сборка частей.

        Отдельно от хранилища: это рабочая область процесса, а не общие данные.
        """
        path = Path(self.settings.workdir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ---------------------------------------------------------- медиа
    @cached_property
    def probe(self) -> MediaProbe:
        from video_loader.probe import FFprobe

        return FFprobe(self.settings.ffprobe_bin)

    @cached_property
    def local_file_provider(self) -> SourceProvider:
        from video_loader.providers.local import LocalFileProvider

        return LocalFileProvider(self.probe, self.settings.local_source_roots)

    @cached_property
    def upload_provider(self) -> SourceProvider:
        from video_loader.providers.upload import UploadProvider

        return UploadProvider(
            self.storage,
            self.probe,
            key_resolver=self._resolve_storage_key,
            workdir=self.workdir,
        )

    @cached_property
    def source_providers(self) -> tuple[SourceProvider, ...]:
        """Все провайдеры, которых умеет собрать этот процесс.

        Порядок важен: первый, ответивший ``supports()``, и будет выбран.
        """
        return (self.upload_provider, self.local_file_provider)

    def resolve_provider(self, uri: str) -> SourceProvider:
        """Провайдер под конкретный URI.

        Сценарии получают эту функцию, а не реестр и не список: им нужно
        «дай, чем открыть вот это», а не знание о том, какие провайдеры
        вообще существуют.
        """
        for provider in self.source_providers:
            if provider.supports(uri):
                return provider
        raise DependencyResolutionError(f"Нет источника для URI: {uri!r}", uri=uri)

    @cached_property
    def scene_detector(self) -> SceneDetector:
        """Детектор сцен по конфигу.

        Обе реализации взаимозаменяемы (проверяется одним и тем же набором
        тестов через порт), поэтому выбор — строчка в ``.env``, а не ветвление
        где-то в конвейере.
        """
        if self.settings.scene_detector == "ffmpeg":
            from video_loader.scenes.ffmpeg import FFmpegSceneDetector

            return FFmpegSceneDetector(
                binary=self.settings.ffmpeg_bin,
                threshold=self.settings.scene_threshold,
                min_len_sec=self.settings.scene_min_len_sec,
                max_len_sec=self.settings.scene_max_len_sec,
            )

        from video_loader.scenes.pyscenedetect import PySceneDetectDetector

        return PySceneDetectDetector(
            threshold=self.settings.scene_threshold,
            min_len_sec=self.settings.scene_min_len_sec,
            max_len_sec=self.settings.scene_max_len_sec,
            downscale=self.settings.scene_downscale,
        )

    @cached_property
    def audio_extractor(self) -> AudioExtractor:
        from audio.extractor import FFmpegAudioExtractor

        return FFmpegAudioExtractor(
            binary=self.settings.ffmpeg_bin,
            workdir=self.workdir,
            sample_rate=self.settings.audio_sample_rate,
            channels=self.settings.audio_channels,
        )

    @cached_property
    def keyframe_extractor(self) -> KeyframeExtractor:
        from video_loader.scenes.keyframes import FFmpegKeyframeExtractor

        return FFmpegKeyframeExtractor(
            binary=self.settings.ffmpeg_bin,
            width=self.settings.scene_keyframe_width,
            max_parallel=self.settings.scene_keyframe_workers,
        )

    async def _resolve_storage_key(self, video_id: UUID) -> str:
        """`video_id` → ключ в хранилище.

        Провайдеру нужен ключ, но ходить в БД он не должен — иначе адаптер
        медиа начал бы зависеть от адаптера БД. Мост между ними строит
        контейнер: это ровно его работа.
        """
        from core.domain.errors import NotFoundError

        async with self.uow() as uow:
            video = await uow.videos.get(video_id)
        if video is None:
            raise NotFoundError("Видео не найдено", video_id=str(video_id))
        return video.storage_key

    # ------------------------------------------------------- лимиты
    @cached_property
    def video_limits(self) -> VideoLimits:
        return VideoLimits(
            allowed_containers=self.settings.allowed_containers_set,
            max_duration_sec=self.settings.max_duration_sec,
            max_bytes=self.settings.max_upload_bytes,
        )

    @cached_property
    def chunked_limits(self) -> ChunkedUploadLimits:
        return ChunkedUploadLimits(
            max_bytes=self.settings.max_upload_bytes,
            allowed_containers=self.settings.allowed_containers_set,
            chunk_size=self.settings.upload_chunk_size,
        )

    # ------------------------------------------------------ сценарии
    @cached_property
    def upload_video(self) -> UploadVideo:
        return UploadVideo(
            uow_factory=self.uow,
            storage=self.storage,
            probe=self.probe,
            limits=self.video_limits,
        )

    @cached_property
    def register_local_video(self) -> RegisterLocalVideo:
        return RegisterLocalVideo(
            uow_factory=self.uow,
            provider=self.local_file_provider,
            limits=self.video_limits,
        )

    @cached_property
    def get_video(self) -> GetVideo:
        return GetVideo(uow_factory=self.uow)

    @cached_property
    def list_videos(self) -> ListVideos:
        return ListVideos(uow_factory=self.uow)

    @cached_property
    def delete_video(self) -> DeleteVideo:
        return DeleteVideo(uow_factory=self.uow, storage=self.storage)

    @cached_property
    def init_chunked_upload(self) -> InitChunkedUpload:
        return InitChunkedUpload(uow_factory=self.uow, limits=self.chunked_limits)

    @cached_property
    def upload_part(self) -> UploadPart:
        return UploadPart(
            uow_factory=self.uow,
            storage=self.storage,
            limits=self.chunked_limits,
            workdir=self.workdir,
        )

    @cached_property
    def complete_chunked_upload(self) -> CompleteChunkedUpload:
        return CompleteChunkedUpload(
            uow_factory=self.uow,
            storage=self.storage,
            upload_video=self.upload_video,
            workdir=self.workdir,
        )

    @cached_property
    def detect_scenes(self) -> DetectScenes:
        return DetectScenes(
            uow_factory=self.uow,
            detector=self.scene_detector,
            keyframes=self.keyframe_extractor,
            storage=self.storage,
            provider_resolver=self.resolve_provider,
            workdir=self.workdir,
            settings=SceneDetectionSettings(
                keyframes_enabled=self.settings.scene_keyframes_enabled,
            ),
        )

    @cached_property
    def transcriber(self) -> Transcriber:
        """Распознавание речи.

        Модель тяжёлая (large-v3 — три гигабайта), поэтому создаётся лениво
        и живёт до конца процесса: загружать её на каждую задачу означало бы
        тратить на инициализацию больше, чем на работу.
        """
        from speech.whisper import FasterWhisperTranscriber

        return FasterWhisperTranscriber(
            model_size=self.settings.whisper_model,
            device=self.settings.device,
            compute_type=self.settings.whisper_compute_type,
            beam_size=self.settings.whisper_beam_size,
            use_vad=self.settings.whisper_vad,
            language=self.settings.whisper_language or None,
            no_speech_threshold=self.settings.whisper_no_speech_threshold,
            chunk_sec=self.settings.whisper_chunk_sec,
            chunk_overlap_sec=self.settings.whisper_chunk_overlap_sec,
            batch_size=self.settings.whisper_batch_size,
        )

    @cached_property
    def extract_audio(self) -> ExtractAudio:
        return ExtractAudio(
            uow_factory=self.uow,
            extractor=self.audio_extractor,
            storage=self.storage,
            provider_resolver=self.resolve_provider,
        )

    @cached_property
    def transcribe(self) -> Transcribe:
        return Transcribe(
            uow_factory=self.uow,
            transcriber=self.transcriber,
            storage=self.storage,
            workdir=self.workdir,
            language=self.settings.whisper_language or None,
            sample_rate=self.settings.audio_sample_rate,
            channels=self.settings.audio_channels,
        )

    # ------------------------------------------------------- анализ признаков
    def _analyzer_stage(
        self,
        *,
        stage: JobStage,
        group: str,
        needs: frozenset[str],
    ) -> AnalyzeFeatures:
        """Один класс на три этапа: отличаются они только группой и сырьём."""
        return AnalyzeFeatures(
            uow_factory=self.uow,
            storage=self.storage,
            provider_resolver=self.resolve_provider,
            workdir=self.workdir,
            stage=stage,
            group=group,
            needs=needs,
            disabled=self.settings.disabled_analyzers_set,
            sample_rate=self.settings.audio_sample_rate,
            channels=self.settings.audio_channels,
        )

    @cached_property
    def analyze_audio(self) -> AnalyzeFeatures:
        return self._analyzer_stage(
            stage=JobStage.ANALYZE_AUDIO,
            group="audio",
            needs=frozenset({ARTIFACT_AUDIO}),
        )

    @cached_property
    def analyze_speech(self) -> AnalyzeFeatures:
        return self._analyzer_stage(
            stage=JobStage.ANALYZE_SPEECH,
            group="speech",
            needs=frozenset({ARTIFACT_AUDIO, ARTIFACT_TRANSCRIPT}),
        )

    @cached_property
    def analyze_vision(self) -> AnalyzeFeatures:
        return self._analyzer_stage(
            stage=JobStage.ANALYZE_VISION,
            group="vision",
            needs=frozenset({ARTIFACT_VIDEO}),
        )

    # ------------------------------------------------------- Этап 7
    @cached_property
    def detector(self) -> DetectorService:
        """Скоринг и отбор. Один объект на оба этапа: они делят профили,
        формулы сигналов и правила нормализации, и разводить их по разным
        реализациям значило бы дублировать всё это."""
        from highlight_detector.service import DetectorService

        return DetectorService()

    @cached_property
    def score_scenes(self) -> ScoreScenes:
        return ScoreScenes(
            uow_factory=self.uow,
            scorer=self.detector,
            profile=self.settings.default_profile,
        )

    @cached_property
    def select_highlights(self) -> SelectHighlights:
        return SelectHighlights(
            uow_factory=self.uow,
            selector=self.detector,
            profile=self.settings.default_profile,
        )

    # ------------------------------------------------------- Этап 8
    @cached_property
    def render_spec(self) -> RenderSpec:
        """Параметры кодирования из конфига.

        Один объект на все виды вывода: клипы одного джоба обязаны
        звучать и выглядеть одинаково независимо от того, каким путём
        они собрались.
        """
        settings = self.settings
        return RenderSpec(
            crf=settings.render_crf,
            preset=settings.render_preset,
            video_codec=settings.render_video_codec,
            audio_codec=settings.render_audio_codec,
            audio_bitrate=settings.render_audio_bitrate,
            loudness_lufs=(
                settings.render_loudness_lufs if settings.render_normalize_audio else None
            ),
            threads=settings.render_threads,
        )

    @cached_property
    def clip_renderer(self) -> ClipRenderer:
        from video_editor.renderer import FFmpegClipRenderer

        return FFmpegClipRenderer(
            binary=self.settings.ffmpeg_bin,
            probe_binary=self.settings.ffprobe_bin,
            thumbnail_width=self.settings.render_thumbnail_width,
            allow_stream_copy=self.settings.render_allow_stream_copy,
        )

    @cached_property
    def crop_planner(self) -> CropPlanner:
        """Планировщик вертикального кадрирования.

        Живёт в ``vision``, а не в ``video_editor``: это компьютерное
        зрение, и ему нужен тот же детектор лиц, что и анализаторам
        Этапа 5. Связывает их — вот эта строчка.
        """
        from vision.reframe import FaceCropPlanner

        return FaceCropPlanner()

    @cached_property
    def subtitle_writer(self) -> SubtitleWriter:
        """Субтитры клипа. Настройки разбиения и оформления — из конфига."""
        from subtitles.ass import AssStyle
        from subtitles.cues import CueSettings
        from subtitles.writer import ClipSubtitleWriter

        settings = self.settings
        return ClipSubtitleWriter(
            settings=CueSettings(
                max_chars=settings.subtitle_max_chars,
                max_lines=settings.subtitle_max_lines,
                reading_speed=settings.subtitle_reading_speed,
                min_duration_sec=settings.subtitle_min_duration_sec,
                max_duration_sec=settings.subtitle_max_duration_sec,
                gap_sec=settings.subtitle_gap_sec,
            ),
            style=AssStyle(font=settings.subtitle_font, size=settings.subtitle_font_size),
            karaoke=settings.subtitle_karaoke,
        )

    @cached_property
    def render_clips(self) -> RenderClips:
        settings = self.settings
        return RenderClips(
            uow_factory=self.uow,
            renderer=self.clip_renderer,
            storage=self.storage,
            provider_resolver=self.resolve_provider,
            workdir=self.workdir,
            planner=self.crop_planner if settings.render_vertical else None,
            subtitles=self.subtitle_writer if settings.subtitle_formats_list else None,
            settings=RenderSettings(
                spec=self.render_spec,
                vertical=settings.render_vertical,
                thumbnails=settings.render_thumbnails,
                concurrency=settings.render_concurrency,
                subtitle_formats=settings.subtitle_formats_list,
                burn=settings.burn_subtitles,
            ),
        )

    # ------------------------------------------------------- Этап 6
    @cached_property
    def llm(self) -> LLMProvider:
        """Провайдер языковой модели — единственное место, где выбирается,
        какая именно модель отвечает за ранжирование.

        Сценарий этапа знает только порт. Замена OpenRouter на Claude —
        одна переменная окружения, ни строчки в конвейере.
        """
        settings = self.settings
        key = settings.llm_api_key
        if not key:
            raise DependencyResolutionError(
                f"Не задан ключ провайдера {settings.llm_provider!r}. "
                "Укажите OPENROUTER_API_KEY или ANTHROPIC_API_KEY, "
                "либо выключите LLM через LLM_ENABLED=false.",
                provider=settings.llm_provider,
            )

        if settings.llm_provider == "openrouter":
            from llm.openrouter import OpenRouterProvider

            return OpenRouterProvider(
                api_key=key,
                model=settings.openrouter_model,
                base_url=settings.openrouter_base_url,
                max_tokens=settings.llm_max_tokens,
                timeout_sec=settings.llm_timeout_sec,
                max_retries=settings.llm_max_retries,
            )

        raise DependencyResolutionError(
            f"Провайдер {settings.llm_provider!r} ещё не реализован",
            provider=settings.llm_provider,
        )

    def scene_ranker(self) -> SceneRanker:
        """Ранжировщик сцен. Метод, а не свойство: провайдер копит расход
        внутри себя, и каждому джобу нужен свой экземпляр — иначе стоимость
        второго джоба включила бы токены первого."""
        from llm.ranker import BatchRanker

        return BatchRanker(
            provider=self.llm,
            batch_size=self.settings.llm_batch_size,
        )

    def clip_annotator(self) -> ClipAnnotator:
        """Аннотатор клипов. Метод по той же причине, что и ранжировщик:
        расход копится в провайдере, и каждому джобу нужен свой."""
        from llm.annotator import ClipMetadataAnnotator

        return ClipMetadataAnnotator(provider=self.llm)

    @cached_property
    def generate_metadata(self) -> GenerateMetadata:
        return GenerateMetadata(
            uow_factory=self.uow,
            annotator_factory=self.clip_annotator,
            enabled=self.settings.llm_enabled,
        )

    @cached_property
    def manifest_builder(self) -> ManifestBuilder:
        from export.manifest import JsonManifestBuilder

        return JsonManifestBuilder()

    @cached_property
    def finalize_job(self) -> FinalizeJob:
        return FinalizeJob(
            uow_factory=self.uow,
            storage=self.storage,
            manifest=self.manifest_builder,
            workdir=self.workdir,
        )

    @cached_property
    def rank_scenes(self) -> RankScenes:
        return RankScenes(
            uow_factory=self.uow,
            ranker_factory=self.scene_ranker,
            enabled=self.settings.llm_enabled,
            profile=self.settings.default_profile,
        )

    # ------------------------------------------------------ жизненный цикл
    async def aclose(self) -> None:
        """Аккуратно закрыть ресурсы при остановке процесса."""
        if "engine" in self.__dict__:
            await self.engine.dispose()


_container: Container | None = None


def get_container() -> Container:
    """Синглтон контейнера на процесс."""
    global _container
    if _container is None:
        _container = Container()
    return _container


def reset_container() -> None:
    """Сброс. Нужен тестам, чтобы подменить настройки."""
    global _container
    _container = None
