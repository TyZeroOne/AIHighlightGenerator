"""Задачи конвейера обработки.

Задача здесь — тонкая обёртка: развернуть аргументы, вызвать сценарий,
вернуть сериализуемый результат. Вся логика живёт в ``core/usecases``,
и это принципиально: тот же сценарий вызывают CLI и (позже) API, и ни один
из них не должен зависеть от Celery.

Аргумент у задач ровно один — ``job_id``. Состояние читается из БД и туда же
пишется (док 04, §4.2): брокер не должен превращаться в хранилище, а сообщение
в Redis — переживать перезапуск с половиной состояния внутри.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from core.container import get_container
from core.domain.enums import JobStage
from core.logging import get_logger
from core.usecases.analyze import AnalysisResult
from workers.app import celery_app
from workers.base import PipelineTask
from workers.runtime import run_async

__all__ = [
    "analyze_audio",
    "analyze_speech",
    "analyze_vision",
    "detect_scenes",
    "extract_audio",
    "transcribe",
]

log = get_logger(__name__)


class DetectScenesTask(PipelineTask):
    stage = JobStage.DETECT_SCENES


class ExtractAudioTask(PipelineTask):
    stage = JobStage.EXTRACT_AUDIO


class AnalyzeAudioTask(PipelineTask):
    stage = JobStage.ANALYZE_AUDIO


class AnalyzeSpeechTask(PipelineTask):
    stage = JobStage.ANALYZE_SPEECH


class AnalyzeVisionTask(PipelineTask):
    stage = JobStage.ANALYZE_VISION
    # Разбор кадров двухчасового фильма — самая долгая работа после
    # распознавания; часового лимита ей мало.
    time_limit = 4 * 3600
    soft_time_limit = 4 * 3600 - 300


class TranscribeTask(PipelineTask):
    stage = JobStage.TRANSCRIBE
    # Модель грузится минуты, а распознавание часового файла на CPU идёт
    # десятки минут. Общий часовой лимит для этой задачи слишком тесен.
    time_limit = 4 * 3600
    soft_time_limit = 4 * 3600 - 300


@celery_app.task(
    base=DetectScenesTask,
    bind=True,
    name="workers.tasks.pipeline.detect_scenes",
)
def detect_scenes(self: DetectScenesTask, job_id: str) -> dict[str, Any]:
    """Разбить видео джоба на сцены.

    Идемпотентна: повторный запуск на уже пройденном этапе не пересчитывает
    сцены, а возвращает их количество (``reused: true``). Это не оптимизация,
    а требование ``acks_late``: убитый воркер вернёт сообщение в очередь,
    и задача выполнится второй раз при живом результате первой.
    """
    result = run_async(get_container().detect_scenes.execute(UUID(job_id)))
    return {
        "job_id": str(result.job_id),
        "scenes": result.scene_count,
        "keyframes": result.keyframe_count,
        "duration_sec": result.duration_sec,
        "reused": result.reused,
    }


@celery_app.task(
    base=ExtractAudioTask,
    bind=True,
    name="workers.tasks.pipeline.extract_audio",
)
def extract_audio(self: ExtractAudioTask, job_id: str) -> dict[str, Any]:
    """Извлечь аудиодорожку в WAV 16 кГц и положить в хранилище.

    Видео без звука не считается ошибкой: этап помечается ``skipped``,
    и конвейер идёт дальше по одному видеоряду (док 04, §4.5).
    """
    result = run_async(get_container().extract_audio.execute(UUID(job_id)))
    return {
        "job_id": str(result.job_id),
        "storage_key": result.storage_key,
        "duration_sec": result.duration_sec,
        "skipped": result.skipped,
        "reused": result.reused,
    }


@celery_app.task(
    base=TranscribeTask,
    bind=True,
    name="workers.tasks.pipeline.transcribe",
)
def transcribe(self: TranscribeTask, job_id: str) -> dict[str, Any]:
    """Распознать речь и привязать текст к сценам.

    Уходит в очередь ``gpu``. Если GPU в развёртывании нет, ту же очередь
    обслуживает CPU-воркер, а модель работает в int8 — решает конфиг,
    а не ветвление в коде (док 04, §4.3).
    """
    result = run_async(get_container().transcribe.execute(UUID(job_id)))
    return {
        "job_id": str(result.job_id),
        "segments": result.segment_count,
        "words": result.word_count,
        "language": result.language,
        "scenes_with_text": result.scenes_with_text,
        "duration_sec": result.duration_sec,
        "skipped": result.skipped,
        "reused": result.reused,
    }


def _analysis_payload(result: AnalysisResult) -> dict[str, Any]:
    return {
        "job_id": str(result.job_id),
        "stage": result.stage.value,
        "analyzers": list(result.analyzers),
        "failed": dict(result.failed),
        "features": result.feature_count,
        "scenes": result.scenes_with_features,
        "duration_sec": result.duration_sec,
        "skipped": result.skipped,
        "reused": result.reused,
    }


@celery_app.task(
    base=AnalyzeAudioTask,
    bind=True,
    name="workers.tasks.pipeline.analyze_audio",
)
def analyze_audio(self: AnalyzeAudioTask, job_id: str) -> dict[str, Any]:
    """Аудиопризнаки: громкость, паузы, события.

    Падение одного анализатора не роняет этап: джоб становится ``partial``,
    остальные признаки остаются на месте (док 04, §4.5).
    """
    return _analysis_payload(run_async(get_container().analyze_audio.execute(UUID(job_id))))


@celery_app.task(
    base=AnalyzeSpeechTask,
    bind=True,
    name="workers.tasks.pipeline.analyze_speech",
)
def analyze_speech(self: AnalyzeSpeechTask, job_id: str) -> dict[str, Any]:
    """Речевые признаки: темп, плотность, эмоция."""
    return _analysis_payload(run_async(get_container().analyze_speech.execute(UUID(job_id))))


@celery_app.task(
    base=AnalyzeVisionTask,
    bind=True,
    name="workers.tasks.pipeline.analyze_vision",
)
def analyze_vision(self: AnalyzeVisionTask, job_id: str) -> dict[str, Any]:
    """Видеопризнаки: движение, склейки, лица, динамика кадра."""
    return _analysis_payload(run_async(get_container().analyze_vision.execute(UUID(job_id))))
