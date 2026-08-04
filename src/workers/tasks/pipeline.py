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
from core.infra.metrics import JOBS_FINISHED
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
    "finalize",
    "generate_metadata",
    "llm_rank",
    "render_clips",
    "score_scenes",
    "select_highlights",
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


class ScoreScenesTask(PipelineTask):
    stage = JobStage.SCORE_SCENES


class SelectHighlightsTask(PipelineTask):
    stage = JobStage.SELECT_HIGHLIGHTS


class LLMRankTask(PipelineTask):
    stage = JobStage.LLM_RANK
    # Ретраи здесь не нужны: сценарий сам переживает недоступность модели
    # и переводит джоб на эвристику. Повтор задачи означал бы вторую оплату
    # тех же токенов ради того же результата.
    max_retries = 0


class FinalizeTask(PipelineTask):
    stage = JobStage.FINALIZE


class GenerateMetadataTask(PipelineTask):
    stage = JobStage.GENERATE_METADATA
    # Ретраи не нужны по той же причине, что у ранжирования: сценарий сам
    # переживает недоступность модели, а повтор означал бы вторую оплату
    # тех же токенов.
    max_retries = 0


class RenderClipsTask(PipelineTask):
    stage = JobStage.RENDER_CLIPS
    # Десять клипов по полторы минуты на медленном пресете — это десятки
    # минут, а на двухчасовом исходнике к ним добавляется перемотка.
    # Часового лимита мало.
    time_limit = 4 * 3600
    soft_time_limit = 4 * 3600 - 300


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


@celery_app.task(
    base=LLMRankTask,
    bind=True,
    name="workers.tasks.pipeline.llm_rank",
)
def llm_rank(self: LLMRankTask, job_id: str) -> dict[str, Any]:
    """Ранжирование сцен языковой моделью.

    Единственный этап конвейера, зависящий от внешнего платного сервиса.
    Его недоступность не проваливает джоб: сцены остаются с эвристикой,
    джоб получает ``partial`` (док 06, §6.10).
    """
    result = run_async(get_container().rank_scenes.execute(UUID(job_id)))
    return {
        "job_id": str(result.job_id),
        "scenes_total": result.scenes_total,
        "scenes_sent": result.scenes_sent,
        "scenes_scored": result.scenes_scored,
        "batches": result.batches,
        "duration_sec": result.duration_sec,
        "cost": dict(result.cost),
        "skipped": result.skipped,
        "reused": result.reused,
        "degraded": result.degraded,
    }


@celery_app.task(
    base=ScoreScenesTask,
    bind=True,
    name="workers.tasks.pipeline.score_scenes",
)
def score_scenes(self: ScoreScenesTask, job_id: str) -> dict[str, Any]:
    """Эвристический скор сцен.

    Идёт до LLM-этапа: модель получает посчитанные здесь сигналы,
    а не сырые признаки (док 06, §6.4).
    """
    result = run_async(get_container().score_scenes.execute(UUID(job_id)))
    return {
        "job_id": str(result.job_id),
        "scenes": result.scenes_scored,
        "mean_score": result.mean_score,
        "max_score": result.max_score,
        "duration_sec": result.duration_sec,
        "skipped": result.skipped,
        "reused": result.reused,
    }


@celery_app.task(
    base=RenderClipsTask,
    bind=True,
    name="workers.tasks.pipeline.render_clips",
)
def render_clips(self: RenderClipsTask, job_id: str) -> dict[str, Any]:
    """Нарезка отобранных моментов в файлы.

    Один из трёх этапов, которые валят джоб (док 04, §4.5) — но только
    если не собрался **ни один** клип. Падение отдельного клипа помечает
    его ``failed`` и на остальные не влияет: девять готовых файлов
    полезнее, чем ноль из-за одного битого отрезка.
    """
    result = run_async(get_container().render_clips.execute(UUID(job_id)))
    return {
        "job_id": str(result.job_id),
        "clips": result.clips_total,
        "rendered": result.rendered,
        "failed": result.failed,
        "vertical": result.vertical,
        "thumbnails": result.thumbnails,
        "subtitles": result.subtitles,
        "burned": result.burned,
        "modes": dict(result.modes),
        "warnings": list(result.warnings),
        "duration_sec": result.duration_sec,
        "skipped": result.skipped,
        "reused": result.reused,
    }


@celery_app.task(
    base=FinalizeTask,
    bind=True,
    name="workers.tasks.pipeline.finalize",
)
def finalize(self: FinalizeTask, job_id: str) -> dict[str, Any]:
    """Манифест и итоговый статус джоба.

    Здесь джоб перестаёт быть «выполняющимся»: ``succeeded``, если всё
    прошло, ``partial`` при деградации по пути, ``failed`` если ни одного
    готового клипа не осталось.
    """
    result = run_async(get_container().finalize_job.execute(UUID(job_id)))
    # Счётчик здесь, а не в сценарии: сценарий описывает предметную
    # область и не должен знать, что кто-то ведёт по нему статистику.
    if not result.reused:
        JOBS_FINISHED.labels(status=result.status.value).inc()
    return {
        "job_id": str(result.job_id),
        "status": result.status.value,
        "clips_ready": result.clips_ready,
        "clips_failed": result.clips_failed,
        "manifest_key": result.manifest_key,
        "duration_sec": result.duration_sec,
        "reused": result.reused,
    }


@celery_app.task(
    base=GenerateMetadataTask,
    bind=True,
    name="workers.tasks.pipeline.generate_metadata",
)
def generate_metadata(self: GenerateMetadataTask, job_id: str) -> dict[str, Any]:
    """Заголовки, описания и хештеги по клипам.

    Недоступность модели не проваливает джоб: клипы остаются без
    подписей, этап помечается пройденным с признаком деградации.
    """
    result = run_async(get_container().generate_metadata.execute(UUID(job_id)))
    return {
        "job_id": str(result.job_id),
        "clips": result.clips_total,
        "annotated": result.annotated,
        "failed": result.failed,
        "cost": dict(result.cost),
        "duration_sec": result.duration_sec,
        "skipped": result.skipped,
        "reused": result.reused,
        "degraded": result.degraded,
    }


@celery_app.task(
    base=SelectHighlightsTask,
    bind=True,
    name="workers.tasks.pipeline.select_highlights",
)
def select_highlights(self: SelectHighlightsTask, job_id: str) -> dict[str, Any]:
    """Отбор моментов и запись клипов.

    Здесь же эвристика сводится с оценкой модели: до этого этапа вердиктов
    в базе ещё нет, а после — уже поздно, отбор идёт по итоговому скору.
    """
    result = run_async(get_container().select_highlights.execute(UUID(job_id)))
    return {
        "job_id": str(result.job_id),
        "clips": result.clips,
        "scenes": result.scenes,
        "top_score": result.top_score,
        "total_duration_sec": result.total_duration_sec,
        "duration_sec": result.duration_sec,
        "skipped": result.skipped,
        "reused": result.reused,
    }
