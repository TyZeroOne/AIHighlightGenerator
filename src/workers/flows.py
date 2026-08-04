"""Граф конвейера и его постановка в очередь.

## Почему цепочка, а не одна большая задача

Десять этапов в одной задаче означали бы, что падение на восьмом теряет
работу первых семи, а перезапуск начинается с нуля. Цепочка даёт границы:
каждый этап — своя запись о состоянии, свой лимит времени, своя очередь
и своя возможность быть пропущенным при повторе.

## Почему параллельные ветки именно такие

После разбиения на сцены работа расходится надвое:

* **звуковая ветка** — извлечь аудио, распознать речь, посчитать
  аудио- и речевые признаки;
* **видеоветка** — движение, монтаж, лица.

Они независимы: видеоанализу не нужен звук, речевым признакам не нужны
кадры. Ветки идут ``group()``, и на двухчасовом фильме это экономит
время самой долгой из них — то есть примерно треть этапа анализа.

Внутри звуковой ветки порядок жёсткий: без аудио нечего распознавать,
а речевым признакам нужен готовый транскрипт.

## Повтор с этапа

``build_flow(job_id, from_stage=...)`` собирает **хвост** конвейера.
Это не оптимизация, а требование: перезапуск с LLM-этапа не должен
заново разбирать кадры двухчасового фильма. Состояния предыдущих этапов
чистит сценарий повтора — здесь только выбор точки входа.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final
from uuid import UUID

from celery import chain, group

from core.domain.enums import JobStage
from core.logging import get_logger
from workers.app import celery_app

if TYPE_CHECKING:
    from celery.canvas import Signature

__all__ = ["FLOW_ORDER", "CeleryDispatcher", "build_flow"]

log = get_logger(__name__)

#: Порядок этапов в конвейере. Отдельный список, а не ``list(JobStage)``:
#: enum описывает *все* этапы, включая те, что могут не участвовать
#: в конкретном графе, а здесь важна именно последовательность запуска.
FLOW_ORDER: Final[tuple[JobStage, ...]] = (
    JobStage.DETECT_SCENES,
    JobStage.EXTRACT_AUDIO,
    JobStage.TRANSCRIBE,
    JobStage.ANALYZE_AUDIO,
    JobStage.ANALYZE_SPEECH,
    JobStage.ANALYZE_VISION,
    JobStage.SCORE_SCENES,
    JobStage.LLM_RANK,
    JobStage.SELECT_HIGHLIGHTS,
    JobStage.RENDER_CLIPS,
    JobStage.GENERATE_METADATA,
    JobStage.FINALIZE,
)

#: Имя задачи по этапу. Задачи адресуются строкой, а не импортом: так
#: API-процесс ставит их в очередь, не подтягивая в себя ни torch,
#: ни OpenCV, ни Whisper — всё то, что нужно только воркеру.
_TASK_NAMES: Final[dict[JobStage, str]] = {
    stage: f"workers.tasks.pipeline.{stage.value}" for stage in FLOW_ORDER
}


def build_flow(job_id: UUID, *, from_stage: JobStage | None = None) -> Signature:
    """Сигнатура конвейера, начиная с указанного этапа.

    Ветвление сохраняется только там, где оно есть в полном графе:
    хвост, начинающийся после видеоанализа, — это простая цепочка,
    и городить в нём ``group()`` из одной задачи незачем.
    """
    stages = _tail(from_stage)
    if not stages:
        raise ValueError(f"После этапа {from_stage} нечего запускать")

    steps: list[Signature] = []
    audio_branch = [stage for stage in _AUDIO_BRANCH if stage in stages]
    vision_branch = [stage for stage in _VISION_BRANCH if stage in stages]

    if JobStage.DETECT_SCENES in stages:
        steps.append(_signature(JobStage.DETECT_SCENES, job_id))

    if audio_branch and vision_branch:
        # Обе ветки на месте — расходимся. Звуковая внутри себя
        # последовательна: распознавать нечего без аудио.
        steps.append(
            group(
                chain(*(_signature(stage, job_id) for stage in audio_branch)),
                chain(*(_signature(stage, job_id) for stage in vision_branch)),
            )
        )
    else:
        steps += [_signature(stage, job_id) for stage in (*audio_branch, *vision_branch)]

    steps += [_signature(stage, job_id) for stage in stages if stage in _TAIL_STAGES]
    return chain(*steps)


class CeleryDispatcher:
    """Реализация порта ``PipelineDispatcher``."""

    def start(self, job_id: UUID, *, from_stage: JobStage | None = None) -> str | None:
        result = build_flow(job_id, from_stage=from_stage).apply_async()
        # У цепочки ``id`` — это идентификатор её **последней** задачи,
        # а отменять надо всю. ``revoke`` по нему тянет за собой родителей,
        # поэтому храним именно его.
        log.info(
            "flow.started",
            job_id=str(job_id),
            from_stage=from_stage.value if from_stage else None,
            root_id=result.id,
        )
        return str(result.id)

    def cancel(self, root_id: str | None) -> bool:
        if not root_id:
            return False
        # ``terminate=True`` посылает сигнал уже выполняющейся задаче.
        # Дочерние процессы (ffmpeg) умирают вместе с ней благодаря
        # PDEATHSIG, который выставляется при запуске (см. core/infra/proc.py).
        celery_app.control.revoke(root_id, terminate=True, signal="SIGTERM")
        log.info("flow.cancelled", root_id=root_id)
        return True


#: Звуковая ветка: порядок внутри неё обязателен.
_AUDIO_BRANCH: Final[tuple[JobStage, ...]] = (
    JobStage.EXTRACT_AUDIO,
    JobStage.TRANSCRIBE,
    JobStage.ANALYZE_AUDIO,
    JobStage.ANALYZE_SPEECH,
)

_VISION_BRANCH: Final[tuple[JobStage, ...]] = (JobStage.ANALYZE_VISION,)

#: Всё, что идёт после схождения веток, — строго последовательно.
_TAIL_STAGES: Final[tuple[JobStage, ...]] = (
    JobStage.SCORE_SCENES,
    JobStage.LLM_RANK,
    JobStage.SELECT_HIGHLIGHTS,
    JobStage.RENDER_CLIPS,
    JobStage.GENERATE_METADATA,
    JobStage.FINALIZE,
)


def _tail(from_stage: JobStage | None) -> list[JobStage]:
    if from_stage is None:
        return list(FLOW_ORDER)
    if from_stage not in FLOW_ORDER:
        raise ValueError(f"Этап {from_stage} не участвует в конвейере")
    return list(FLOW_ORDER[FLOW_ORDER.index(from_stage) :])


def _signature(stage: JobStage, job_id: UUID) -> Signature:
    # ``immutable``: задача принимает только job_id и не должна получать
    # результат предыдущей — иначе цепочка начнёт передавать по сети
    # словари, которым место в базе (док 04, §4.2).
    return celery_app.signature(_TASK_NAMES[stage], args=(str(job_id),), immutable=True)
