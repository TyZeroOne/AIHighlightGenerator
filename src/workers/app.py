"""Celery-приложение: очереди, маршрутизация, жизненный цикл воркера."""

from __future__ import annotations

import os
from typing import Any

from celery import Celery
from celery.schedules import crontab
from celery.signals import (
    worker_process_init,
    worker_process_shutdown,
    worker_ready,
    worker_shutting_down,
)
from kombu import Queue

from core.config import get_settings
from core.logging import configure_logging, get_logger

__all__ = ["celery_app"]

log = get_logger(__name__)
settings = get_settings()

#: Модули с задачами. Список явный, а не `autodiscover_tasks`, намеренно:
#: autodiscover ищет модуль с именем `tasks` внутри каждого указанного пакета,
#: и ошибиться в нём легко — а промах виден только на живом воркере в виде
#: `NotRegistered`. Явный список проверяем глазами и греппается.
#: Каждый новый модуль задач добавляется сюда.
TASK_MODULES: list[str] = [
    "workers.tasks.system",
    "workers.tasks.pipeline",
    "workers.tasks.maintenance",
]

celery_app = Celery("highlight_generator", include=TASK_MODULES)

celery_app.conf.update(
    broker_url=settings.celery_broker_url,
    result_backend=settings.celery_result_backend,
    # --- сериализация ---
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # pickle умеет сериализовать что угодно — и именно поэтому опасен:
    # сообщение из брокера исполняется как код. Только json.
    timezone="UTC",
    enable_utc=True,
    # --- надёжность ---
    task_acks_late=True,
    # Подтверждение ПОСЛЕ выполнения: если воркера убили посреди задачи,
    # сообщение вернётся в очередь и будет выполнено заново. Именно поэтому
    # все задачи обязаны быть идемпотентными (док 04, §4.6).
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    # По умолчанию воркер набирает пачку задач впрок. Для наших задач
    # (минуты, а не миллисекунды) это приводит к тому, что один воркер
    # держит очередь, пока другие простаивают.
    task_track_started=True,
    task_time_limit=settings.celery_task_time_limit,
    task_soft_time_limit=settings.celery_task_time_limit - 300,
    # --- результаты ---
    result_expires=86400,
    result_extended=True,
    # --- маршрутизация ---
    task_default_queue="cpu",
    task_queues=[
        Queue("cpu"),  # ffmpeg, скоринг, нарезка
        Queue("io"),  # скачивание, загрузка в storage, финализация
        Queue("gpu"),  # whisper, SER, детекция лиц
        Queue("net"),  # вызовы LLM
    ],
)

# Очереди по типу ресурса, а не по «модулю». Смысл: задача, ждущая сеть,
# не должна занимать слот у задачи, считающей на GPU (док 04, §4.3).
celery_app.conf.task_routes = {
    "workers.tasks.system.*": {"queue": "io"},
    "workers.tasks.pipeline.ingest*": {"queue": "io"},
    "workers.tasks.pipeline.finalize*": {"queue": "io"},
    "workers.tasks.pipeline.transcribe*": {"queue": "gpu"},
    "workers.tasks.pipeline.analyze_vision*": {"queue": "gpu"},
    "workers.tasks.pipeline.analyze_speech*": {"queue": "gpu"},
    # Аудиогруппа поехала в очередь GPU на Этапе 5, когда в ней появился
    # AudioSet-классификатор: он в ней самый дорогой, и на процессоре
    # занимает больше времени, чем видеогруппа целиком. Громкость и паузы
    # рядом с ним — арифметика, ради которой держать отдельный маршрут
    # незачем.
    #
    # На конфигурации по умолчанию не меняется ничего: CPU-воркер слушает
    # и очередь `gpu` тоже.
    "workers.tasks.pipeline.analyze_audio*": {"queue": "gpu"},
    "workers.tasks.pipeline.llm_*": {"queue": "net"},
    # Всё остальное — на CPU. Сюда попадает detect_scenes: это чистая
    # арифметика по кадрам, ни сети, ни GPU она не ждёт. И render_clips:
    # там всё время съедает x264, а детектор лиц для вертикального кропа
    # весит 232 КБ и на общем фоне незаметен — отправлять этап в очередь
    # `gpu` значило бы занять видеокарту работой, которой она не нужна.
    "workers.tasks.*": {"queue": "cpu"},
}

# Расписание. Обе задачи уходят в очередь `io`: они ничего не считают,
# только ходят в базу и хранилище, и занимать ими процессорный воркер,
# пока тот режет клипы, незачем.
celery_app.conf.beat_schedule = {
    "cleanup-expired-assets": {
        "task": "workers.tasks.maintenance.cleanup_expired_assets",
        # Раз в час, а не раз в сутки: артефакты копятся непрерывно,
        # и суточный запуск означал бы пилу на графике свободного места
        # с амплитудой в десятки гигабайт.
        "schedule": crontab(minute="17"),
        "options": {"queue": "io", "expires": 3000},
    },
    "reap-stuck-jobs": {
        "task": "workers.tasks.maintenance.reap_stuck_jobs",
        # Чаще уборки: зависший джоб виден пользователю, а лишние файлы —
        # только нам.
        "schedule": crontab(minute="*/10"),
        "options": {"queue": "io", "expires": 500},
    },
}


@worker_ready.connect
def _start_metrics_server(**_: Any) -> None:
    """Поднять экспортёр метрик в главном процессе воркера.

    Именно в главном, а не в детях: порт один, а детей четыре, и второй
    из них упал бы на «address already in use». Значения самих детей
    попадают сюда через каталог ``PROMETHEUS_MULTIPROC_DIR`` — они пишут
    туда файлы, экспортёр складывает их при опросе.

    Каталог готовится (и чистится) здесь же, до старта сервера: файлы
    процессов, умерших в прошлый запуск, иначе досчитывались бы вечно.
    """
    if not settings.metrics_enabled:
        return

    from prometheus_client import start_http_server

    from core.infra.metrics import build_registry, prepare_multiprocess_dir

    prepare_multiprocess_dir()
    try:
        start_http_server(settings.worker_metrics_port, registry=build_registry())
    except OSError as error:  # pragma: no cover — занятый порт
        log.warning("metrics.server_failed", port=settings.worker_metrics_port, error=str(error))
        return
    log.info("metrics.server_started", port=settings.worker_metrics_port)


@worker_ready.connect
def _warn_about_gpu_queue_on_cpu(**_: Any) -> None:
    """Предупредить, что тяжёлые задачи пойдут на процессоре.

    Воркер без видеокарты, слушающий очередь ``gpu``, — это штатная и нужная
    конфигурация: иначе на машине без GPU распознавание речи выполнять
    некому, и джоб зависает молча. Но выполняться оно будет впятеро дольше,
    и это должно быть видно в логе, а не выясняться через полчаса ожидания.
    """
    queues = {name.strip() for name in os.environ.get("CELERY_QUEUES", "").split(",")}
    if "gpu" not in queues:
        return

    from speech.whisper import resolve_device

    if resolve_device(settings.device) != "cuda":
        log.warning(
            "worker.gpu_queue_without_cuda",
            model=settings.whisper_model,
            hint="распознавание речи пойдёт на CPU; поднимите профиль gpu "
            "и сузьте WORKER_CPU_QUEUES до cpu,io",
        )


@worker_process_init.connect
def _init_worker_process(**_: Any) -> None:
    """Инициализация каждого процесса-воркера.

    Именно ``worker_process_init``, а не импорт модуля: при пуле ``prefork``
    процессы форкаются, и ресурс, созданный до форка (соединение с БД,
    модель на GPU), либо ломается, либо дублируется в каждом процессе.
    """
    configure_logging(level=settings.log_level, fmt=settings.log_format)
    from core.container import get_container

    get_container()
    log.info("worker.process_initialized")


@worker_process_shutdown.connect
def _shutdown_worker_process(**_: Any) -> None:
    """Отпустить ресурсы процесса-воркера.

    Порядок обязателен: сначала закрываем пул соединений (это асинхронная
    операция, ей нужен живой цикл), только потом гасим сам цикл. Наоборот —
    незакрытые сокеты к Postgres и предупреждения в логе остановки.
    """
    from core.container import get_container
    from workers.runtime import close_loop, run_async

    try:
        run_async(get_container().aclose())
    except Exception as error:  # pragma: no cover — путь остановки процесса
        log.warning("worker.shutdown_cleanup_failed", error=str(error))
    finally:
        close_loop()
    log.info("worker.process_shutdown")


@worker_shutting_down.connect
def _shutdown_worker(**_: Any) -> None:
    log.info("worker.shutting_down")
