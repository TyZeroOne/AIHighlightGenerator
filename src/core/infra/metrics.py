"""Метрики Prometheus: что меряем и почему именно это.

## Три вопроса, на которые должны отвечать метрики

**«Работает ли вообще?»** — счётчик завершённых этапов с меткой исхода.
Всплеск `result="failed"` виден раньше, чем приходит первая жалоба.

**«Не встало ли колом?»** — глубина очередей. Это единственная метрика,
по которой отличается «медленно, но идёт» от «воркеры мертвы»: в обоих
случаях джоб висит, но в первом очередь пустая, а во втором растёт.

**«Где именно медленно?»** — гистограмма длительности этапов. Средним
здесь пользоваться нельзя: распознавание часового файла и распознавание
трейлера отличаются на два порядка, и среднее между ними не описывает
ни одного реального запуска. Нужны квантили, а для них — гистограмма.

## Почему multiprocess-режим

Воркер Celery работает пулом процессов: задачи выполняются в детях,
а HTTP-сервер метрик живёт в родителе. Счётчик, увеличенный в ребёнке,
родителю не виден — он в чужой памяти.

``prometheus_client`` решает это каталогом, в который дети пишут свои
значения, а экспортёр складывает их при опросе (`PROMETHEUS_MULTIPROC_DIR`).
Каталог обязан быть **локальным для процесса-родителя и очищаться при
старте**: файлы умерших детей иначе копятся и досчитываются вечно.

## Почему всё через один модуль

Метрика, объявленная дважды, роняет процесс на старте
(`Duplicated timeseries`). Единственный способ этого избежать —
объявлять их в одном месте и импортировать, а не создавать по месту
использования.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, multiprocess

__all__ = [
    "HTTP_DURATION",
    "HTTP_REQUESTS",
    "JOBS_FINISHED",
    "QUEUE_DEPTH",
    "STAGE_DURATION",
    "STAGE_RESULT",
    "build_registry",
    "prepare_multiprocess_dir",
    "stage_timer",
]

#: Границы гистограммы этапов, секунды. Не по умолчанию (те кончаются
#: на 10 с) — у нас нормальная длительность этапа лежит между секундой
#: и часом, и без верхних корзин всё сваливалось бы в `+Inf`.
_STAGE_BUCKETS: Final = (1, 5, 15, 30, 60, 120, 300, 600, 1200, 2400, 4800)

#: Границы для HTTP, секунды. Здесь наоборот: интересна миллисекундная
#: часть, а всё, что дольше пяти секунд, одинаково плохо.
_HTTP_BUCKETS: Final = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


STAGE_DURATION = Histogram(
    "highlight_stage_duration_seconds",
    "Длительность этапа конвейера",
    labelnames=("stage",),
    buckets=_STAGE_BUCKETS,
)

STAGE_RESULT = Counter(
    "highlight_stage_total",
    "Завершённые этапы по исходу",
    labelnames=("stage", "result"),
)

JOBS_FINISHED = Counter(
    "highlight_jobs_finished_total",
    "Джобы, дошедшие до терминального статуса",
    labelnames=("status",),
)

QUEUE_DEPTH = Gauge(
    "highlight_queue_depth",
    "Сообщений в очереди Celery",
    labelnames=("queue",),
    # Глубину очереди меряет тот, кто спросил, — она общая для всех
    # процессов, и складывать её по репликам было бы неверно.
    multiprocess_mode="livemostrecent",
)

HTTP_REQUESTS = Counter(
    "highlight_http_requests_total",
    "HTTP-запросы к API",
    labelnames=("method", "path", "status"),
)

HTTP_DURATION = Histogram(
    "highlight_http_duration_seconds",
    "Длительность обработки HTTP-запроса",
    labelnames=("method", "path"),
    buckets=_HTTP_BUCKETS,
)


@contextmanager
def stage_timer(stage: str) -> Iterator[None]:
    """Замерить этап и записать исход.

    Исход берётся из того, вылетело ли исключение, а не из возвращённого
    значения: пропуск этапа и его успех для метрики одинаковы — работа
    дошла до конца, — а вот падение отличается принципиально.
    """
    with STAGE_DURATION.labels(stage=stage).time():
        try:
            yield
        except Exception:
            STAGE_RESULT.labels(stage=stage, result="failed").inc()
            raise
    STAGE_RESULT.labels(stage=stage, result="succeeded").inc()


def prepare_multiprocess_dir() -> Path | None:
    """Подготовить каталог для метрик из дочерних процессов.

    Очистка при старте обязательна: ``prometheus_client`` не убирает файлы
    умерших процессов сам, и после десятка перезапусков воркера экспортёр
    начинает досчитывать значения давно не существующих детей.
    """
    raw = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if not raw:
        return None

    path = Path(raw)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_registry() -> CollectorRegistry:
    """Реестр для отдачи метрик.

    Без ``PROMETHEUS_MULTIPROC_DIR`` — обычный процессный реестр
    (так работает API: один процесс, всё в памяти). С ним — сборный,
    который дочитывает значения детей с диска.
    """
    registry = CollectorRegistry()
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        # Конструктор регистрирует коллектор в реестре побочным эффектом
        # и ничего не возвращает; типов у него нет.
        multiprocess.MultiProcessCollector(registry)  # type: ignore[no-untyped-call]
        return registry

    # Процессный реестр по умолчанию уже содержит наши метрики —
    # отдаём его, а не пустой.
    from prometheus_client import REGISTRY

    return REGISTRY
