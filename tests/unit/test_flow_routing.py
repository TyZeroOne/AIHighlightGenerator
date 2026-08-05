"""Граф конвейера и маршрутизация задач по очередям.

Файл существует из-за одного дефекта, который стоил дороже всех
остальных на Этапе 10 — и стоил ровно потому, что не проявлялся никак.

Очередь ``net`` завели на Этапе 6 под вызовы модели. Воркеру её слушать
не поручили. Полгода это ничего не значило: LLM-этап запускали командой
из CLI, а она выполняет сценарий на месте, не трогая брокер. На Этапе 10
конвейер впервые пошёл целиком — и встал после скоринга. Без ошибок,
без записи в логе, без единого следа: задача ушла в очередь, к которой
никто не подключён, и осталась там лежать.

Отсюда главная проверка ниже: **каждая очередь, в которую что-то
маршрутизируется, должна кем-то слушаться**. Это свойство конфигурации,
и проверяется оно арифметикой над двумя списками, а не поднятием стека.
"""

from __future__ import annotations

import fnmatch

import pytest

from core.domain.enums import JobStage
from workers.app import celery_app
from workers.flows import FLOW_ORDER, build_flow

pytestmark = pytest.mark.unit

#: Очереди, которые слушает воркер по умолчанию. Держится синхронно
#: с ``docker/entrypoint.sh`` и ``docker-compose.yml``: развёртывание
#: без отдельных GPU- и net-воркеров — основное.
DEFAULT_WORKER_QUEUES = frozenset({"cpu", "io", "gpu", "net"})


class TestRouting:
    def test_every_routed_queue_is_listened_to(self) -> None:
        """Регресс дефекта Этапа 10: задача в очередь без слушателя
        уходит молча, и джоб замирает без ошибок."""
        routed = {
            route["queue"]
            for route in celery_app.conf.task_routes.values()
            if isinstance(route, dict) and "queue" in route
        }
        forgotten = routed - DEFAULT_WORKER_QUEUES
        assert not forgotten, (
            f"Очереди {sorted(forgotten)} никто не слушает по умолчанию. "
            "Задачи в них уйдут молча, и джоб зависнет без единой ошибки."
        )

    def test_declared_queues_match_routing(self) -> None:
        """Обратная проверка: маршрут в необъявленную очередь — опечатка."""
        declared = {queue.name for queue in celery_app.conf.task_queues}
        routed = {
            route["queue"]
            for route in celery_app.conf.task_routes.values()
            if isinstance(route, dict) and "queue" in route
        }
        assert routed <= declared, f"Маршрут в необъявленную очередь: {sorted(routed - declared)}"

    def test_every_pipeline_stage_has_a_route(self) -> None:
        """У каждого этапа конвейера должна находиться очередь.

        Совпадение по маске проверяется тем же способом, каким его
        делает Celery, — иначе тест проверял бы не то, что происходит
        в рантайме.
        """
        patterns = list(celery_app.conf.task_routes.items())
        for stage in FLOW_ORDER:
            name = f"workers.tasks.pipeline.{stage.value}"
            matched = next(
                (route for pattern, route in patterns if fnmatch.fnmatch(name, pattern)),
                None,
            )
            assert matched is not None, f"{name} никуда не маршрутизируется"
            assert matched["queue"] in DEFAULT_WORKER_QUEUES


class TestFlowShape:
    def test_full_flow_covers_every_stage(self) -> None:
        """Все этапы, объявленные в порядке, попадают в граф.

        Проверка по именам задач в собранной сигнатуре, а не по
        ``FLOW_ORDER``: между списком и графом стоит код, и именно
        он может потерять этап.
        """
        from uuid import uuid4

        names = _task_names(build_flow(uuid4()))
        for stage in FLOW_ORDER:
            assert f"workers.tasks.pipeline.{stage.value}" in names, stage

    def test_retry_from_stage_drops_everything_before(self) -> None:
        from uuid import uuid4

        names = _task_names(build_flow(uuid4(), from_stage=JobStage.LLM_RANK))

        assert "workers.tasks.pipeline.detect_scenes" not in names
        assert "workers.tasks.pipeline.transcribe" not in names
        assert "workers.tasks.pipeline.llm_rank" in names
        assert "workers.tasks.pipeline.finalize" in names

    def test_retry_from_the_last_stage_is_a_single_task(self) -> None:
        from uuid import uuid4

        names = _task_names(build_flow(uuid4(), from_stage=JobStage.FINALIZE))
        assert names == ["workers.tasks.pipeline.finalize"]

    def test_unknown_stage_is_rejected(self) -> None:
        from uuid import uuid4

        with pytest.raises(ValueError, match="не участвует"):
            build_flow(uuid4(), from_stage=JobStage.INGEST)


def _task_names(signature: object) -> list[str]:
    """Имена всех задач внутри сигнатуры, как бы глубоко они ни лежали.

    Граф — это дерево из цепочек, групп и аккордов, и обходить его
    приходится рекурсивно: у каждого узла свои поля с вложенными
    сигнатурами.
    """
    names: list[str] = []
    task = getattr(signature, "task", None)
    if isinstance(task, str) and not task.startswith("celery."):
        names.append(task)

    for attribute in ("tasks", "body", "header"):
        nested = getattr(signature, attribute, None) or (
            signature.get(attribute) if isinstance(signature, dict) else None
        )
        if nested is None:
            continue
        if isinstance(nested, list | tuple):
            for item in nested:
                names += _task_names(item)
        else:
            names += _task_names(nested)
    return names
