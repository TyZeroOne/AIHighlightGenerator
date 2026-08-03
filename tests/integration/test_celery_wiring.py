"""Проверка обвязки Celery без брокера.

``task_always_eager`` выполняет задачи синхронно, прямо в процессе теста.
Это не заменяет проверку реального воркера (она в DoD этапа выполняется
вручную через Docker), но ловит главное: опечатки в именах, неверную
маршрутизацию, несериализуемые аргументы.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from core.domain.enums import JobStage
from workers.app import TASK_MODULES, celery_app
from workers.base import PipelineTask
from workers.tasks.pipeline import detect_scenes
from workers.tasks.system import echo, ping

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _eager_mode() -> None:
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True


class TestTasks:
    def test_ping_executes(self) -> None:
        result = ping.apply().get()
        assert result["status"] == "pong"
        assert result["hostname"]

    def test_echo_roundtrips_payload(self) -> None:
        payload = {"job_id": "abc", "count": 3}
        assert echo.apply(args=[payload]).get() == payload

    def test_tasks_have_explicit_names(self) -> None:
        """Имя задаётся явно, а не выводится из пути модуля: переезд файла
        не должен ломать сообщения, уже лежащие в очереди."""
        assert ping.name == "workers.tasks.system.ping"
        assert echo.name == "workers.tasks.system.echo"
        assert detect_scenes.name == "workers.tasks.pipeline.detect_scenes"

    def test_detect_scenes_declares_its_stage(self) -> None:
        """Этап задачи — источник ``stage`` в логах и трассировке."""
        assert detect_scenes.stage is JobStage.DETECT_SCENES


class TestTaskDiscovery:
    """Регистрация задач в чистом процессе.

    Это не педантизм, а тест на реальный баг, найденный при первом запуске
    в Docker. В обычном тесте задачи «регистрируются» просто потому, что
    тестовый файл импортировал их сам. Воркер же импортирует **только**
    ``workers.app`` — и если модуль задач в нём не подключён, задача уходит
    в очередь и возвращается как ``NotRegistered``.

    Поэтому проверка идёт в отдельном процессе, который импортирует ровно
    то же, что импортирует воркер, — и ничего больше.
    """

    def test_worker_registers_tasks_importing_only_the_app(self) -> None:
        src = str(Path(__file__).resolve().parents[2] / "src")
        # `import_default_modules()` — ровно то, что Celery делает при старте
        # воркера: подтягивает модули из conf.include / conf.imports.
        # Без этого вызова задачи не зарегистрированы даже при корректной
        # конфигурации, поэтому проверять просто `celery_app.tasks` бессмысленно.
        code = (
            f"import sys; sys.path.insert(0, {src!r});"
            "from workers.app import celery_app;"
            "celery_app.loader.import_default_modules();"
            "names = set(celery_app.tasks);"
            "expected = {'workers.tasks.system.ping', 'workers.tasks.system.echo',"
            " 'workers.tasks.pipeline.detect_scenes'};"
            "missing = expected - names;"
            "print('MISSING:' + ','.join(sorted(missing)))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, result.stderr
        assert "MISSING:\n" in result.stdout or result.stdout.strip() == "MISSING:", (
            f"Воркер не увидит эти задачи: {result.stdout.strip()}"
        )

    def test_every_declared_module_is_importable(self) -> None:
        """Опечатка в TASK_MODULES не должна доживать до продакшена."""
        import importlib

        for module in TASK_MODULES:
            importlib.import_module(module)


class TestConfiguration:
    def test_only_json_is_accepted(self) -> None:
        """pickle в брокере = исполнение произвольного кода из сообщения."""
        assert celery_app.conf.accept_content == ["json"]
        assert celery_app.conf.task_serializer == "json"

    def test_late_acknowledgement(self) -> None:
        """Убитый воркер не должен терять задачу."""
        assert celery_app.conf.task_acks_late is True
        assert celery_app.conf.task_reject_on_worker_lost is True

    def test_prefetch_is_disabled(self) -> None:
        """Задачи идут минутами: набирать их впрок — значит держать очередь."""
        assert celery_app.conf.worker_prefetch_multiplier == 1

    def test_all_four_queues_declared(self) -> None:
        names = {queue.name for queue in celery_app.conf.task_queues}
        assert names == {"cpu", "io", "gpu", "net"}

    def test_gpu_and_net_routes_exist(self) -> None:
        routes = celery_app.conf.task_routes
        assert routes["workers.tasks.pipeline.transcribe*"]["queue"] == "gpu"
        assert routes["workers.tasks.pipeline.llm_*"]["queue"] == "net"

    def test_detect_scenes_goes_to_the_cpu_queue(self) -> None:
        """Проверяем не запись в конфиге, а результат маршрутизации: правила
        сопоставляются по шаблонам, и «похожий» шаблон легко промахивается.
        Детекция сцен — чистый CPU: ни сети, ни GPU она не ждёт."""
        route = celery_app.amqp.router.route({}, "workers.tasks.pipeline.detect_scenes")

        assert route["queue"].name == "cpu"

    def test_system_tasks_go_to_the_io_queue(self) -> None:
        route = celery_app.amqp.router.route({}, "workers.tasks.system.ping")

        assert route["queue"].name == "io"


class TestPipelineTask:
    def test_retries_only_transient_errors(self) -> None:
        from core.domain.errors import ExternalServiceError, TransientError

        assert PipelineTask.autoretry_for == (TransientError, ExternalServiceError)
        assert PipelineTask.retry_backoff is True
        assert PipelineTask.retry_jitter is True

    def test_stage_defaults_to_none(self) -> None:
        assert PipelineTask.stage is None
        assert ping.stage is None

    def test_stage_can_be_declared(self) -> None:
        class Fake(PipelineTask):
            stage = JobStage.TRANSCRIBE

        assert Fake.stage is JobStage.TRANSCRIBE
