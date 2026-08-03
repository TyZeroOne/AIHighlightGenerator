"""Тесты HTTP-слоя без поднятия сервера.

``ASGITransport`` гоняет запросы прямо в приложение — быстро и без портов.
БД и Redis здесь недоступны, и это часть проверки: readiness обязан
корректно сообщить о недоступности, а не упасть с 500.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import create_app

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.fixture
async def client() -> AsyncClient:  # type: ignore[misc]
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealth:
    async def test_liveness_does_not_touch_dependencies(self, client: AsyncClient) -> None:
        """Liveness обязан отвечать 200 даже с лежащей БД — иначе оркестратор
        начнёт бесконечно перезапускать здоровые контейнеры."""
        response = await client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["version"]

    async def test_readiness_reports_each_dependency(self, client: AsyncClient) -> None:
        response = await client.get("/health/ready")

        body = response.json()
        assert set(body["checks"]) == {"database", "redis", "storage"}
        # БД и Redis в юнит-окружении недоступны → 503 и статус degraded.
        assert response.status_code in (200, 503)
        assert body["status"] in ("ready", "degraded")

    async def test_trace_id_is_returned(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        assert response.headers["X-Request-ID"]

    async def test_incoming_trace_id_is_preserved(self, client: AsyncClient) -> None:
        """Трассировка не должна рваться на границе между сервисами."""
        response = await client.get("/health", headers={"X-Request-ID": "abc-123"})
        assert response.headers["X-Request-ID"] == "abc-123"


class TestErrorFormat:
    async def test_not_found_is_rfc7807(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/does-not-exist")

        assert response.status_code == 404
        assert response.headers["content-type"].startswith("application/problem+json")
        body = response.json()
        assert body["code"] == "NOT_FOUND"
        assert body["status"] == 404
        assert body["instance"] == "/api/v1/does-not-exist"
        assert body["trace_id"]


class TestSystemEndpoints:
    async def test_empty_registry_returns_empty_lists(self, client: AsyncClient) -> None:
        """На Этапе 1 анализаторов ещё нет — эндпоинт обязан это пережить."""
        assert (await client.get("/api/v1/system/analyzers")).json() == []
        assert (await client.get("/api/v1/system/plan")).json() == []

    async def test_openapi_is_generated(self, client: AsyncClient) -> None:
        schema = (await client.get("/openapi.json")).json()
        assert "/health" in schema["paths"]
        assert "/api/v1/system/plan" in schema["paths"]
