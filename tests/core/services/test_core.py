"""Tests for velari_core.integrations.fastapi.services.core."""

from fastapi.testclient import TestClient


def test_services_core_module_imports_and_builds_app():
    from velari_core.integrations.fastapi.services.core import app

    assert app.title == "Core Web Services"


def test_health_endpoint_returns_service_info():
    from velari_core.integrations.fastapi.services.core import app

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["service"] == "Core Web Services"
