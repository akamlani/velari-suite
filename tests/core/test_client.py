"""Tests for velari_core.core.services.client."""

import httpx
import pytest


class _FakeResponse:
    def __init__(self, status_code=200, content=b""):
        self.status_code = status_code
        self._content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPError(f"status {self.status_code}")


def test_get_returns_response_on_success(monkeypatch):
    from velari_core.core.services.client import HttpClient

    monkeypatch.setattr(httpx.Client, "get", lambda self, url, **kwargs: _FakeResponse(content=b"hello"))

    client = HttpClient()
    response = client.get("https://example.com")

    assert response._content == b"hello"


def test_get_propagates_http_status_error(monkeypatch):
    from velari_core.core.services.client import HttpClient

    monkeypatch.setattr(httpx.Client, "get", lambda self, url, **kwargs: _FakeResponse(status_code=500))

    client = HttpClient()
    with pytest.raises(httpx.HTTPError):
        client.get("https://example.com")


def test_get_propagates_connection_error(monkeypatch):
    from velari_core.core.services.client import HttpClient

    def _raise(self, url, **kwargs):
        raise httpx.HTTPError("simulated connection failure")

    monkeypatch.setattr(httpx.Client, "get", _raise)

    client = HttpClient()
    with pytest.raises(httpx.HTTPError):
        client.get("https://example.com")


def test_headers_merged_into_client():
    from velari_core.core.services.client import HttpClient

    client = HttpClient(headers={"User-Agent": "velari-ai/1.0"})

    assert client._client.headers["User-Agent"] == "velari-ai/1.0"


def test_default_headers_applied_when_unset():
    from velari_core.core.services.client import HttpClient

    client = HttpClient()

    assert client._client.headers["Content-Type"] == "application/json"
    assert client._client.headers["Accept"] == "application/json"


def test_custom_headers_merge_with_defaults_instead_of_replacing():
    from velari_core.core.services.client import HttpClient

    client = HttpClient(headers={"Authorization": "Bearer token"})

    assert client._client.headers["Authorization"] == "Bearer token"
    assert client._client.headers["Content-Type"] == "application/json"


def test_follow_redirects_defaults_true():
    from velari_core.core.services.client import HttpClient

    client = HttpClient()

    assert client._client.follow_redirects is True


def test_get_joins_base_url_and_path(monkeypatch):
    from velari_core.core.services.client import HttpClient

    captured = {}

    def _fake_get(self, url, **kwargs):
        captured["url"] = url
        return _FakeResponse()

    monkeypatch.setattr(httpx.Client, "get", _fake_get)

    client = HttpClient(base_url="https://api.internal")
    client.get("/datasets/churn-2026")

    assert captured["url"] == "https://api.internal/datasets/churn-2026"


def test_post_returns_response_on_success(monkeypatch):
    from velari_core.core.services.client import HttpClient

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kwargs: _FakeResponse(content=b"created"))

    client = HttpClient(base_url="https://api.internal")
    response = client.post("/datasets", json={"name": "churn-2026"})

    assert response._content == b"created"


def test_post_propagates_http_error(monkeypatch):
    from velari_core.core.services.client import HttpClient

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kwargs: _FakeResponse(status_code=500))

    client = HttpClient(base_url="https://api.internal")
    with pytest.raises(httpx.HTTPError):
        client.post("/datasets", json={"name": "churn-2026"})


def test_connect_returns_self_on_successful_health_check(monkeypatch):
    from velari_core.core.services.client import HttpClient

    monkeypatch.setattr(httpx.Client, "get", lambda self, url, **kwargs: _FakeResponse())

    client = HttpClient(base_url="https://api.internal")
    result = client.connect()

    assert result is client


def test_connect_raises_and_preserves_prior_session_on_failed_health_check(monkeypatch):
    from velari_core.core.services.client import HttpClient

    monkeypatch.setattr(httpx.Client, "get", lambda self, url, **kwargs: _FakeResponse(status_code=503))

    client = HttpClient(base_url="https://api.internal")
    prior_session = client._client

    with pytest.raises(httpx.HTTPError):
        client.connect()

    assert client._client is prior_session


def test_close_closes_underlying_session():
    from velari_core.core.services.client import HttpClient

    client = HttpClient()
    client.close()

    assert client._client.is_closed


def test_context_manager_closes_session_on_exit():
    from velari_core.core.services.client import HttpClient

    with HttpClient() as client:
        assert not client._client.is_closed

    assert client._client.is_closed


def test_context_manager_closes_session_on_exception():
    from velari_core.core.services.client import HttpClient

    client = HttpClient()
    with pytest.raises(ValueError):
        with client:
            raise ValueError("boom")

    assert client._client.is_closed
