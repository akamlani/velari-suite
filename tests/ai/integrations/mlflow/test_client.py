"""Tests for velari_ai.integrations.mlflow.client."""

import pytest

pytest.importorskip("mlflow")


class _FakeVersion:
    def __init__(self, version=1):
        self.version = version


def _make_artifact_registry(monkeypatch, registered=True):
    import mlflow
    from mlflow.exceptions import MlflowException
    from velari_ai.integrations.mlflow import client as client_module
    from velari_ai.integrations.mlflow.connection import Connector

    calls = []

    def _record(name, result=None):
        def _fn(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result
        return _fn

    def _get_registered_model(name):
        calls.append(("get_registered_model", (name,), {}))
        if not registered:
            raise MlflowException("not found")

    class _FakeClient:
        def __init__(self, tracking_uri=None):
            calls.append(("MlflowClient", (), {"tracking_uri": tracking_uri}))

        get_registered_model          = staticmethod(_get_registered_model)
        create_registered_model       = staticmethod(_record("create_registered_model"))
        create_model_version          = staticmethod(_record("create_model_version", _FakeVersion()))
        set_registered_model_alias    = staticmethod(_record("set_registered_model_alias"))
        search_registered_models      = staticmethod(_record("search_registered_models", iter(["m1", "m2"])))

    monkeypatch.setattr(mlflow, "set_tracking_uri", _record("set_tracking_uri"))
    monkeypatch.setattr(client_module, "MlflowClient", _FakeClient)

    class _FakeConnector(Connector):
        def __init__(self) -> None:
            self._url = "sqlite:///registry-test.db"

    return client_module.ArtifactRegistry(_FakeConnector()), calls


def test_artifact_registry_scopes_client_to_connector_without_global_state(monkeypatch):
    _, calls = _make_artifact_registry(monkeypatch)

    assert calls == [("MlflowClient", (), {"tracking_uri": "sqlite:///registry-test.db"})]


@pytest.mark.parametrize(
    "registered, tags",
    [(True, None), (False, {"team": "support"})],
)
def test_artifact_register_creates_registered_model_only_when_missing(monkeypatch, registered, tags):
    registry, calls = _make_artifact_registry(monkeypatch, registered=registered)

    version = registry.register("models:/m-abc123", "billing-agent", tags=tags)

    assert version.version == 1
    assert any(name == "create_registered_model" for name, *_ in calls) is (not registered)
    assert calls[-1] == ("create_model_version", ("billing-agent", "models:/m-abc123"), {"tags": tags})


def test_artifact_set_alias_coerces_version_to_str(monkeypatch):
    registry, calls = _make_artifact_registry(monkeypatch)

    registry.set_alias("doc-retrieval-chain", "champion", 4)

    assert calls[-1] == ("set_registered_model_alias", ("doc-retrieval-chain", "champion", "4"), {})


def test_artifact_list_forwards_filter_and_returns_list(monkeypatch):
    registry, calls = _make_artifact_registry(monkeypatch)

    assert registry.list("name LIKE 'billing%'") == ["m1", "m2"]
    assert calls[-1] == ("search_registered_models", (), {"filter_string": "name LIKE 'billing%'"})
