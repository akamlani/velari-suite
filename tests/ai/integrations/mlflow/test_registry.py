"""Tests for velari_ai.integrations.mlflow.registry."""

import pytest

pytest.importorskip("mlflow")


class _FakeVersion:
    def __init__(self, version=1):
        self.version = version


def _make_registry(monkeypatch, load_result=None):
    import mlflow
    from mlflow import genai
    from velari_ai.integrations.mlflow.connection import Connector
    from velari_ai.integrations.mlflow.registry import PromptRegistry

    calls = []

    def _record(name, result=None):
        def _fn(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result
        return _fn

    monkeypatch.setattr(mlflow, "set_tracking_uri", _record("set_tracking_uri"))
    monkeypatch.setattr(genai, "register_prompt", _record("register_prompt", _FakeVersion()))
    monkeypatch.setattr(genai, "load_prompt", _record("load_prompt", load_result))
    monkeypatch.setattr(genai, "set_prompt_alias", _record("set_prompt_alias"))
    monkeypatch.setattr(genai, "delete_prompt_alias", _record("delete_prompt_alias"))
    monkeypatch.setattr(genai, "search_prompts", _record("search_prompts", iter(["p1", "p2"])))

    class _FakeConnector(Connector):
        def __init__(self) -> None:
            self._url = "sqlite:///registry-test.db"

    return PromptRegistry(_FakeConnector()), calls


def test_init_applies_connector_tracking_uri(monkeypatch):
    _, calls = _make_registry(monkeypatch)

    assert calls == [("set_tracking_uri", ("sqlite:///registry-test.db",), {})]


def test_register_omits_unset_optionals(monkeypatch):
    registry, calls = _make_registry(monkeypatch)

    version = registry.register("ticket-triage", "Classify: {{text}}")

    assert version.version == 1
    assert calls[-1] == ("register_prompt", ("ticket-triage", "Classify: {{text}}"), {})


def test_register_forwards_commit_message_and_tags(monkeypatch):
    registry, calls = _make_registry(monkeypatch)

    registry.register("ticket-triage", "Classify: {{text}}", commit_message="v2", tags={"team": "support"})

    assert calls[-1][2] == {"commit_message": "v2", "tags": {"team": "support"}}


@pytest.mark.parametrize(
    "kwargs, expected_call",
    [
        ({"version": 3}, (("churn-summary",), {"version": 3})),
        ({"alias": "production"}, (("prompts:/churn-summary@production",), {"version": None})),
        ({}, (("churn-summary",), {"version": None})),
    ],
)
def test_load_resolves_version_alias_or_latest(monkeypatch, kwargs, expected_call):
    sentinel = object()
    registry, calls = _make_registry(monkeypatch, load_result=sentinel)

    assert registry.load("churn-summary", **kwargs) is sentinel
    assert calls[-1] == ("load_prompt", *expected_call)


def test_load_with_version_and_alias_raises_valueerror(monkeypatch):
    registry, _ = _make_registry(monkeypatch)

    with pytest.raises(ValueError):
        registry.load("churn-summary", version=1, alias="production")


def test_set_and_delete_alias_forward_to_genai(monkeypatch):
    registry, calls = _make_registry(monkeypatch)

    registry.set_alias("contract-extractor", "production", 3)
    registry.delete_alias("contract-extractor", "staging")

    assert calls[-2:] == [
        ("set_prompt_alias", ("contract-extractor", "production", 3), {}),
        ("delete_prompt_alias", ("contract-extractor", "staging"), {}),
    ]


def test_list_forwards_filter_and_returns_list(monkeypatch):
    registry, calls = _make_registry(monkeypatch)

    assert registry.list("name LIKE 'ticket%'") == ["p1", "p2"]
    assert calls[-1] == ("search_prompts", (), {"filter_string": "name LIKE 'ticket%'"})


def _make_artifact_loader(monkeypatch, load_result=None):
    import mlflow
    from velari_ai.integrations.mlflow import registry as registry_module
    from velari_ai.integrations.mlflow.connection import Connector

    calls = []

    def _record(name, result=None):
        def _fn(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result
        return _fn

    monkeypatch.setattr(mlflow, "set_tracking_uri", _record("set_tracking_uri"))
    monkeypatch.setattr(registry_module, "load_model", _record("load_model", load_result))
    monkeypatch.setattr(mlflow, "register_model", _record("register_model", _FakeVersion()))
    monkeypatch.setattr(mlflow, "search_registered_models", _record("search_registered_models", ["m1", "m2"]))

    class _FakeConnector(Connector):
        def __init__(self) -> None:
            self._url = "sqlite:///registry-test.db"

    return registry_module.ArtifactLoader(_FakeConnector()), calls


@pytest.mark.parametrize(
    "kwargs, expected_uri",
    [
        ({"version": 3}, "models:/research-agent/3"),
        ({"alias": "champion"}, "models:/research-agent@champion"),
        ({}, "models:/research-agent/latest"),
    ],
)
def test_artifact_load_resolves_version_alias_or_latest(monkeypatch, kwargs, expected_uri):
    sentinel = object()
    loader, calls = _make_artifact_loader(monkeypatch, load_result=sentinel)

    assert loader.load("research-agent", **kwargs) is sentinel
    assert calls == [
        ("set_tracking_uri", ("sqlite:///registry-test.db",), {}),
        ("load_model", (expected_uri,), {}),
    ]


def test_artifact_load_with_version_and_alias_raises_valueerror(monkeypatch):
    loader, _ = _make_artifact_loader(monkeypatch)

    with pytest.raises(ValueError):
        loader.load("research-agent", version=1, alias="champion")


def test_artifact_loader_register_forwards_tags_and_kwargs(monkeypatch):
    loader, calls = _make_artifact_loader(monkeypatch)

    version = loader.register("models:/m-abc123", "contract-review-agent", tags={"team": "legal"}, await_registration_for=0)

    assert version.version == 1
    assert calls[-1] == (
        "register_model",
        ("models:/m-abc123", "contract-review-agent"),
        {"tags": {"team": "legal"}, "await_registration_for": 0},
    )


def test_artifact_loader_list_forwards_filter_and_returns_list(monkeypatch):
    loader, calls = _make_artifact_loader(monkeypatch)

    assert loader.list("name LIKE 'contract%'") == ["m1", "m2"]
    assert calls[-1] == ("search_registered_models", (), {"filter_string": "name LIKE 'contract%'", "max_results": None})
