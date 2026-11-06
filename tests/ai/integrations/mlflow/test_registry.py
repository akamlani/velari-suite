"""Tests for velari_ai.integrations.mlflow.registry."""

from typing import Any

import pandas as pd
import pytest

pytest.importorskip("mlflow")


class _FakeVersion:
    def __init__(self, version=1):
        self.version = version


class _FakePrompt:
    def __init__(self, name):
        self.name = name
        self.description = f"{name} prompt"
        self.tags = {"team": "support"}
        self.creation_timestamp = 1700000000000


class _FakeModel(_FakePrompt):
    def __init__(self, name):
        super().__init__(name)
        self.last_updated_timestamp = 1700000005000


class _FakeConnectorFactory:
    @staticmethod
    def make():
        from velari_ai.integrations.mlflow.connection import Connector

        class _FakeConnector(Connector):
            def __init__(self) -> None:
                self._url = "sqlite:///registry-test.db"

        return _FakeConnector()


def _recorder(calls):
    def _record(name, result=None):
        def _fn(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result
        return _fn
    return _record


def _make_prompt_registry(monkeypatch, load_result=None, prompts=None):
    import mlflow
    from mlflow import genai
    from velari_ai.integrations.mlflow.registry import PromptRegistry

    calls = []
    _record = _recorder(calls)
    monkeypatch.setattr(mlflow, "set_tracking_uri", _record("set_tracking_uri"))
    monkeypatch.setattr(genai, "register_prompt", _record("register_prompt", _FakeVersion()))
    monkeypatch.setattr(genai, "load_prompt", _record("load_prompt", load_result))
    monkeypatch.setattr(genai, "set_prompt_alias", _record("set_prompt_alias"))
    monkeypatch.setattr(genai, "delete_prompt_alias", _record("delete_prompt_alias"))
    monkeypatch.setattr(genai, "search_prompts", _record("search_prompts", prompts or ["p1", "p2"]))

    return PromptRegistry(_FakeConnectorFactory.make()), calls


class TestPromptRegistry:
    def test_init_applies_connector_tracking_uri(self, monkeypatch):
        _, calls = _make_prompt_registry(monkeypatch)

        assert calls == [("set_tracking_uri", ("sqlite:///registry-test.db",), {})]

    def test_create_omits_unset_optionals(self, monkeypatch):
        registry, calls = _make_prompt_registry(monkeypatch)

        version = registry.create("ticket-triage", "Classify: {{text}}")

        assert version.version == 1
        assert calls[-1] == ("register_prompt", ("ticket-triage", "Classify: {{text}}"), {})

    def test_create_forwards_commit_message_and_tags(self, monkeypatch):
        registry, calls = _make_prompt_registry(monkeypatch)

        registry.create("ticket-triage", "Classify: {{text}}", commit_message="v2", tags={"team": "support"})

        assert calls[-1][2] == {"commit_message": "v2", "tags": {"team": "support"}}

    @pytest.mark.parametrize(
        "kwargs, expected_call",
        [
            ({"version": 3}, (("churn-summary",), {"version": 3})),
            ({"alias": "production"}, (("prompts:/churn-summary@production",), {"version": None})),
            ({}, (("churn-summary",), {"version": None})),
        ],
    )
    def test_get_resolves_version_alias_or_latest(self, monkeypatch, kwargs, expected_call):
        sentinel = object()
        registry, calls = _make_prompt_registry(monkeypatch, load_result=sentinel)

        assert registry.get("churn-summary", **kwargs) is sentinel
        assert calls[-1] == ("load_prompt", *expected_call)

    def test_get_with_version_and_alias_raises_valueerror(self, monkeypatch):
        registry, _ = _make_prompt_registry(monkeypatch)

        with pytest.raises(ValueError):
            registry.get("churn-summary", version=1, alias="production")

    def test_has_false_when_get_raises(self, monkeypatch):
        from mlflow import genai
        from mlflow.exceptions import MlflowException

        registry, _ = _make_prompt_registry(monkeypatch)

        def _missing(*args, **kwargs):
            raise MlflowException("prompt not found")

        monkeypatch.setattr(genai, "load_prompt", _missing)

        assert "no-such-prompt" not in registry

    def test_set_and_delete_alias_forward_to_genai(self, monkeypatch):
        registry, calls = _make_prompt_registry(monkeypatch)

        registry.set_alias("contract-extractor", "production", 3)
        registry.delete_alias("contract-extractor", "staging")

        assert calls[-2:] == [
            ("set_prompt_alias", ("contract-extractor", "production", 3), {}),
            ("delete_prompt_alias", ("contract-extractor", "staging"), {}),
        ]

    def test_list_forwards_filter_and_returns_list(self, monkeypatch):
        registry, calls = _make_prompt_registry(monkeypatch)

        assert registry.list("name LIKE 'ticket%'") == ["p1", "p2"]
        assert calls[-1] == ("search_prompts", (), {"filter_string": "name LIKE 'ticket%'"})

    def test_list_forwards_extra_options(self, monkeypatch):
        registry, calls = _make_prompt_registry(monkeypatch)

        registry.list(order_by=["name"])

        assert calls[-1] == ("search_prompts", (), {"order_by": ["name"]})

    def test_to_dataframe_lists_prompts(self, monkeypatch):
        registry, _ = _make_prompt_registry(monkeypatch, prompts=[_FakePrompt("ticket-triage"), _FakePrompt("ticket-summary")])

        df_prompts = registry.to_dataframe()

        assert df_prompts.columns.tolist() == ["name", "description", "tags", "creation_timestamp"]
        assert df_prompts["name"].tolist() == ["ticket-triage", "ticket-summary"]


def _make_artifact_registry(monkeypatch, load_result=None, models=None):
    import mlflow
    from velari_ai.integrations.mlflow import registry as registry_module

    calls = []
    _record = _recorder(calls)
    monkeypatch.setattr(mlflow, "set_tracking_uri", _record("set_tracking_uri"))
    monkeypatch.setattr(registry_module, "load_model", _record("load_model", load_result))
    monkeypatch.setattr(mlflow, "register_model", _record("register_model", _FakeVersion()))
    monkeypatch.setattr(mlflow, "search_registered_models", _record("search_registered_models", models or ["m1", "m2"]))

    return registry_module.ArtifactRegistry(_FakeConnectorFactory.make()), calls


class TestArtifactRegistry:
    @pytest.mark.parametrize(
        "kwargs, expected_uri",
        [
            ({"version": 3}, "models:/research-agent/3"),
            ({"alias": "champion"}, "models:/research-agent@champion"),
            ({}, "models:/research-agent/latest"),
        ],
    )
    def test_get_resolves_version_alias_or_latest(self, monkeypatch, kwargs, expected_uri):
        sentinel = object()
        registry, calls = _make_artifact_registry(monkeypatch, load_result=sentinel)

        assert registry.get("research-agent", **kwargs) is sentinel
        assert calls == [
            ("set_tracking_uri", ("sqlite:///registry-test.db",), {}),
            ("load_model", (expected_uri,), {}),
        ]

    def test_get_with_version_and_alias_raises_valueerror(self, monkeypatch):
        registry, _ = _make_artifact_registry(monkeypatch)

        with pytest.raises(ValueError):
            registry.get("research-agent", version=1, alias="champion")

    def test_has_false_when_get_raises(self, monkeypatch):
        from mlflow.exceptions import MlflowException
        from velari_ai.integrations.mlflow import registry as registry_module

        registry, _ = _make_artifact_registry(monkeypatch)

        def _missing(*args, **kwargs):
            raise MlflowException("model not found")

        monkeypatch.setattr(registry_module, "load_model", _missing)

        assert "no-such-agent" not in registry

    def test_create_forwards_tags_and_kwargs(self, monkeypatch):
        registry, calls = _make_artifact_registry(monkeypatch)

        version = registry.create("models:/m-abc123", "contract-review-agent", tags={"team": "legal"}, await_registration_for=0)

        assert version.version == 1
        assert calls[-1] == (
            "register_model",
            ("models:/m-abc123", "contract-review-agent"),
            {"tags": {"team": "legal"}, "await_registration_for": 0},
        )

    def test_list_forwards_filter_and_returns_list(self, monkeypatch):
        registry, calls = _make_artifact_registry(monkeypatch)

        assert registry.list("name LIKE 'contract%'") == ["m1", "m2"]
        assert calls[-1] == ("search_registered_models", (), {"filter_string": "name LIKE 'contract%'", "max_results": None})

    def test_list_forwards_extra_options(self, monkeypatch):
        registry, calls = _make_artifact_registry(monkeypatch)

        registry.list(order_by=["name"])

        assert calls[-1][2]["order_by"] == ["name"]

    def test_to_dataframe_lists_registered_models(self, monkeypatch):
        registry, _ = _make_artifact_registry(monkeypatch, models=[_FakeModel("contract-review-agent")])

        df_models = registry.to_dataframe()

        assert df_models.columns.tolist() == ["name", "description", "tags", "creation_timestamp", "last_updated_timestamp"]
        assert df_models["name"].tolist() == ["contract-review-agent"]


class _FakeDataset:
    def __init__(self, calls, name="support-ticket-qa", tags=None):
        self._calls = calls
        self.name = name
        self.dataset_id = f"d-{name}"
        self.digest = "abc123"
        self.tags = tags or {}

    def merge_records(self, records):
        self._calls.append(("merge_records", (records,), {}))
        return self

    def delete_records(self, record_ids):
        self._calls.append(("delete_records", (record_ids,), {}))
        return len(record_ids)


def _make_dataset_registry(monkeypatch, options=None):
    import mlflow
    from mlflow import genai
    from velari_ai.integrations.mlflow.connection import Connector
    from velari_ai.integrations.mlflow.registry import DatasetRegistry

    calls = []

    def _record(name, result=None):
        def _fn(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result
        return _fn

    fake = _FakeDataset(calls)
    monkeypatch.setattr(mlflow, "set_tracking_uri", _record("set_tracking_uri"))
    monkeypatch.setattr(genai.datasets, "create_dataset", _record("create_dataset", fake))
    monkeypatch.setattr(genai.datasets, "get_dataset", _record("get_dataset", fake))
    monkeypatch.setattr(genai.datasets, "search_datasets", _record("search_datasets", [fake]))
    monkeypatch.setattr(genai.datasets, "delete_dataset", _record("delete_dataset"))
    monkeypatch.setattr(genai.datasets, "set_dataset_tags", _record("set_dataset_tags"))
    monkeypatch.setattr(genai.datasets, "delete_dataset_tag", _record("delete_dataset_tag"))

    class _FakeConnector(Connector):
        def __init__(self) -> None:
            self._url = "sqlite:///registry-test.db"

    return DatasetRegistry(_FakeConnector(), options), calls, fake


@pytest.fixture
def make_real_dataset_registry(tmp_path):
    import mlflow
    from velari_ai.integrations.mlflow.connection import Connector
    from velari_ai.integrations.mlflow.registry import DatasetRegistry

    class _FakeConnector(Connector):
        def __init__(self) -> None:
            self._url = f"sqlite:///{tmp_path / 'mlflow.db'}"

    previous_uri = mlflow.get_tracking_uri()
    experiment_id = None

    def _make(options=None):
        nonlocal experiment_id
        registry = DatasetRegistry(_FakeConnector(), options)
        experiment_id = experiment_id or mlflow.create_experiment(
            "dataset-tests", artifact_location=str(tmp_path / "artifacts"),
        )
        return registry, experiment_id

    yield _make
    mlflow.set_tracking_uri(previous_uri)


@pytest.fixture
def real_dataset_registry(make_real_dataset_registry):
    return make_real_dataset_registry()


_TICKETS = {
    "question": ["How do I reset my password?"],
    "answer":   ["Visit /account/reset."],
    "team":     ["support"],
}


class TestDatasetRegistry:
    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({"input_keys": ["question"]}, [{"inputs": {"question": "How do I reset my password?"}}]),
            (
                {"input_keys": ["question"], "output_keys": ["answer"]},
                [{"inputs": {"question": "How do I reset my password?"}, "expectations": {"answer": "Visit /account/reset."}}],
            ),
            (
                {"input_keys": ["question"], "metadata_keys": ["team"]},
                [{"inputs": {"question": "How do I reset my password?"}, "tags": {"team": "support"}}],
            ),
        ],
    )
    def test_from_dataframe_maps_columns_to_record_fields(self, kwargs, expected):
        from velari_ai.integrations.mlflow.registry import DatasetRegistry

        assert DatasetRegistry.from_dataframe(pd.DataFrame(_TICKETS), **kwargs) == expected

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({}, {"name": "support-ticket-qa"}),
            ({"dataset_tags": {"team": "support"}}, {"name": "support-ticket-qa", "tags": {"team": "support"}}),
            ({"description": "Support QA"}, {"name": "support-ticket-qa", "tags": {"velari.description": "Support QA"}}),
            ({"experiment_id": "7"}, {"name": "support-ticket-qa", "experiment_id": "7"}),
            ({"backend_option": 1}, {"name": "support-ticket-qa", "backend_option": 1}),
        ],
    )
    def test_create_forwards_only_set_options(self, monkeypatch, kwargs, expected):
        registry, calls, _ = _make_dataset_registry(monkeypatch)

        registry.create("support-ticket-qa", **kwargs)

        assert calls[-1] == ("create_dataset", (), expected)

    def test_create_with_records_merges_converted_records(self, monkeypatch):
        registry, calls, fake = _make_dataset_registry(monkeypatch)

        dataset = registry.create("support-ticket-qa", pd.DataFrame(_TICKETS), input_keys=["question"])

        assert dataset is fake
        assert calls[-1] == ("merge_records", ([{"inputs": {"question": "How do I reset my password?"}}],), {})

    def test_create_with_records_without_input_keys_raises_valueerror(self, monkeypatch):
        registry, calls, _ = _make_dataset_registry(monkeypatch)

        with pytest.raises(ValueError):
            registry.create("support-ticket-qa", pd.DataFrame(_TICKETS))
        assert calls == [("set_tracking_uri", ("sqlite:///registry-test.db",), {})]

    def test_get_returns_dataset_by_name(self, monkeypatch):
        registry, calls, fake = _make_dataset_registry(monkeypatch)

        assert registry.get("support-ticket-qa") is fake
        assert calls[-1] == ("get_dataset", (), {"name": "support-ticket-qa"})

    def test_has_true_for_existing_dataset(self, monkeypatch):
        registry, _, _ = _make_dataset_registry(monkeypatch)

        assert "support-ticket-qa" in registry

    def test_extend_upserts_records_then_dataset_tags(self, monkeypatch):
        registry, calls, _ = _make_dataset_registry(monkeypatch)

        registry.extend("support-ticket-qa", pd.DataFrame(_TICKETS), input_keys=["question"], dataset_tags={"stage": "v2"})

        assert calls[-2:] == [
            ("merge_records", ([{"inputs": {"question": "How do I reset my password?"}}],), {}),
            ("set_dataset_tags", ("d-support-ticket-qa", {"stage": "v2"}), {}),
        ]

    def test_extend_accepts_dataset_object(self, monkeypatch):
        registry, calls, fake = _make_dataset_registry(monkeypatch)

        existing: Any = fake
        registry.extend(existing, pd.DataFrame(_TICKETS), input_keys=["question"])

        assert ("get_dataset", (), {"name": "support-ticket-qa"}) in calls
        assert calls[-1][0] == "merge_records"

    def test_extend_unknown_option_raises_typeerror(self, monkeypatch):
        registry, calls, _ = _make_dataset_registry(monkeypatch)

        with pytest.raises(TypeError):
            registry.extend("support-ticket-qa", pd.DataFrame(_TICKETS), input_keys=["question"], split_key="team")
        assert "merge_records" not in [name for name, _, _ in calls]

    def test_extend_without_dataset_tags_leaves_tags_untouched(self, monkeypatch):
        registry, calls, _ = _make_dataset_registry(monkeypatch)

        registry.extend("support-ticket-qa", pd.DataFrame(_TICKETS), input_keys=["question"])

        assert "set_dataset_tags" not in [name for name, _, _ in calls]

    def test_update_sets_values_and_deletes_none_valued_tags(self, monkeypatch):
        registry, calls, _ = _make_dataset_registry(monkeypatch)

        registry.update("support-ticket-qa", dataset_tags={"stage": "prod", "draft": None})

        assert calls[-2:] == [
            ("set_dataset_tags", ("d-support-ticket-qa", {"stage": "prod"}), {}),
            ("delete_dataset_tag", ("d-support-ticket-qa", "draft"), {}),
        ]

    def test_update_with_none_value_removes_tag(self, real_dataset_registry):
        registry, experiment_id = real_dataset_registry
        registry.create(
            "support-ticket-qa", dataset_tags={"draft": "yes", "owner": "support-team"}, experiment_id=experiment_id,
        )

        registry.update("support-ticket-qa", dataset_tags={"draft": None, "stage": "prod"})

        tags = registry.get("support-ticket-qa").tags
        assert "draft" not in tags
        assert tags["owner"] == "support-team"
        assert tags["stage"] == "prod"

    def test_delete_without_extras_deletes_dataset_by_id(self, monkeypatch):
        registry, calls, _ = _make_dataset_registry(monkeypatch)

        registry.delete("support-ticket-qa")

        assert calls[-1] == ("delete_dataset", (), {"dataset_id": "d-support-ticket-qa"})

    def test_delete_removes_dataset_from_real_store(self, real_dataset_registry):
        registry, experiment_id = real_dataset_registry
        registry.create("support-ticket-qa", pd.DataFrame(_TICKETS), input_keys=["question"], experiment_id=experiment_id)

        registry.delete("support-ticket-qa")

        assert "support-ticket-qa" not in registry

    def test_delete_record_ids_deletes_only_records(self, monkeypatch):
        registry, calls, _ = _make_dataset_registry(monkeypatch)

        registry.delete("support-ticket-qa", record_ids=["r1", "r2"])

        assert calls[-1] == ("delete_records", (["r1", "r2"],), {})
        assert "delete_dataset" not in [name for name, _, _ in calls]

    def test_delete_dataset_tags_deletes_only_tags(self, monkeypatch):
        registry, calls, _ = _make_dataset_registry(monkeypatch)

        registry.delete("support-ticket-qa", dataset_tags=["draft", "wip"])

        assert calls[-2:] == [
            ("delete_dataset_tag", ("d-support-ticket-qa", "draft"), {}),
            ("delete_dataset_tag", ("d-support-ticket-qa", "wip"), {}),
        ]
        assert "delete_dataset" not in [name for name, _, _ in calls]

    def test_list_forwards_filter_and_returns_list(self, monkeypatch):
        registry, calls, fake = _make_dataset_registry(monkeypatch)

        assert registry.list(filter_string="name LIKE 'support%'") == [fake]
        assert calls[-1] == ("search_datasets", (), {"filter_string": "name LIKE 'support%'"})

    def test_to_dataframe_lists_datasets(self, monkeypatch):
        registry, _, _ = _make_dataset_registry(monkeypatch)

        df_datasets = registry.to_dataframe()

        assert df_datasets["name"].tolist() == ["support-ticket-qa"]

    def test_snapshot_round_trip_restores_records_at_snapshot_time(self, real_dataset_registry):
        registry, experiment_id = real_dataset_registry
        df_more = pd.DataFrame({"question": ["How do I update billing?"], "answer": ["Account Settings > Billing."]})
        registry.create(
            "support-ticket-qa", pd.DataFrame(_TICKETS),
            input_keys=["question"], output_keys=["answer"], experiment_id=experiment_id,
        )
        registry.snapshot("support-ticket-qa", "v1")
        registry.extend("support-ticket-qa", df_more, input_keys=["question"], output_keys=["answer"])

        assert len(registry.get("support-ticket-qa").to_df()) == 2
        assert len(registry.load_snapshot("support-ticket-qa", "v1")) == 1

    def test_list_snapshots_returns_labels(self, real_dataset_registry):
        registry, experiment_id = real_dataset_registry
        registry.create("support-ticket-qa", pd.DataFrame(_TICKETS), input_keys=["question"], experiment_id=experiment_id)
        registry.snapshot("support-ticket-qa", "v1")
        registry.snapshot("support-ticket-qa", "v2")

        assert registry.list_snapshots("support-ticket-qa") == ["v1", "v2"]

    def test_load_snapshot_missing_label_raises_valueerror(self, real_dataset_registry):
        registry, experiment_id = real_dataset_registry
        registry.create("support-ticket-qa", pd.DataFrame(_TICKETS), input_keys=["question"], experiment_id=experiment_id)

        with pytest.raises(ValueError):
            registry.load_snapshot("support-ticket-qa", "v9")

    def test_create_description_tag_uses_configured_namespace(self, monkeypatch):
        from velari_ai.integrations.mlflow.registry import DatasetOptions

        registry, calls, _ = _make_dataset_registry(monkeypatch, DatasetOptions(tag_namespace="acme"))

        registry.create("support-ticket-qa", description="Support QA")

        assert calls[-1] == ("create_dataset", (), {"name": "support-ticket-qa", "tags": {"acme.description": "Support QA"}})

    def test_snapshot_round_trip_uses_configured_names(self, make_real_dataset_registry):
        from velari_ai.integrations.mlflow.registry import DatasetOptions

        options = DatasetOptions(tag_namespace="acme", snapshot_prefix="frozen.", snapshot_file="frozen.json")
        registry, experiment_id = make_real_dataset_registry(options)
        registry.create("support-ticket-qa", pd.DataFrame(_TICKETS), input_keys=["question"], experiment_id=experiment_id)

        meta = registry.snapshot("support-ticket-qa", "v1")

        assert all(key.startswith("frozen.v1.") for key in meta)
        assert registry.list_snapshots("support-ticket-qa") == ["v1"]
        assert len(registry.load_snapshot("support-ticket-qa", "v1")) == 1
