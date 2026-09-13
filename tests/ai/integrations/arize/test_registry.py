"""Tests for velari_ai.integrations.arize.registry."""

from typing import Any

import pandas as pd
from omegaconf import OmegaConf


class _FakePrompt:
    def __init__(self, id="prompt-1"):
        self.id = id


class _FakePromptTags:
    def __init__(self):
        self.create_calls = []
        self.list_result = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)

    def list(self, *, prompt_version_id):
        return self.list_result


class _FakePrompts:
    def __init__(self):
        self.create_calls = []
        self.get_calls = []
        self.tags = _FakePromptTags()

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return _FakePrompt()

    def get(self, **kwargs):
        self.get_calls.append(kwargs)
        return _FakePrompt()


class _FakeHttpResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return {"data": self._data}


class _FakeRawClient:
    def __init__(self):
        self.get_calls = []
        self.list_result = []

    def get(self, path, params=None):
        self.get_calls.append({"path": path, "params": params})
        return _FakeHttpResponse(self.list_result)


class _FakeDataset:
    def __init__(self, name, id="dataset-1", version_id="version-1", size=1):
        self.name = name
        self.id = id
        self.version_id = version_id
        self._size = size

    def __len__(self):
        return self._size


class _FakeDatasetsResource:
    def __init__(self):
        self.create_calls = []
        self.get_calls = []
        self.list_calls = []
        self.list_result = []
        self.add_examples_calls = []

    def create_dataset(self, **kwargs):
        self.create_calls.append(kwargs)
        return _FakeDataset(name=kwargs["name"])

    def get_dataset(self, *, dataset, version_id=None):
        self.get_calls.append({"dataset": dataset, "version_id": version_id})
        return _FakeDataset(name=dataset)

    def list(self, **kwargs):
        self.list_calls.append(kwargs)
        return self.list_result

    def add_examples_to_dataset(self, **kwargs):
        self.add_examples_calls.append(kwargs)
        name = kwargs["dataset"] if isinstance(kwargs["dataset"], str) else kwargs["dataset"].name
        return _FakeDataset(name=name, version_id="version-2", size=2)


class _FakeExperiment(dict):
    pass


class _FakeExperimentsResource:
    def __init__(self):
        self.create_calls = []
        self.get_calls = []
        self.list_calls = []
        self.list_result = []
        self.run_experiment_calls = []
        self.run_experiment_result = _FakeExperiment(experiment_id="experiment-1", task_runs=[], evaluation_runs=[])
        self.evaluate_experiment_calls = []
        self.evaluate_experiment_result = _FakeExperiment(experiment_id="experiment-1", evaluation_runs=[])

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return _FakeExperiment(id="experiment-1", name=kwargs.get("experiment_name"))

    def get(self, *, experiment_id):
        self.get_calls.append({"experiment_id": experiment_id})
        return _FakeExperiment(id=experiment_id, name="support-ticket-qa-prompt-v1")

    def list(self, **kwargs):
        self.list_calls.append(kwargs)
        return self.list_result

    def run_experiment(self, **kwargs):
        self.run_experiment_calls.append(kwargs)
        return self.run_experiment_result

    def evaluate_experiment(self, **kwargs):
        self.evaluate_experiment_calls.append(kwargs)
        return self.evaluate_experiment_result


class _FakeClient:
    def __init__(self):
        self.prompts = _FakePrompts()
        self.datasets = _FakeDatasetsResource()
        self.experiments = _FakeExperimentsResource()
        self._client = _FakeRawClient()


def _make_cfg():
    return OmegaConf.create({
        "name": "billing-reminder",
        "model_name": "gpt-4o",
        "messages": [{"role": "system", "content": "You are a billing assistant."}],
    })


def _make_dataframe():
    return pd.DataFrame({"question": ["a"], "answer": ["b"]})


def _make_version_data(id="version-1"):
    return {
        "id": id,
        "template": {
            "type": "chat",
            "messages": [{"role": "system", "content": "You are a billing assistant."}],
        },
        "model_name": "gpt-4o",
        "model_provider": "OPENAI",
        "template_format": "MUSTACHE",
        "invocation_parameters": {"type": "openai", "openai": {}},
    }


class TestPromptRegistry:
    def test_create_passes_required_fields(self):
        from velari_ai.integrations.arize.registry import PromptRegistry

        client: Any = _FakeClient()
        registry = PromptRegistry(client)

        prompt = registry.create(_make_cfg())

        assert prompt.id == "prompt-1"
        call = client.prompts.create_calls[0]
        assert call["name"] == "billing-reminder"

    def test_get_prefers_uuid_over_tag_and_name(self):
        from velari_ai.integrations.arize.registry import PromptRegistry

        client: Any = _FakeClient()
        registry = PromptRegistry(client)

        registry.get("billing-reminder", tag="prod", uuid="version-1")

        assert client.prompts.get_calls[0] == {"prompt_version_id": "version-1"}

    def test_get_falls_back_to_tag_then_name(self):
        from velari_ai.integrations.arize.registry import PromptRegistry

        client: Any = _FakeClient()
        registry = PromptRegistry(client)

        registry.get("billing-reminder", tag="prod")
        registry.get("billing-reminder")

        assert client.prompts.get_calls[0] == {"prompt_identifier": "billing-reminder", "tag": "prod"}
        assert client.prompts.get_calls[1] == {"prompt_identifier": "billing-reminder"}

    def test_list_parses_raw_endpoint_response(self):
        from velari_ai.integrations.arize.registry import PromptRegistry

        client: Any = _FakeClient()
        client._client.list_result = [{"id": "1", "name": "billing-reminder"}]
        registry = PromptRegistry(client)

        result = registry.list()

        assert result == [{"id": "1", "name": "billing-reminder"}]
        assert client._client.get_calls[0]["path"] == "v1/prompts"

    def test_list_returns_tags_when_uuid_given(self):
        from velari_ai.integrations.arize.registry import PromptRegistry

        client: Any = _FakeClient()
        client.prompts.tags.list_result = [{"id": "tag-1", "name": "prod"}]
        registry = PromptRegistry(client)

        result = registry.list(uuid="version-1")

        assert result == [{"id": "tag-1", "name": "prod"}]
        assert client._client.get_calls == []

    def test_get_versions_true_returns_all_prompt_version_objects(self):
        from phoenix.client.types.prompts import PromptVersion
        from velari_ai.integrations.arize.registry import PromptRegistry

        client: Any = _FakeClient()
        client._client.list_result = [_make_version_data("version-1"), _make_version_data("version-2")]
        registry = PromptRegistry(client)

        result: Any = registry.get(name="billing-reminder", versions=True)

        assert len(result) == 2
        assert all(isinstance(v, PromptVersion) for v in result)
        assert result[0].id == "version-1"
        assert result[1].id == "version-2"
        assert client._client.get_calls[0]["path"] == "v1/prompts/billing-reminder/versions"

    def test_to_dataframe_builds_from_list(self):
        from velari_ai.integrations.arize.registry import PromptRegistry

        client: Any = _FakeClient()
        client._client.list_result = [
            {"id": "1", "name": "billing-reminder"},
            {"id": "2", "name": "support-qa"},
        ]
        registry = PromptRegistry(client)

        df = registry.to_dataframe()

        assert list(df["name"]) == ["billing-reminder", "support-qa"]


class TestDatasetRegistry:
    def test_create_passes_required_fields(self):
        from velari_ai.integrations.arize.registry import DatasetRegistry

        client: Any = _FakeClient()
        registry = DatasetRegistry(client)
        df = _make_dataframe()

        dataset = registry.create("qa-dataset", df, input_keys=["question"])

        assert dataset.name == "qa-dataset"
        call = client.datasets.create_calls[0]
        assert call["name"] == "qa-dataset"
        assert call["input_keys"] == ["question"]
        assert "output_keys" not in call
        assert "dataset_description" not in call

    def test_create_includes_output_keys_and_description_when_given(self):
        from velari_ai.integrations.arize.registry import DatasetRegistry

        client: Any = _FakeClient()
        registry = DatasetRegistry(client)
        df = _make_dataframe()

        registry.create(
            "qa-dataset",
            df,
            input_keys=["question"],
            output_keys=["answer"],
            description="support ticket QA pairs",
        )

        call = client.datasets.create_calls[0]
        assert call["output_keys"] == ["answer"]
        assert call["dataset_description"] == "support ticket QA pairs"

    def test_add_examples_passes_required_fields(self):
        from velari_ai.integrations.arize.registry import DatasetRegistry

        client: Any = _FakeClient()
        registry = DatasetRegistry(client)
        df = _make_dataframe()

        dataset = registry.add_examples("qa-dataset", df, input_keys=["question"])

        assert dataset.name == "qa-dataset"
        assert dataset.version_id == "version-2"
        call = client.datasets.add_examples_calls[0]
        assert call["dataset"] == "qa-dataset"
        assert call["input_keys"] == ["question"]
        assert "output_keys" not in call

    def test_add_examples_forwards_dataset_object(self):
        from velari_ai.integrations.arize.registry import DatasetRegistry

        client: Any = _FakeClient()
        registry = DatasetRegistry(client)
        existing: Any = _FakeDataset(name="qa-dataset")
        df = _make_dataframe()

        registry.add_examples(existing, df, input_keys=["question"])

        call = client.datasets.add_examples_calls[0]
        assert call["dataset"] is existing

    def test_add_examples_includes_output_keys_when_given(self):
        from velari_ai.integrations.arize.registry import DatasetRegistry

        client: Any = _FakeClient()
        registry = DatasetRegistry(client)
        df = _make_dataframe()

        registry.add_examples("qa-dataset", df, input_keys=["question"], output_keys=["answer"])

        call = client.datasets.add_examples_calls[0]
        assert call["output_keys"] == ["answer"]

    def test_get_defaults_version_id_to_none(self):
        from velari_ai.integrations.arize.registry import DatasetRegistry

        client: Any = _FakeClient()
        registry = DatasetRegistry(client)

        dataset = registry.get("qa-dataset")

        assert dataset.name == "qa-dataset"
        assert client.datasets.get_calls[0]["version_id"] is None

    def test_get_forwards_version_id(self):
        from velari_ai.integrations.arize.registry import DatasetRegistry

        client: Any = _FakeClient()
        registry = DatasetRegistry(client)

        registry.get("qa-dataset", version_id="version-2")

        assert client.datasets.get_calls[0]["version_id"] == "version-2"

    def test_list_returns_underlying_datasets(self):
        from velari_ai.integrations.arize.registry import DatasetRegistry

        client: Any = _FakeClient()
        client.datasets.list_result = [{"id": "1", "name": "qa-dataset"}]
        registry = DatasetRegistry(client)

        assert registry.list() == [{"id": "1", "name": "qa-dataset"}]

    def test_has_and_len_work_through_inheritance(self):
        from velari_ai.integrations.arize.registry import DatasetRegistry

        client: Any = _FakeClient()
        client.datasets.list_result = [{"id": "1", "name": "qa-dataset"}]
        registry = DatasetRegistry(client)

        assert registry.has("qa-dataset") is True
        assert "qa-dataset" in registry
        assert len(registry) == 1

    def test_list_forwards_kwargs_to_client(self):
        from velari_ai.integrations.arize.registry import DatasetRegistry

        client: Any = _FakeClient()
        registry = DatasetRegistry(client)

        registry.list(limit=5)

        assert client.datasets.list_calls[0] == {"limit": 5}

    def test_to_dataframe_builds_from_list(self):
        from velari_ai.integrations.arize.registry import DatasetRegistry

        client: Any = _FakeClient()
        client.datasets.list_result = [
            {"id": "1", "name": "qa-dataset", "example_count": 2},
            {"id": "2", "name": "churn-eval", "example_count": 5},
        ]
        registry = DatasetRegistry(client)

        df = registry.to_dataframe()

        assert list(df["name"]) == ["qa-dataset", "churn-eval"]
        assert list(df["example_count"]) == [2, 5]

    def test_has_checks_against_list_without_calling_get(self):
        from velari_ai.integrations.arize.registry import DatasetRegistry

        client: Any = _FakeClient()
        client.datasets.list_result = [{"id": "1", "name": "qa-dataset"}]

        def _fail_if_called(**kwargs):
            raise AssertionError("get_dataset must not be called by has()")

        client.datasets.get_dataset = _fail_if_called
        registry = DatasetRegistry(client)

        assert registry.has("qa-dataset") is True
        assert registry.has("missing-dataset") is False


class TestExperimentsRegistry:
    def test_create_passes_required_fields(self):
        from velari_ai.integrations.arize.registry import ExperimentsRegistry

        client: Any = _FakeClient()
        registry = ExperimentsRegistry(client)

        experiment = registry.create(dataset_id="dataset-1")

        assert experiment["id"] == "experiment-1"
        call = client.experiments.create_calls[0]
        assert call["dataset_id"] == "dataset-1"
        assert call["repetitions"] == 1
        assert "experiment_name" not in call
        assert "dataset_version_id" not in call

    def test_create_includes_optional_fields_when_given(self):
        from velari_ai.integrations.arize.registry import ExperimentsRegistry

        client: Any = _FakeClient()
        registry = ExperimentsRegistry(client)

        registry.create(
            dataset_id="dataset-1",
            dataset_version_id="version-2",
            experiment_name="support-ticket-qa-prompt-v2",
            experiment_description="prompt v2 comparison",
            experiment_metadata={"prompt_version": "v2"},
            splits=["holdout"],
            repetitions=3,
        )

        call = client.experiments.create_calls[0]
        assert call["dataset_version_id"] == "version-2"
        assert call["experiment_name"] == "support-ticket-qa-prompt-v2"
        assert call["experiment_description"] == "prompt v2 comparison"
        assert call["experiment_metadata"] == {"prompt_version": "v2"}
        assert call["splits"] == ["holdout"]
        assert call["repetitions"] == 3

    def test_get_forwards_experiment_id(self):
        from velari_ai.integrations.arize.registry import ExperimentsRegistry

        client: Any = _FakeClient()
        registry = ExperimentsRegistry(client)

        experiment = registry.get("experiment-1")

        assert experiment["id"] == "experiment-1"
        assert client.experiments.get_calls[0] == {"experiment_id": "experiment-1"}

    def test_list_forwards_dataset_id(self):
        from velari_ai.integrations.arize.registry import ExperimentsRegistry

        client: Any = _FakeClient()
        client.experiments.list_result = [{"id": "experiment-1", "name": "support-ticket-qa-prompt-v1"}]
        registry = ExperimentsRegistry(client)

        result = registry.list(dataset_id="dataset-1")

        assert result == [{"id": "experiment-1", "name": "support-ticket-qa-prompt-v1"}]
        assert client.experiments.list_calls[0] == {"dataset_id": "dataset-1"}

    def test_list_raises_without_dataset_id(self):
        import pytest
        from velari_ai.integrations.arize.registry import ExperimentsRegistry

        client: Any = _FakeClient()
        registry = ExperimentsRegistry(client)

        with pytest.raises(ValueError):
            registry.list()

    def test_to_dataframe_builds_from_list(self):
        from velari_ai.integrations.arize.registry import ExperimentsRegistry

        client: Any = _FakeClient()
        client.experiments.list_result = [
            {"id": "experiment-1", "name": "prompt-v1", "successful_run_count": 8},
            {"id": "experiment-2", "name": "prompt-v2", "successful_run_count": 10},
        ]
        registry = ExperimentsRegistry(client)

        df = registry.to_dataframe(dataset_id="dataset-1")

        assert list(df["name"]) == ["prompt-v1", "prompt-v2"]
        assert client.experiments.list_calls[0] == {"dataset_id": "dataset-1"}

    def test_make_task_binds_extra_args_and_forwards_example_input(self):
        from velari_ai.integrations.arize.registry import ExperimentsRegistry

        calls = []

        def answer_question(*, input, model):
            calls.append({"input": input, "model": model})
            return f"answer for {input}"

        task = ExperimentsRegistry.make_task(answer_question, model="gpt-4o")
        example: Any = {"question": "What's our refund policy?"}
        result = task(example)

        assert result == "answer for {'question': \"What's our refund policy?\"}"
        assert calls[0] == {"input": {"question": "What's our refund policy?"}, "model": "gpt-4o"}

    def test_make_task_result_is_callable_the_way_phoenix_invokes_a_single_arg_task(self):
        import inspect
        from velari_ai.integrations.arize.registry import ExperimentsRegistry

        def echo(*, input):
            return input

        task = ExperimentsRegistry.make_task(echo)

        # mirrors Phoenix's own single-parameter task binding (_bind_task_signature):
        # `sig.bind(value)` — a single positional value, regardless of the parameter's name.
        sig = inspect.signature(task)
        bound = sig.bind("What's our refund policy?")

        assert task(*bound.args, **bound.kwargs) == "What's our refund policy?"

    def test_run_experiment_forwards_dataset_task_and_evaluators(self):
        from velari_ai.integrations.arize.registry import ExperimentsRegistry

        client: Any = _FakeClient()
        registry = ExperimentsRegistry(client)
        dataset: Any = object()
        task = lambda input: input  # noqa: E731
        evaluators = [lambda output: True]

        result = registry.run_experiment(dataset, task, evaluators=evaluators)

        assert result["experiment_id"] == "experiment-1"
        call = client.experiments.run_experiment_calls[0]
        assert call["dataset"] is dataset
        assert call["task"] is task
        assert call["evaluators"] is evaluators

    def test_run_experiment_uses_default_options_when_omitted(self):
        from velari_ai.integrations.arize.registry import ExperimentsRegistry

        client: Any = _FakeClient()
        registry = ExperimentsRegistry(client)
        dataset: Any = object()

        registry.run_experiment(dataset, lambda input: input)

        call = client.experiments.run_experiment_calls[0]
        assert call["experiment_name"] is None
        assert call["repetitions"] == 1
        assert call["dry_run"] is False
        assert call["timeout"] == 60
        assert call["retries"] == 3
        assert call["rate_limit_errors"] is None
        assert call["print_summary"] is True

    def test_run_experiment_threads_through_options(self):
        from velari_ai.integrations.arize.registry import ExperimentsRegistry, ExperimentOptions

        client: Any = _FakeClient()
        registry = ExperimentsRegistry(client)
        options = ExperimentOptions(
            identity=ExperimentOptions.Identity(
                name="prompt-v2", description="prompt v2 comparison", metadata={"prompt_version": "v2"}, repetitions=3,
            ),
            execution=ExperimentOptions.Execution(dry_run=5, timeout=120, retries=1),
            verbose=False,
        )
        dataset: Any = object()

        registry.run_experiment(dataset, lambda input: input, options=options)

        call = client.experiments.run_experiment_calls[0]
        assert call["experiment_name"] == "prompt-v2"
        assert call["experiment_description"] == "prompt v2 comparison"
        assert call["experiment_metadata"] == {"prompt_version": "v2"}
        assert call["repetitions"] == 3
        assert call["dry_run"] == 5
        assert call["timeout"] == 120
        assert call["retries"] == 1
        assert call["print_summary"] is False
        assert call["experiment_name"] == "prompt-v2"

    def test_evaluate_experiment_forwards_experiment_and_evaluators(self):
        from velari_ai.integrations.arize.registry import ExperimentsRegistry

        client: Any = _FakeClient()
        registry = ExperimentsRegistry(client)
        experiment: Any = _FakeExperiment(experiment_id="experiment-1")
        evaluators = [lambda output: True]

        result = registry.evaluate_experiment(experiment, evaluators=evaluators)

        assert result["experiment_id"] == "experiment-1"
        call = client.experiments.evaluate_experiment_calls[0]
        assert call["experiment"] is experiment
        assert call["evaluators"] is evaluators
        assert call["dry_run"] is False
        assert call["print_summary"] is True

    def test_evaluate_experiment_threads_through_options(self):
        from velari_ai.integrations.arize.registry import ExperimentsRegistry, ExperimentOptions

        client: Any = _FakeClient()
        registry = ExperimentsRegistry(client)
        experiment: Any = _FakeExperiment(experiment_id="experiment-1")
        options = ExperimentOptions(execution=ExperimentOptions.Execution(dry_run=True, timeout=30, retries=2), verbose=False)

        registry.evaluate_experiment(experiment, evaluators=[lambda output: True], options=options)

        call = client.experiments.evaluate_experiment_calls[0]
        assert call["dry_run"] is True
        assert call["timeout"] == 30
        assert call["retries"] == 2
        assert call["print_summary"] is False

    def test_evaluate_experiment_coerces_int_dry_run_to_bool(self):
        from velari_ai.integrations.arize.registry import ExperimentsRegistry, ExperimentOptions

        client: Any = _FakeClient()
        registry = ExperimentsRegistry(client)
        experiment: Any = _FakeExperiment(experiment_id="experiment-1")
        options = ExperimentOptions(execution=ExperimentOptions.Execution(dry_run=5))

        registry.evaluate_experiment(experiment, evaluators=[lambda output: True], options=options)

        call = client.experiments.evaluate_experiment_calls[0]
        assert call["dry_run"] is True
