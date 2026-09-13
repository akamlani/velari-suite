import  logging
from    dataclasses import dataclass, field
from    typing import Any, Callable, Dict, List, Optional, Union

import  pandas as pd
from    omegaconf import DictConfig
from    phoenix.client import Client
from    phoenix.client.__generated__.v1 import Dataset as DatasetSummary, Experiment, PromptMessage, PromptVersionTag
from    phoenix.client.resources.datasets import Dataset
from    phoenix.client.resources.experiments.types import ExperimentEvaluators, ExperimentTask, RanExperiment, RateLimitErrors
from    phoenix.client.types.prompts import PromptVersion
# package modules
from    velari_data.registry import Registry

logger = logging.getLogger(__name__)


class PromptRegistry(Registry):
    def __init__(self, client: Client) -> None:
        self._client = client

    def create(self, cfg: DictConfig, **kwargs) -> PromptVersion:
        messages: List[PromptMessage] = [{"role": m.role, "content": m.content} for m in cfg.messages]
        pv_optional = ["model_provider", "template_format", "description"]
        version     = PromptVersion(
            messages,
            model_name = cfg.model_name,
            **{k: cfg[k] for k in pv_optional if k in cfg},
        )
        cr_optional = {"metadata": "prompt_metadata", "prompt_description": "prompt_description"}
        cr_kwargs: Dict[str, Any] = dict(
            name    = cfg.name,
            version = version,
            **{dst: cfg[src] for src, dst in cr_optional.items() if src in cfg},
        )
        prompt = self._client.prompts.create(**cr_kwargs)
        logger.info(f"Created prompt '{cfg.name}' — id={prompt.id}")
        return prompt

    def create_tag(self, uuid: str, name: str, description: Optional[str] = None) -> None:
        kwargs: Dict[str, Any] = dict(prompt_version_id=uuid, name=name)
        if description:
            kwargs["description"] = description
        self._client.prompts.tags.create(**kwargs)
        logger.info(f"Tagged version '{uuid}' as '{name}'")

    def _list_tags(self, uuid: str) -> List[PromptVersionTag]:
        result = self._client.prompts.tags.list(prompt_version_id=uuid)
        logger.info(f"Found {len(result)} tags for version '{uuid}'")
        return result

    def get(
        self,
        name: str,
        tag:      Optional[str] = None,
        uuid:     Optional[str] = None,
        versions: bool          = False,
        limit:    int           = 100,
        **kwargs: Any,
    ) -> Union[PromptVersion, List[PromptVersion]]:
        """Retrieve a prompt version, or all versions of a prompt.

        Precedence for a single version: uuid > tag > latest.

        Args:
            name (str): Prompt name.
            tag (Optional[str]): Retrieve the version tagged with this name.
            uuid (Optional[str]): Retrieve this specific prompt version ID.
            versions (bool): When True, return every version instead of one (delegates to
                `_get_versions(name, limit)`).
            limit (int): Maximum versions to return; only used when `versions=True`.

        Returns:
            Union[PromptVersion, List[PromptVersion]]: A single version, or all versions.

        Examples:
            >>> registry = PromptRegistry(client)
            >>> registry.get("billing-reminder")                  # latest version
            >>> registry.get("billing-reminder", tag="prod")      # tagged version
            >>> registry.get("billing-reminder", versions=True)   # every version
        """
        if versions:
            return self._get_versions(name, limit=limit)
        if uuid:
            prompt = self._client.prompts.get(prompt_version_id=uuid)
        elif tag:
            prompt = self._client.prompts.get(prompt_identifier=name, tag=tag)
        else:
            prompt = self._client.prompts.get(prompt_identifier=name)
        logger.info(f"Retrieved prompt '{name}' — id={prompt.id}")
        return prompt

    def _get_versions(self, name: str, limit: int = 100) -> List[PromptVersion]:
        try:
            response = self._client._client.get(
                f"v1/prompts/{name}/versions",
                params={"limit": limit},
            )
            response.raise_for_status()
            data = response.json().get("data", [])
            # PromptVersion._loads reconstructs a full object from the same raw shape this
            # endpoint returns — avoids an extra round-trip per version.
            versions = [PromptVersion._loads(v) for v in data]
            logger.info(f"Found {len(versions)} versions for prompt '{name}'")
            return versions
        except Exception as e:
            logger.error(f"Failed to list versions for prompt '{name}': {e}")
            raise

    def list(self, uuid: Optional[str] = None, limit: int = 100, **kwargs: Any) -> List[Dict[str, Any]]:
        """List all prompts, or a specific prompt version's tags.

        Not exposed by the `Prompts` client resource (only `create`/`get`/`update` are) — the
        all-prompts case reaches the server's `GET /v1/prompts` route directly, same raw-HTTP
        pattern as `_get_versions()`.

        Args:
            uuid (Optional[str]): When given, returns that version's tags instead of listing all
                prompts (delegates to `_list_tags(uuid)`).
            limit (int): Maximum number of prompts to return; ignored when `uuid` is given.

        Returns:
            List[Dict[str, Any]]: Prompt summaries, or that version's tags if `uuid` is given.

        Examples:
            >>> registry = PromptRegistry(client)
            >>> registry.list()                    # all prompts
            >>> registry.list(uuid="version-1")     # that version's tags
        """
        if uuid is not None:
            return [dict(tag) for tag in self._list_tags(uuid)]
        response = self._client._client.get("v1/prompts", params={"limit": limit})
        response.raise_for_status()
        data = response.json().get("data", [])
        logger.info(f"Found {len(data)} prompts")
        return data

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the connected Phoenix instance's prompt listing to a DataFrame.

        Overrides the base `Registry.to_dataframe()` (which reads a local `_catalog` this class
        never populates) — built directly from `list()` instead.

        Returns:
            pd.DataFrame: One row per prompt (id/name/description/source_prompt_id/metadata).
        """
        return pd.DataFrame(self.list())


class DatasetRegistry(Registry):
    """Create, retrieve, and list Arize Phoenix evaluation datasets.

    Args:
        client (Client): Already-connected Phoenix client (e.g. `connection.Connector.client`).
    """
    def __init__(self, client: Client) -> None:
        self._client = client

    def create(
        self,
        name: str,
        dataframe:   pd.DataFrame,
        input_keys:  List[str],
        output_keys: Optional[List[str]] = None,
        description: Optional[str] = None,
        **kwargs: Any,
    ) -> Dataset:
        """Upload a dataframe as a new Phoenix dataset.

        Args:
            name (str): Dataset name, shown in the Phoenix UI.
            dataframe (pd.DataFrame): Rows to upload; columns referenced by `input_keys`/
                `output_keys` become the dataset's example fields.
            input_keys (List[str]): Columns holding example inputs.
            output_keys (Optional[List[str]]): Columns holding expected outputs. Omit when the
                dataframe has no target yet — just inputs.
            description (Optional[str]): Human-readable dataset description.

        Returns:
            Dataset: The created dataset, with its assigned `id`/`version_id`.

        Examples:
            # fields: {'context', 'generated', 'output'}
            >>> registry = DatasetRegistry(client)
            >>> df_support_tickets = pd.DataFrame({
            ...     "question": ["How do I reset my password?"],
            ...     "answer":   ["Visit /account/reset and follow the emailed link."],
            ... })
            >>> dataset = registry.create(
            ...     "support-ticket-qa",
            ...     df_support_tickets,
            ...     input_keys=["question"],
            ...     output_keys=["answer"],
            ... )
            # Output
            >>> dataset.name            # The name of the created dataset
            >>> dataset.version_id      # The version ID of the created dataset
            >>> dataset.example_count   # The number of examples in this version

            >>> # No target/expected-output column yet — queries collected before ground-truth
            >>> # answers exist. Omit output_keys and pass only input_keys.
            >>> df_queries = pd.DataFrame({"question": ["What's our refund policy?"]})
            >>> dataset = registry.create("support-ticket-queries", df_queries, input_keys=["question"])
        """
        create_kwargs: Dict[str, Any] = dict(
            name       = name,
            dataframe  = dataframe,
            input_keys = input_keys,
        )
        if output_keys is not None:
            create_kwargs["output_keys"] = output_keys
        if description is not None:
            create_kwargs["dataset_description"] = description
        dataset = self._client.datasets.create_dataset(**create_kwargs)
        logger.info(f"Created dataset '{dataset.name}' — id={dataset.id}, version={dataset.version_id}, {len(dataset)} examples")
        logger.info(f"Columns: {dataframe.columns.tolist()}")
        logger.info(f"Sample:\n{dataframe.head().to_string()}")
        return dataset

    def add_examples(
        self,
        dataset: Union[str, Dataset, Dict[str, Any]],
        dataframe:   pd.DataFrame,
        input_keys:  List[str],
        output_keys: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dataset:
        """Append examples to an existing Phoenix dataset.

        Args:
            dataset (Union[str, Dataset, Dict[str, Any]]): Dataset to append to — a name, a
                `Dataset` object (e.g. from `create()`/`get()`), or a dict with `id`/`name`.
            dataframe (pd.DataFrame): Rows to append; columns referenced by `input_keys`/
                `output_keys` become the new examples' fields.
            input_keys (List[str]): Columns holding example inputs.
            output_keys (Optional[List[str]]): Columns holding expected outputs. Omit when the
                dataframe has no target yet — just inputs.
            **kwargs (Any): Forwarded to Phoenix's `Datasets.add_examples_to_dataset()` (e.g.
                `metadata_keys`, `split_key`, `example_id_key`).

        Returns:
            Dataset: The updated dataset, with the new examples included in a new version.

        Examples:
            >>> registry = DatasetRegistry(client)
            >>> dataset = registry.create(
            ...     "support-ticket-qa",
            ...     pd.DataFrame({"question": ["How do I reset my password?"], "answer": ["Visit /account/reset."]}),
            ...     input_keys=["question"],
            ...     output_keys=["answer"],
            ... )
            >>> df_more_tickets = pd.DataFrame({
            ...     "question": ["How do I update my billing address?"],
            ...     "answer":   ["Go to Account Settings > Billing and edit your address."],
            ... })
            >>> dataset = registry.add_examples(dataset, df_more_tickets, input_keys=["question"], output_keys=["answer"])
        """
        add_kwargs: Dict[str, Any] = dict(
            dataset    = dataset,
            dataframe  = dataframe,
            input_keys = input_keys,
        )
        if output_keys is not None:
            add_kwargs["output_keys"] = output_keys
        updated = self._client.datasets.add_examples_to_dataset(**add_kwargs, **kwargs)
        logger.info(f"Added examples to dataset '{updated.name}' — id={updated.id}, version={updated.version_id}, {len(updated)} examples")
        return updated

    def get(self, name: str, version_id: Optional[str] = None) -> Dataset:
        """Retrieve a dataset by name.

        Args:
            name (str): Dataset name.
            version_id (Optional[str]): Specific version to retrieve; defaults to the latest.

        Returns:
            Dataset: The matching dataset.

        Examples:
            >>> registry = DatasetRegistry(client)
            >>> dataset = registry.get("support-ticket-qa")
        """
        return self._client.datasets.get_dataset(dataset=name, version_id=version_id)

    def list(self, **kwargs: Any) -> List[DatasetSummary]:
        """List all datasets on the connected Phoenix instance.

        Args:
            **kwargs (Any): Forwarded to Phoenix's `Datasets.list()` (e.g. `limit`, `timeout`).

        Returns:
            List[DatasetSummary]: Dataset summaries (id/name/description/metadata/example_count) —
                a lighter-weight type than `create()`/`get()`'s full `Dataset`, without examples.

        Examples:
            >>> registry = DatasetRegistry(client)
            >>> [dataset["name"] for dataset in registry.list(limit=10)]
            ['support-ticket-qa', 'churn-analysis-eval']
        """
        return self._client.datasets.list(**kwargs)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the connected Phoenix instance's dataset listing to a DataFrame.

        Overrides the base `Registry.to_dataframe()` (which reads a local `_catalog` this class
        never populates) — built directly from `list()` instead, since Phoenix already returns
        dataset summaries in a DataFrame-ready shape.

        Returns:
            pd.DataFrame: One row per dataset (id/name/description/metadata/example_count/...).

        Examples:
            >>> registry = DatasetRegistry(client)
            >>> registry.to_dataframe()[["name", "example_count"]]
        """
        return pd.DataFrame(self.list())

    def has(self, name: str) -> bool:
        """Check whether a dataset with the given name exists.

        Overrides the base `Registry.has()` (which calls `get()` and catches failure) — checks
        against `list()`'s summaries instead, avoiding a full dataset+examples fetch just to test
        existence.

        Args:
            name (str): Dataset name to check for.

        Returns:
            bool: True if a dataset with this name exists.

        Examples:
            >>> registry = DatasetRegistry(client)
            >>> registry.has("support-ticket-qa")
            True
        """
        return any(dataset["name"] == name for dataset in self.list())


@dataclass
class ExperimentOptions:
    """Optional settings for `ExperimentsRegistry.run_experiment()`/`evaluate_experiment()`.

    `identity` only applies to `run_experiment()` (defines a new experiment); `execution` and
    `verbose` are shared by both — `evaluate_experiment()` ignores `identity` entirely.
    """
    @dataclass
    class Identity:
        name:        Optional[str]             = field(default=None)
        description: Optional[str]             = field(default=None)
        metadata:    Optional[Dict[str, Any]]  = field(default=None)
        repetitions: int                       = field(default=1)

        def to_kwargs(self) -> Dict[str, Any]:
            return dict(
                experiment_name        = self.name,
                experiment_description = self.description,
                experiment_metadata    = self.metadata,
                repetitions            = self.repetitions,
            )

    @dataclass
    class Execution:
        dry_run:           Union[bool, int]           = field(default=False)
        timeout:           Optional[int]              = field(default=60)
        retries:           int                        = field(default=3)
        rate_limit_errors: Optional[RateLimitErrors]  = field(default=None)

        def to_kwargs(self) -> Dict[str, Any]:
            return dict(
                dry_run           = self.dry_run,
                timeout           = self.timeout,
                retries           = self.retries,
                rate_limit_errors = self.rate_limit_errors,
            )

    identity:  Identity  = field(default_factory=Identity)
    execution: Execution = field(default_factory=Execution)
    verbose:   bool      = field(default=True)


class ExperimentsRegistry(Registry):
    """Create, run, and evaluate Arize Phoenix experiments against a dataset.

    Args:
        client (Client): Already-connected Phoenix client (e.g. `connection.Connector.client`).
    """
    def __init__(self, client: Client) -> None:
        self._client = client

    def create(
        self,
        dataset_id: str,
        dataset_version_id:     Optional[str]           = None,
        experiment_name:        Optional[str]           = None,
        experiment_description: Optional[str]           = None,
        experiment_metadata:    Optional[Dict[str, Any]] = None,
        splits:                 Optional[List[str]]      = None,
        repetitions: int = 1,
        **kwargs: Any,
    ) -> Experiment:
        """Create a new, not-yet-run experiment against a dataset.

        Args:
            dataset_id (str): Dataset to run the experiment against.
            dataset_version_id (Optional[str]): Specific dataset version; defaults to the latest.
            experiment_name (Optional[str]): Experiment name, shown in the Phoenix UI.
            experiment_description (Optional[str]): Human-readable experiment description.
            experiment_metadata (Optional[Dict[str, Any]]): Metadata to associate with the experiment.
            splits (Optional[List[str]]): Named dataset splits to restrict the experiment to.
            repetitions (int): Number of times to run the task per example.

        Returns:
            Experiment: The created experiment, with its assigned `id`.

        Examples:
            >>> registry = ExperimentsRegistry(client)
            >>> experiment = registry.create(
            ...     dataset_id="RGF0YXNldDox",
            ...     experiment_name="support-ticket-qa-prompt-v2",
            ...     experiment_metadata={"prompt_version": "v2"},
            ... )
        """
        create_kwargs: Dict[str, Any] = dict(dataset_id=dataset_id, repetitions=repetitions)
        if dataset_version_id is not None:
            create_kwargs["dataset_version_id"] = dataset_version_id
        if experiment_name is not None:
            create_kwargs["experiment_name"] = experiment_name
        if experiment_description is not None:
            create_kwargs["experiment_description"] = experiment_description
        if experiment_metadata is not None:
            create_kwargs["experiment_metadata"] = experiment_metadata
        if splits is not None:
            create_kwargs["splits"] = splits
        experiment = self._client.experiments.create(**create_kwargs, **kwargs)
        logger.info(f"Created experiment '{experiment['name']}' — id={experiment['id']}, dataset_id={dataset_id}")
        return experiment

    def get(self, experiment_id: str) -> Experiment:
        """Retrieve an experiment by id.

        Args:
            experiment_id (str): Experiment id. Experiments have no name-based lookup — unlike
                `DatasetRegistry`/`PromptRegistry`, only `id` identifies one.

        Returns:
            Experiment: The matching experiment.

        Examples:
            >>> registry = ExperimentsRegistry(client)
            >>> experiment = registry.get("RXhwZXJpbWVudDox")
        """
        return self._client.experiments.get(experiment_id=experiment_id)

    def list(self, dataset_id: Optional[str] = None, **kwargs: Any) -> List[Experiment]:
        """List all experiments run against a dataset.

        Args:
            dataset_id (Optional[str]): Dataset to list experiments for — required; experiments
                are always dataset-scoped in Phoenix, unlike prompts/datasets.
            **kwargs (Any): Forwarded to Phoenix's `Experiments.list()` (e.g. `timeout`).

        Returns:
            List[Experiment]: Experiment summaries for the given dataset.

        Raises:
            ValueError: If `dataset_id` is omitted.

        Examples:
            >>> registry = ExperimentsRegistry(client)
            >>> [experiment["name"] for experiment in registry.list(dataset_id="RGF0YXNldDox")]
            ['support-ticket-qa-prompt-v1', 'support-ticket-qa-prompt-v2']
        """
        if dataset_id is None:
            raise ValueError("list() requires dataset_id — Phoenix experiments are always scoped to a dataset")
        return self._client.experiments.list(dataset_id=dataset_id, **kwargs)

    def to_dataframe(self, dataset_id: Optional[str] = None) -> pd.DataFrame:
        """Convert a dataset's experiment listing to a DataFrame.

        Overrides the base `Registry.to_dataframe()` (which reads a local `_catalog` this class
        never populates) — built directly from `list()` instead.

        Args:
            dataset_id (Optional[str]): Dataset to list experiments for; forwarded to `list()`.

        Returns:
            pd.DataFrame: One row per experiment (id/name/repetitions/example_count/...).

        Examples:
            >>> registry = ExperimentsRegistry(client)
            >>> registry.to_dataframe(dataset_id="RGF0YXNldDox")[["name", "successful_run_count"]]
        """
        return pd.DataFrame(self.list(dataset_id=dataset_id))

    @staticmethod
    def make_task(func: Callable[..., Any], *args: Any, **kwargs: Any) -> ExperimentTask:
        """Bind extra fixed args/kwargs to `func`, producing a task for `run_experiment()`.

        Phoenix calls the returned task once per dataset example, passing that example's `input`
        field as this wrapper's sole argument — `func` itself must accept it as an `input` keyword.

        Args:
            func (Callable[..., Any]): Function to run per example.
            *args (Any): Additional positional arguments passed to `func` on every call.
            **kwargs (Any): Additional keyword arguments passed to `func` on every call.

        Returns:
            ExperimentTask: A single-argument callable suitable for `run_experiment()`'s `task=`.

        Examples:
            >>> def answer_question(*, input, model):
            ...     return llm_client.invoke(input["question"], model=model)
            >>> registry = ExperimentsRegistry(client)
            >>> task = registry.make_task(answer_question, model="gpt-4o")
            >>> result = registry.run_experiment(dataset, task, experiment_name="prompt-v2")
        """
        def _task(example_input: Any) -> Any:
            ### Retrieve Output that will be evaluatged
            # query = example_input.get("query")
            # response = func(*args, input=example_input, **kwargs)
            # return response.content
            return func(*args, input=example_input, **kwargs)
        return _task

    def run_experiment(
        self,
        dataset: Dataset,
        task: ExperimentTask,
        evaluators: Optional[ExperimentEvaluators] = None,
        options: Optional[ExperimentOptions] = None,
        **kwargs: Any,
    ) -> RanExperiment:
        """Run a task against every example in a dataset, optionally scoring it with evaluators.

        Args:
            dataset (Dataset): Dataset to run the task against — e.g. from `DatasetRegistry.get()`.
            task (ExperimentTask): Callable run per example; receives the example's `input` (or
                named args like `input`/`expected`/`metadata`/`example`) and returns JSON-serializable
                output.
            evaluators (Optional[ExperimentEvaluators]): Evaluator(s) scoring each task output —
                a dict of name to evaluator, a list, or a single evaluator.
            options (Optional[ExperimentOptions]): Experiment name/description/metadata/repetitions
                (`.identity`), execution controls (`.execution`) and `.verbose`; defaults applied
                when omitted.
            **kwargs (Any): Forwarded to Phoenix's `Experiments.run_experiment()` directly, for
                anything not covered by `options`.

        Returns:
            RanExperiment: The completed experiment, with task runs and any evaluation runs.

        Examples:
            >>> registry = ExperimentsRegistry(client)
            >>> dataset = DatasetRegistry(client).get("support-ticket-qa")
            >>> def answer_question(input):
            ...     return llm.invoke(input["question"]).content
            >>> def is_correct(output, expected):
            ...     return output.strip() == expected["answer"].strip()
            >>> options = ExperimentOptions(identity=ExperimentOptions.Identity(name="prompt-v2", repetitions=3))
            >>> result = registry.run_experiment(dataset, answer_question, evaluators=[is_correct], options=options)
        """
        options = options or ExperimentOptions()
        result = self._client.experiments.run_experiment(
            dataset=dataset,
            task=task,
            evaluators=evaluators,
            **options.identity.to_kwargs(),
            **options.execution.to_kwargs(),
            print_summary=options.verbose,
            **kwargs,
        )
        logger.info(
            f"Ran experiment '{result['experiment_id']}' — {len(result['task_runs'])} runs, "
            f"{len(result['evaluation_runs'])} evaluations"
        )
        return result

    def evaluate_experiment(
        self,
        experiment: RanExperiment,
        evaluators: ExperimentEvaluators,
        options: Optional[ExperimentOptions] = None,
        **kwargs: Any,
    ) -> RanExperiment:
        """Run evaluators against an already-completed experiment.

        Args:
            experiment (RanExperiment): Completed experiment to evaluate, from `run_experiment()`.
            evaluators (ExperimentEvaluators): Evaluator(s) scoring each task output — a dict of
                name to evaluator, a list, or a single evaluator.
            options (Optional[ExperimentOptions]): Execution controls (`.execution`) and
                `.verbose`; `.identity` is ignored (only meaningful for `run_experiment()`).
            **kwargs (Any): Forwarded to Phoenix's `Experiments.evaluate_experiment()` directly,
                for anything not covered by `options`.

        Returns:
            RanExperiment: The experiment with the new evaluation results merged in.

        Examples:
            >>> registry = ExperimentsRegistry(client)
            >>> result = registry.run_experiment(dataset, answer_question)
            >>> def is_polite(output):
            ...     return "please" in output.lower() or "thank you" in output.lower()
            >>> result = registry.evaluate_experiment(result, evaluators=[is_polite], options=ExperimentOptions(verbose=False))
        """
        options = options or ExperimentOptions()
        # unlike run_experiment, evaluate_experiment's dry_run has no "sample size" form
        execution_kwargs = {**options.execution.to_kwargs(), "dry_run": bool(options.execution.dry_run)}
        result = self._client.experiments.evaluate_experiment(
            experiment=experiment,
            evaluators=evaluators,
            **execution_kwargs,
            print_summary=options.verbose,
            **kwargs,
        )
        logger.info(f"Evaluated experiment '{result['experiment_id']}' — {len(result['evaluation_runs'])} evaluation runs")
        return result
