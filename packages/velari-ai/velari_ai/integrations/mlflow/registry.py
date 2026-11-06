from __future__ import annotations
import  logging
import  mlflow
import  pandas as pd
from    dataclasses import dataclass, field
from    datetime import datetime, timezone
from    typing import Any, Dict, List, Optional, Union
from    mlflow import MlflowClient, genai
from    mlflow.genai.datasets import EvaluationDataset
from    mlflow.pyfunc import PyFuncModel, load_model
from    mlflow.entities.model_registry import Prompt, PromptVersion
from    mlflow.entities.model_registry.model_version import ModelVersion
from    mlflow.entities.model_registry.registered_model import RegisteredModel
# package modules
from velari_data.registry import DatasetRegistryInterface, Registry
from .connection import Connector

logger = logging.getLogger(__name__)


class PromptRegistry(Registry):
    """MLflow Prompt Registry wrapper: create, get, alias, tag, and list versioned prompts.

    Args:
        connector (Connector): Connected MLflow tracking server; its URI is (re)applied so `mlflow.genai` calls target it.
    """
    def __init__(self, connector: Connector) -> None:
        mlflow.set_tracking_uri(connector.url)

    def create(
        self,
        name: str,
        template: Union[str, List[Dict[str, Any]]],
        commit_message: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> PromptVersion:
        """Register a prompt, creating it if new or adding a version if it already exists.

        Args:
            name (str): Prompt name.
            template (Union[str, List[Dict[str, Any]]]): Text with `{{variables}}`, or chat messages.
            commit_message (Optional[str]): Description of this version's change.
            tags (Optional[Dict[str, str]]): Version-level tags.
            **kwargs (Any): Passed to MLflow as-is (e.g. `response_format`, `model_config`).

        Returns:
            PromptVersion: The newly created version.

        Examples:
            >>> registry = PromptRegistry(Connector.from_config(cfg))
            >>> v = registry.create(
            ...     "ticket-triage",
            ...     "Classify the severity of this support ticket: {{ticket_text}}",
            ...     commit_message="Add severity rubric",
            ... )
            >>> v.version
            1
        """
        options = {"commit_message": commit_message, "tags": tags}
        prompt = genai.register_prompt(name, template, **{k: v for k, v in options.items() if v is not None}, **kwargs)
        logger.info(f"Registered prompt {name!r} version {prompt.version}")
        return prompt

    def get(self, name: str, version: Optional[Union[str, int]] = None, alias: Optional[str] = None) -> PromptVersion:
        """Get a prompt by version, by alias, or (with neither) its latest version.

        Args:
            name (str): Prompt name.
            version (Optional[Union[str, int]]): Specific version; mutually exclusive with `alias`.
            alias (Optional[str]): Alias to resolve, e.g. `"production"`.

        Returns:
            PromptVersion: The resolved prompt version.

        Raises:
            ValueError: If both `version` and `alias` are given.
            MlflowException: If the prompt, version, or alias doesn't exist.

        Examples:
            >>> registry = PromptRegistry(Connector.from_config(cfg))
            >>> prompt = registry.get("churn-report-summary", alias="production")
            >>> prompt.format(quarter="Q3", segment="enterprise")
        """
        if version is not None and alias is not None:
            raise ValueError("Pass either 'version' or 'alias', not both")
        return genai.load_prompt(f"prompts:/{name}@{alias}" if alias else name, version=version)

    def set_alias(self, name: str, alias: str, version: int) -> None:
        """Point an alias at a prompt version, moving it if it already exists.

        Args:
            name (str): Prompt name.
            alias (str): Alias to set, e.g. `"staging"`.
            version (int): Version the alias should resolve to.

        Examples:
            >>> registry = PromptRegistry(Connector.from_config(cfg))
            >>> registry.set_alias("contract-clause-extractor", "production", version=3)
        """
        genai.set_prompt_alias(name, alias, version)
        logger.info(f"Alias {alias!r} on prompt {name!r} -> version {version}")

    def delete_alias(self, name: str, alias: str) -> None:
        """Remove an alias from a prompt; the underlying version is kept.

        Args:
            name (str): Prompt name.
            alias (str): Alias to remove.

        Examples:
            >>> registry = PromptRegistry(Connector.from_config(cfg))
            >>> registry.delete_alias("contract-clause-extractor", "staging")
        """
        genai.delete_prompt_alias(name, alias)
        logger.info(f"Deleted alias {alias!r} from prompt {name!r}")

    def set_tag(self, name: str, key: str, value: str) -> None:
        """Set a prompt-level tag.

        Args:
            name (str): Prompt name.
            key (str): Tag key.
            value (str): Tag value.

        Examples:
            >>> registry = PromptRegistry(Connector.from_config(cfg))
            >>> registry.set_tag("kb-answer-grounding", "owner", "retrieval-team")
        """
        genai.set_prompt_tag(name, key, value)

    def delete_tag(self, name: str, key: str) -> None:
        """Remove a prompt-level tag.

        Args:
            name (str): Prompt name.
            key (str): Tag key.

        Examples:
            >>> registry = PromptRegistry(Connector.from_config(cfg))
            >>> registry.delete_tag("kb-answer-grounding", "owner")
        """
        genai.delete_prompt_tag(name, key)

    def list(self, filter_string: Optional[str] = None, max_results: Optional[int] = None, **kwargs: Any) -> List[Prompt]:
        """List registered prompts, optionally filtered.

        Args:
            filter_string (Optional[str]): MLflow filter, e.g. `"name LIKE 'ticket%'"`.
            max_results (Optional[int]): Cap on results; MLflow's default applies when omitted.
            **kwargs (Any): Passed to `genai.search_prompts` as-is (e.g. `order_by`).

        Returns:
            List[Prompt]: Matching prompts.

        Examples:
            >>> registry = PromptRegistry(Connector.from_config(cfg))
            >>> [p.name for p in registry.list("name LIKE 'ticket%'")]
            ['ticket-triage', 'ticket-summary']
        """
        options = {"filter_string": filter_string, "max_results": max_results}
        return list(genai.search_prompts(**{k: v for k, v in options.items() if v is not None}, **kwargs))

    def to_dataframe(self) -> pd.DataFrame:
        """List the registered prompts as a DataFrame, one row per prompt.

        Returns:
            pd.DataFrame: Columns `name`, `description`, `tags`, `creation_timestamp`.

        Examples:
            >>> registry = PromptRegistry(Connector.from_config(cfg))
            >>> registry.to_dataframe()[["name", "description"]]
        """
        return pd.DataFrame([
            {"name": p.name, "description": p.description, "tags": p.tags, "creation_timestamp": p.creation_timestamp}
            for p in self.list()
        ])


class ArtifactRegistry(Registry):
    """Client-side, global-state MLflow API for registered apps: get into this process, plus create and list.

    Not the same class as `client.ArtifactRegistry`, which manages aliases and tags through `MlflowClient`.

    Args:
        connector (Connector): Connected MLflow tracking server; its URI is (re)applied since these calls read global state.
    """
    def __init__(self, connector: Connector) -> None:
        mlflow.set_tracking_uri(connector.url)

    def get(self, name: str, version: Optional[Union[str, int]] = None, alias: Optional[str] = None) -> PyFuncModel:
        """Get a registered app by version, by alias, or (with neither) its latest version.

        Args:
            name (str): Registered name.
            version (Optional[Union[str, int]]): Specific version; mutually exclusive with `alias`.
            alias (Optional[str]): Alias to resolve, e.g. `"champion"`.

        Returns:
            PyFuncModel: The loaded app; call `.predict()` on it.

        Raises:
            ValueError: If both `version` and `alias` are given.
            MlflowException: If the name, version, or alias doesn't exist.

        Examples:
            >>> registry = ArtifactRegistry(Connector.from_config(cfg))
            >>> agent = registry.get("research-assistant", alias="champion")
            >>> agent.predict([{"question": "Summarize Q3 churn drivers for ORG-4471."}])
        """
        if version is not None and alias is not None:
            raise ValueError("Pass either 'version' or 'alias', not both")
        ref = f"@{alias}" if alias else f"/{'latest' if version is None else version}"
        return load_model(f"models:/{name}{ref}")

    def create(
        self,
        model_uri: str,
        name: str,
        tags: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> ModelVersion:
        """Register a logged app under a name, creating it if new or adding a version if it already exists.

        Args:
            model_uri (str): Logged model to register, e.g. `info.model_uri` from `pyfunc.log_model()`.
            name (str): Registered name.
            tags (Optional[Dict[str, str]]): Version-level tags.
            **kwargs (Any): Passed to `mlflow.register_model` (e.g. `await_registration_for`).

        Returns:
            ModelVersion: The newly created version.

        Examples:
            >>> registry = ArtifactRegistry(Connector.from_config(cfg))
            >>> info = mlflow.pyfunc.log_model(name="agent", python_model=contract_review_agent)
            >>> registry.create(info.model_uri, "contract-review-agent").version
            1
        """
        model_version = mlflow.register_model(model_uri, name, tags=tags, **kwargs)
        logger.info(f"Registered app {name!r} version {model_version.version}")
        return model_version

    def list(self, filter_string: Optional[str] = None, max_results: Optional[int] = None, **kwargs: Any) -> List[RegisteredModel]:
        """List registered apps, optionally filtered.

        Args:
            filter_string (Optional[str]): MLflow filter, e.g. `"name LIKE 'contract%'"`.
            max_results (Optional[int]): Cap on results; MLflow's default applies when omitted.
            **kwargs (Any): Passed to `mlflow.search_registered_models` as-is (e.g. `order_by`).

        Returns:
            List[RegisteredModel]: Matching registered apps.

        Examples:
            >>> registry = ArtifactRegistry(Connector.from_config(cfg))
            >>> [m.name for m in registry.list("name LIKE 'contract%'")]
            ['contract-review-agent', 'contract-clause-extractor']
        """
        return mlflow.search_registered_models(filter_string=filter_string, max_results=max_results, **kwargs)

    def to_dataframe(self) -> pd.DataFrame:
        """List the registered apps as a DataFrame, one row per app.

        Returns:
            pd.DataFrame: Columns `name`, `description`, `tags`, `creation_timestamp`, `last_updated_timestamp`.

        Examples:
            >>> registry = ArtifactRegistry(Connector.from_config(cfg))
            >>> registry.to_dataframe()[["name", "last_updated_timestamp"]]
        """
        return pd.DataFrame([
            {
                "name": m.name, "description": m.description, "tags": m.tags,
                "creation_timestamp": m.creation_timestamp, "last_updated_timestamp": m.last_updated_timestamp,
            }
            for m in self.list()
        ])


@dataclass
class DatasetOptions(object):
    """Names `DatasetRegistry` writes to MLflow; defaults give `velari.description`, `velari.dataset`, `velari.snapshot`.

    Args:
        tag_namespace (str): Prefix for the description, dataset, and snapshot tags.
        snapshot_prefix (str): Prefix of the dataset tags `snapshot()` writes (`<prefix><label>.digest`, ...).
        snapshot_file (str): Artifact file the snapshot table is logged to; must end in `.json`.
    """
    tag_namespace:   str  = field(default="velari")
    snapshot_prefix: str  = field(default="snapshot.")
    snapshot_file:   str  = field(default="snapshot.json")

    @property
    def description_tag(self) -> str:
        return f"{self.tag_namespace}.description"

    @property
    def dataset_tag(self) -> str:
        return f"{self.tag_namespace}.dataset"

    @property
    def snapshot_tag(self) -> str:
        return f"{self.tag_namespace}.snapshot"


class DatasetRegistry(DatasetRegistryInterface):
    """MLflow evaluation datasets: create, extend, tag, delete, and snapshot versioned test sets.

    Args:
        connector (Connector): Connected MLflow tracking server; its URI is (re)applied since `mlflow.genai.datasets` reads global state.
        options (Optional[DatasetOptions]): Tag and snapshot names; defaults to `DatasetOptions()`.
    """
    def __init__(self, connector: Connector, options: Optional[DatasetOptions] = None) -> None:
        mlflow.set_tracking_uri(connector.url)
        self._options = options or DatasetOptions()

    @staticmethod
    def from_dataframe(
        df_records: pd.DataFrame,
        input_keys: List[str],
        output_keys: Optional[List[str]] = None,
        metadata_keys: Optional[List[str]] = None,
    ) -> List[Dict[str, Dict[str, Any]]]:
        """Convert DataFrame rows into MLflow dataset records (`inputs`, `expectations`, `tags`).

        Args:
            df_records (pd.DataFrame): One row per record.
            input_keys (List[str]): Columns holding record inputs.
            output_keys (Optional[List[str]]): Columns holding expected outputs; omitted from the record when unset.
            metadata_keys (Optional[List[str]]): Columns stored as each record's `tags`; omitted from the record when unset.

        Returns:
            List[Dict[str, Dict[str, Any]]]: Records ready for `merge_records`.

        Examples:
            >>> df_tickets = pd.DataFrame({"question": ["How do I reset my password?"], "answer": ["Visit /account/reset."]})
            >>> DatasetRegistry.from_dataframe(df_tickets, input_keys=["question"], output_keys=["answer"])
            [{'inputs': {'question': 'How do I reset my password?'}, 'expectations': {'answer': 'Visit /account/reset.'}}]
        """
        fields = {"inputs": input_keys, "expectations": output_keys, "tags": metadata_keys}
        return [
            {field: {key: row[key] for key in keys} for field, keys in fields.items() if keys}
            for row in df_records.to_dict("records")
        ]

    def create(
        self,
        name: str,
        df_records: Optional[pd.DataFrame] = None,
        input_keys: Optional[List[str]] = None,
        output_keys: Optional[List[str]] = None,
        description: Optional[str] = None,
        metadata_keys: Optional[List[str]] = None,
        dataset_tags: Optional[Dict[str, str]] = None,
        experiment_id: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationDataset:
        """Create a dataset, optionally seeded from a DataFrame.

        Args:
            name (str): Dataset name.
            df_records (Optional[pd.DataFrame]): Initial rows; requires `input_keys`.
            input_keys (Optional[List[str]]): Columns holding record inputs.
            output_keys (Optional[List[str]]): Columns holding expected outputs.
            description (Optional[str]): Stored as the description tag (`velari.description` by default; see `DatasetOptions`).
            metadata_keys (Optional[List[str]]): Columns stored as each record's `tags`.
            dataset_tags (Optional[Dict[str, str]]): Tags on the dataset as a whole.
            experiment_id (Optional[str]): Experiment to attach to; defaults to the connector's active experiment.
            **kwargs (Any): Passed to `mlflow.genai.datasets.create_dataset` as-is.

        Returns:
            EvaluationDataset: The created dataset.

        Raises:
            ValueError: If `df_records` is given without `input_keys`.

        Examples:
            >>> registry = DatasetRegistry(Connector.from_config(cfg))
            >>> df_tickets = pd.DataFrame({"question": ["How do I reset my password?"], "answer": ["Visit /account/reset."]})
            >>> dataset = registry.create(
            ...     "support-ticket-qa", df_tickets, input_keys=["question"], output_keys=["answer"],
            ...     description="Support ticket Q&A pairs",
            ... )
            >>> dataset.name
            'support-ticket-qa'
        """
        if df_records is not None and not input_keys:
            raise ValueError("'input_keys' is required when 'df_records' is given")
        all_tags = {**(dataset_tags or {}), **({self._options.description_tag: description} if description else {})}
        options = {"experiment_id": experiment_id, "tags": all_tags or None}
        dataset = genai.datasets.create_dataset(name=name, **{k: v for k, v in options.items() if v is not None}, **kwargs)
        if df_records is not None and input_keys:
            dataset.merge_records(self.from_dataframe(df_records, input_keys, output_keys, metadata_keys))
        logger.info(f"Created dataset {name!r}")
        return dataset

    def get(self, name: str) -> EvaluationDataset:
        """Retrieve a dataset by name; call `.to_df()` on it for its records.

        Args:
            name (str): Dataset name.

        Returns:
            EvaluationDataset: The matching dataset.

        Raises:
            MlflowException: If the dataset doesn't exist.

        Examples:
            >>> registry = DatasetRegistry(Connector.from_config(cfg))
            >>> df_records = registry.get("churn-analysis-eval").to_df()
        """
        return genai.datasets.get_dataset(name=name)

    def extend(
        self,
        dataset: Union[str, EvaluationDataset],
        df_records: pd.DataFrame,
        input_keys: List[str],
        output_keys: Optional[List[str]] = None,
        metadata_keys: Optional[List[str]] = None,
        dataset_tags: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> EvaluationDataset:
        """Upsert DataFrame rows into an existing dataset, optionally updating its tags.

        Rows MLflow treats as the same record are replaced, not duplicated. No new version is created;
        call `snapshot()` for a restore point.

        Args:
            dataset (Union[str, EvaluationDataset]): Dataset name, or the dataset object from `create()`/`get()`.
            df_records (pd.DataFrame): Rows to upsert.
            input_keys (List[str]): Columns holding record inputs.
            output_keys (Optional[List[str]]): Columns holding expected outputs.
            metadata_keys (Optional[List[str]]): Columns stored as each record's `tags`.
            dataset_tags (Optional[Dict[str, str]]): Tags on the dataset as a whole, to set or overwrite.
            **kwargs (Any): Accepted for `DatasetRegistryInterface` compatibility; none are supported here.

        Returns:
            EvaluationDataset: The updated dataset.

        Raises:
            TypeError: If any backend-specific option is passed (e.g. Phoenix's `split_key`).

        Examples:
            >>> registry = DatasetRegistry(Connector.from_config(cfg))
            >>> df_new = pd.DataFrame({"query": ["Summarize Q3 churn drivers for ORG-4471."], "label": ["churn"]})
            >>> dataset = registry.extend("churn-analysis-eval", df_new, input_keys=["query"], output_keys=["label"], dataset_tags={"stage": "v2"})
        """
        if kwargs:
            raise TypeError(f"extend() got unsupported options for MLflow: {sorted(kwargs)}")
        target = self.get(dataset if isinstance(dataset, str) else dataset.name)
        target.merge_records(self.from_dataframe(df_records, input_keys, output_keys, metadata_keys))
        if dataset_tags:
            genai.datasets.set_dataset_tags(target.dataset_id, dataset_tags)
        logger.info(f"Extended dataset {target.name!r} with {len(df_records)} records")
        return target

    def update(self, name: str, dataset_tags: Dict[str, Optional[str]]) -> EvaluationDataset:
        """Change dataset tags without touching records; a `None` value removes that tag.

        Args:
            name (str): Dataset name.
            dataset_tags (Dict[str, Optional[str]]): Tags to set or overwrite; `None` removes the key.

        Returns:
            EvaluationDataset: The dataset (its `.tags` reflect the state before this call).

        Examples:
            >>> registry = DatasetRegistry(Connector.from_config(cfg))
            >>> registry.update("kb-retrieval-eval", dataset_tags={"owner": "retrieval-team", "draft": None})
        """
        dataset = self.get(name)
        to_set = {key: value for key, value in dataset_tags.items() if value is not None}
        if to_set:
            genai.datasets.set_dataset_tags(dataset.dataset_id, to_set)
        for key in (key for key, value in dataset_tags.items() if value is None):
            genai.datasets.delete_dataset_tag(dataset.dataset_id, key)
        logger.info(f"Updated tags on dataset {name!r}")
        return dataset

    def delete(
        self,
        name: str,
        record_ids: Optional[List[str]] = None,
        dataset_tags: Optional[List[str]] = None,
    ) -> None:
        """Delete a dataset, or only specific records and/or dataset tags when given.

        Args:
            name (str): Dataset name.
            record_ids (Optional[List[str]]): Delete just these records; the dataset is kept.
            dataset_tags (Optional[List[str]]): Delete just these dataset tag keys; the dataset is kept.

        Examples:
            >>> registry = DatasetRegistry(Connector.from_config(cfg))
            >>> registry.delete("contract-clause-eval", dataset_tags=["draft"])
            >>> registry.delete("contract-clause-eval")
        """
        dataset = self.get(name)
        if record_ids is None and dataset_tags is None:
            genai.datasets.delete_dataset(dataset_id=dataset.dataset_id)
            logger.info(f"Deleted dataset {name!r}")
            return
        if record_ids:
            dataset.delete_records(record_ids)
        for key in dataset_tags or []:
            genai.datasets.delete_dataset_tag(dataset.dataset_id, key)
        logger.info(f"Deleted {len(record_ids or [])} records and {len(dataset_tags or [])} dataset tags from dataset {name!r}")

    def list(self, **kwargs: Any) -> List[EvaluationDataset]:
        """List datasets, optionally filtered.

        Args:
            **kwargs (Any): Passed to `search_datasets` (e.g. `filter_string`, `max_results`, `order_by`).

        Returns:
            List[EvaluationDataset]: Matching datasets.

        Examples:
            >>> registry = DatasetRegistry(Connector.from_config(cfg))
            >>> [d.name for d in registry.list(filter_string="name LIKE 'support%'")]
            ['support-ticket-qa', 'support-ticket-queries']
        """
        return list(genai.datasets.search_datasets(**kwargs))

    def to_dataframe(self) -> pd.DataFrame:
        """List the datasets as a DataFrame, one row per dataset (same meaning as Arize's `to_dataframe()`).

        Returns:
            pd.DataFrame: Columns `name`, `dataset_id`, `digest`, `tags`.

        Examples:
            >>> registry = DatasetRegistry(Connector.from_config(cfg))
            >>> registry.to_dataframe()[["name", "digest"]]
        """
        return pd.DataFrame([
            {"name": d.name, "dataset_id": d.dataset_id, "digest": d.digest, "tags": d.tags}
            for d in self.list()
        ])

    def snapshot(self, name: str, label: str) -> Dict[str, str]:
        """Freeze the dataset's current records under a label; reusing a label moves it to the latest snapshot.

        Records are logged as a table artifact on a run tagged with the dataset and snapshot names;
        digest, record count, and timestamp are written to the dataset as `<snapshot_prefix><label>.*` tags.

        Args:
            name (str): Dataset name.
            label (str): Snapshot label, e.g. `"v1"` or `"2026-q3-release"`.

        Returns:
            Dict[str, str]: The `<snapshot_prefix><label>.*` tags written to the dataset.

        Examples:
            >>> registry = DatasetRegistry(Connector.from_config(cfg))
            >>> registry.snapshot("contract-clause-eval", "pre-annotation-review")
        """
        dataset = self.get(name)
        df_records = dataset.to_df()
        prefix = f"{self._options.snapshot_prefix}{label}"
        meta = {
            f"{prefix}.digest":     str(dataset.digest),
            f"{prefix}.count":      str(len(df_records)),
            f"{prefix}.created_at": datetime.now(timezone.utc).isoformat(),
        }
        with mlflow.start_run(experiment_id=dataset.experiment_ids[0], run_name=f"snapshot-{name}-{label}", nested=True):
            mlflow.set_tags({self._options.dataset_tag: name, self._options.snapshot_tag: label})
            mlflow.log_table(df_records, self._options.snapshot_file)
        genai.datasets.set_dataset_tags(dataset.dataset_id, meta)
        logger.info(f"Snapshot {label!r} of dataset {name!r}: {len(df_records)} records")
        return meta

    def list_snapshots(self, name: str) -> List[str]:
        """List a dataset's snapshot labels.

        Args:
            name (str): Dataset name.

        Returns:
            List[str]: Labels, sorted alphabetically.

        Examples:
            >>> registry = DatasetRegistry(Connector.from_config(cfg))
            >>> registry.list_snapshots("contract-clause-eval")
            ['pre-annotation-review', 'v1']
        """
        return sorted(
            key.removeprefix(self._options.snapshot_prefix).removesuffix(".created_at")
            for key in (self.get(name).tags or {})
            if key.startswith(self._options.snapshot_prefix) and key.endswith(".created_at")
        )

    def load_snapshot(self, name: str, label: str) -> pd.DataFrame:
        """Load the records frozen by `snapshot()`.

        Args:
            name (str): Dataset name.
            label (str): Snapshot label.

        Returns:
            pd.DataFrame: Records as they were when the snapshot was taken.

        Raises:
            ValueError: If the dataset has no snapshot with this label.

        Examples:
            >>> registry = DatasetRegistry(Connector.from_config(cfg))
            >>> df_v1 = registry.load_snapshot("support-ticket-qa", "v1")
        """
        dataset = self.get(name)
        df_runs = mlflow.search_runs(
            experiment_ids=dataset.experiment_ids,
            filter_string=f"tags.`{self._options.dataset_tag}` = '{name}' AND tags.`{self._options.snapshot_tag}` = '{label}'",
            max_results=1,
            order_by=["attributes.start_time DESC"],
        )
        run_id = next(iter(df_runs["run_id"]), None)
        if run_id is None:
            raise ValueError(f"Dataset {name!r} has no snapshot {label!r}")
        return MlflowClient().load_table(dataset.experiment_ids[0], self._options.snapshot_file, run_ids=[run_id])
