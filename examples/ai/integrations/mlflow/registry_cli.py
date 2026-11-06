# uv run python examples/ai/integrations/mlflow/registry_cli.py register ticket-triage --template "Classify: {{ticket_text}}" --tag team=support
# uv run python examples/ai/integrations/mlflow/registry_cli.py set-alias ticket-triage production 1
# uv run python examples/ai/integrations/mlflow/registry_cli.py load ticket-triage --alias production
# uv run python examples/ai/integrations/mlflow/registry_cli.py list --filter "name LIKE 'ticket%'"
# uv run python examples/ai/integrations/mlflow/registry_cli.py list --endpoint http://localhost:5001
# uv run python examples/ai/integrations/mlflow/registry_cli.py artifact register models:/m-abc123 billing-agent --tag team=support
# uv run python examples/ai/integrations/mlflow/registry_cli.py artifact load billing-agent --alias champion --input '[{"question": "..."}]'
# uv run python examples/ai/integrations/mlflow/registry_cli.py artifact list --filter "name LIKE 'billing%'"
# uv run python examples/ai/integrations/mlflow/registry_cli.py dataset create support-qa --file tickets.csv --input-key question --output-key answer --tag team=support
# uv run python examples/ai/integrations/mlflow/registry_cli.py dataset extend support-qa --file more.csv --input-key question --output-key answer
# uv run python examples/ai/integrations/mlflow/registry_cli.py dataset snapshot support-qa v1
# uv run python examples/ai/integrations/mlflow/registry_cli.py dataset get support-qa --head 3
# See examples/ai/integrations/mlflow/README.md for details.

import  json
import  logging
import  pandas as pd
from    abc import ABC, abstractmethod
from    contextlib import contextmanager
from    pathlib import Path
from    typing import Any, Dict, Generator, List, Optional

import  typer
from    mlflow.exceptions import MlflowException
from    omegaconf import DictConfig
from    rich.console import Console
from    rich.panel import Panel
from    rich.table import Table
# package modules
from    velari_core.core import read_root_dir
from    velari_core.core.io.filesystem import Filesystem
from    velari_core.core.io.partition.hydra import read_hydra_compose
from    velari_data.registry import Registry
from    velari_ai.integrations.mlflow.connection import Connector, ConnectorConfig
from    velari_ai.integrations.mlflow.registry import ArtifactRegistry, DatasetRegistry, PromptRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------------------------------------------------
# supporting logic (what the commands need)
# ---------------------------------------------------------------------------------------------------------------------
class RegistryService(ABC):
    """Supporting logic shared by every command group: connecting, tag parsing, error reporting, and output.

    Args:
        cfg (DictConfig): Composed app config; builds the local-store Connector when `--endpoint` is not given.
    """
    console        = Console(stderr=True, color_system="auto", force_terminal=True)
    EndpointOption = typer.Option(None, "--endpoint", help="Connect to an already-running MLflow tracking server instead of the local store.")

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg

    def connector(self, endpoint: Optional[str]) -> Connector:
        """Build a Connector to a remote tracking server (`endpoint`) or the config's local store.

        Args:
            endpoint (Optional[str]): Tracking server URL; None uses the config's local store.

        Returns:
            Connector: The connected tracker; its URI is printed to the console.
        """
        connector = Connector(ConnectorConfig(remote=ConnectorConfig.Remote(endpoint=endpoint))) if endpoint else Connector.from_config(self._cfg)
        self.console.print(f"[dim]MLflow tracking → {connector.url}[/dim]")
        return connector

    @staticmethod
    def parse_tags(tags: List[str]) -> Dict[str, str]:
        """Parse repeated `KEY=VALUE` options into a dict.

        Args:
            tags (List[str]): Raw `--tag` values.

        Returns:
            Dict[str, str]: Tag key to value.

        Raises:
            typer.BadParameter: If any value is not `KEY=VALUE`.
        """
        try:
            return dict(tag.split("=", 1) for tag in tags)
        except ValueError:
            raise typer.BadParameter("--tag must be KEY=VALUE")

    @contextmanager
    def report_mlflow_errors(self) -> Generator[None, None, None]:
        """Print an MlflowException in red and exit with code 1 instead of a traceback."""
        try:
            yield
        except MlflowException as exc:
            self.console.print(f"[red]{exc.message}[/red]")
            raise typer.Exit(code=1)

    @abstractmethod
    def registry(self, endpoint: Optional[str]) -> Registry:
        """Build this command group's registry against `endpoint` (or the config's local store)."""


class PromptService(RegistryService):
    """Support for the top-level prompt commands."""
    def registry(self, endpoint: Optional[str]) -> PromptRegistry:
        return PromptRegistry(self.connector(endpoint))


class ArtifactService(RegistryService):
    """Support for the `artifact` group."""
    def registry(self, endpoint: Optional[str]) -> ArtifactRegistry:
        return ArtifactRegistry(self.connector(endpoint))


class DatasetService(RegistryService):
    """Support for the `dataset` group: building the registry, reading input files, and rendering records and tags."""
    OutputKeyOption   = typer.Option(None, "--output-key", help="Column holding expected outputs; repeatable.")
    MetadataKeyOption = typer.Option(None, "--metadata-key", help="Column stored as per-record metadata; repeatable.")
    DatasetTagOption  = typer.Option(None, "--tag", help="Dataset tag as KEY=VALUE; repeatable.")

    def registry(self, endpoint: Optional[str]) -> DatasetRegistry:
        return DatasetRegistry(self.connector(endpoint))

    @staticmethod
    def read_records(file: Path) -> pd.DataFrame:
        """Read a CSV/JSON/Excel file into a DataFrame, reporting an unreadable or unsupported file as a bad parameter."""
        try:
            return Filesystem.parse(file)
        except (ValueError, OSError) as exc:
            raise typer.BadParameter(f"--file {file}: {exc}")

    def print_records(self, df_records: pd.DataFrame, head: int) -> None:
        """Print the first `head` records' inputs, expectations, and tags as a table."""
        columns = [column for column in ("inputs", "expectations", "tags") if column in df_records.columns]
        table = Table(*columns, title=f"{len(df_records)} records (showing {min(head, len(df_records))})")
        for row in df_records.head(head)[columns].itertuples(index=False):
            table.add_row(*(str(value) for value in row))
        self.console.print(table)

    @staticmethod
    def visible_tags(tags: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Drop MLflow's own `mlflow.*` tags so only user tags are shown."""
        return {key: value for key, value in (tags or {}).items() if not key.startswith("mlflow.")}


# ---------------------------------------------------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------------------------------------------------
def _load_cfg(primary_config_dir: str, config_name: str, searchpath_dir: str) -> DictConfig:
    cfg, _ = read_hydra_compose(
        primary_config_dir,
        config_name,
        overrides=[f"hydra.searchpath=[file://{searchpath_dir}]"],
    )
    return cfg


# ---------------------------------------------------------------------------------------------------------------------
# prompts (top-level commands)
# ---------------------------------------------------------------------------------------------------------------------
def build_prompt_app(cfg: DictConfig) -> typer.Typer:
    """Build the prompt commands (register, load, list, aliases, tags), mounted at the top level."""
    app = typer.Typer(rich_markup_mode="rich")
    service = PromptService(cfg)

    @app.command(help="Register a prompt, creating it or adding a new version.")
    def register(
        name:           str                = typer.Argument(..., help="Prompt name."),
        template:       Optional[str]      = typer.Option(None, help="Prompt text, with {{variables}}."),
        template_file:  Optional[Path]     = typer.Option(None, help="Read the prompt text from this file instead of --template."),
        commit_message: Optional[str]      = typer.Option(None, help="Description of this version's change."),
        tag:            Optional[List[str]] = typer.Option(None, help="Version tag as KEY=VALUE; repeatable."),
        endpoint:       Optional[str]      = service.EndpointOption,
    ) -> None:
        if (template is None) == (template_file is None):
            raise typer.BadParameter("Pass exactly one of --template or --template-file")
        text = template if template is not None else Path(str(template_file)).read_text()
        with service.report_mlflow_errors():
            version = service.registry(endpoint).create(name, text, commit_message=commit_message, tags=service.parse_tags(tag or []) or None)
        service.console.print(f"[green]Registered[/green] {name} version {version.version}")

    @app.command(help="Load a prompt by version, by alias, or its latest version.")
    def load(
        name:     str           = typer.Argument(..., help="Prompt name."),
        version:  Optional[int] = typer.Option(None, help="Version number; not combinable with --alias."),
        alias:    Optional[str] = typer.Option(None, help="Alias to resolve, e.g. production."),
        endpoint: Optional[str] = service.EndpointOption,
    ) -> None:
        with service.report_mlflow_errors():
            try:
                prompt = service.registry(endpoint).get(name, version=version, alias=alias)
            except ValueError as exc:
                raise typer.BadParameter(str(exc))
        service.console.print(Panel(
            str(prompt.template),
            title=f"{prompt.name} v{prompt.version}",
            subtitle=f"aliases={prompt.aliases} · {prompt.commit_message or 'no commit message'}",
        ))

    @app.command(name="list", help="List registered prompts.")
    def list_prompts(
        filter_string: Optional[str] = typer.Option(None, "--filter", help="MLflow filter, e.g. \"name LIKE 'ticket%'\"."),
        max_results:   Optional[int] = typer.Option(None, help="Cap on the number of prompts returned."),
        endpoint:      Optional[str] = service.EndpointOption,
    ) -> None:
        with service.report_mlflow_errors():
            prompts = service.registry(endpoint).list(filter_string, max_results)
        table = Table("name", "tags", "description")
        for prompt in prompts:
            table.add_row(prompt.name, str({k: v for k, v in prompt.tags.items() if not k.startswith("_mlflow")}), prompt.description or "")
        service.console.print(table)

    @app.command(name="set-alias", help="Point an alias at a prompt version, moving it if it already exists.")
    def set_alias(
        name:     str           = typer.Argument(..., help="Prompt name."),
        alias:    str           = typer.Argument(..., help="Alias to set, e.g. staging."),
        version:  int           = typer.Argument(..., help="Version the alias should resolve to."),
        endpoint: Optional[str] = service.EndpointOption,
    ) -> None:
        with service.report_mlflow_errors():
            service.registry(endpoint).set_alias(name, alias, version)
        service.console.print(f"[green]Alias[/green] {alias} → {name} v{version}")

    @app.command(name="delete-alias", help="Remove an alias; the prompt version is kept.")
    def delete_alias(
        name:     str           = typer.Argument(..., help="Prompt name."),
        alias:    str           = typer.Argument(..., help="Alias to remove."),
        endpoint: Optional[str] = service.EndpointOption,
    ) -> None:
        with service.report_mlflow_errors():
            service.registry(endpoint).delete_alias(name, alias)
        service.console.print(f"[green]Deleted alias[/green] {alias} from {name}")

    @app.command(name="set-tag", help="Set a prompt-level tag.")
    def set_tag(
        name:     str           = typer.Argument(..., help="Prompt name."),
        key:      str           = typer.Argument(..., help="Tag key."),
        value:    str           = typer.Argument(..., help="Tag value."),
        endpoint: Optional[str] = service.EndpointOption,
    ) -> None:
        with service.report_mlflow_errors():
            service.registry(endpoint).set_tag(name, key, value)
        service.console.print(f"[green]Tagged[/green] {name} {key}={value}")

    @app.command(name="delete-tag", help="Remove a prompt-level tag.")
    def delete_tag(
        name:     str           = typer.Argument(..., help="Prompt name."),
        key:      str           = typer.Argument(..., help="Tag key."),
        endpoint: Optional[str] = service.EndpointOption,
    ) -> None:
        with service.report_mlflow_errors():
            service.registry(endpoint).delete_tag(name, key)
        service.console.print(f"[green]Removed tag[/green] {key} from {name}")

    return app


# ---------------------------------------------------------------------------------------------------------------------
# artifacts (`artifact` group)
# ---------------------------------------------------------------------------------------------------------------------
def build_artifact_app(cfg: DictConfig) -> typer.Typer:
    """Build the `artifact` group: register, load, and list registered GenAI apps."""
    app = typer.Typer(help="Registered GenAI apps/agents (logged models): register, load, and list.", rich_markup_mode="rich")
    service = ArtifactService(cfg)

    @app.command(name="register", help="Register a logged model, creating it or adding a new version.")
    def register_artifact(
        model_uri: str                 = typer.Argument(..., help="Logged model URI, e.g. models:/m-abc123 or runs:/<run_id>/agent."),
        name:      str                 = typer.Argument(..., help="Registered name."),
        tag:       Optional[List[str]] = typer.Option(None, help="Version tag as KEY=VALUE; repeatable."),
        endpoint:  Optional[str]       = service.EndpointOption,
    ) -> None:
        with service.report_mlflow_errors():
            version = service.registry(endpoint).create(model_uri, name, tags=service.parse_tags(tag or []) or None)
        service.console.print(f"[green]Registered[/green] {name} version {version.version}")

    @app.command(name="load", help="Load a registered app by version, by alias, or its latest version; optionally run it.")
    def load_artifact(
        name:       str           = typer.Argument(..., help="Registered name."),
        version:    Optional[int] = typer.Option(None, help="Version number; not combinable with --alias."),
        alias:      Optional[str] = typer.Option(None, help="Alias to resolve, e.g. champion."),
        input_json: Optional[str] = typer.Option(None, "--input", help="JSON passed to the model's predict(), e.g. '[1, 2]'."),
        endpoint:   Optional[str] = service.EndpointOption,
    ) -> None:
        data: Any = None
        try:
            data = json.loads(input_json) if input_json is not None else None
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"--input must be valid JSON: {exc}")
        with service.report_mlflow_errors():
            try:
                model = service.registry(endpoint).get(name, version=version, alias=alias)
            except ValueError as exc:
                raise typer.BadParameter(str(exc))
            service.console.print(Panel(
                f"model_id={model.model_id}\nflavors={list(model.metadata.flavors)}\nsignature={model.metadata.signature}",
                title=f"{name} loaded",
            ))
            if input_json is not None:
                service.console.print(model.predict(data))

    @app.command(name="list", help="List registered apps.")
    def list_artifacts(
        filter_string: Optional[str] = typer.Option(None, "--filter", help="MLflow filter, e.g. \"name LIKE 'billing%'\"."),
        max_results:   Optional[int] = typer.Option(None, help="Cap on the number of apps returned."),
        endpoint:      Optional[str] = service.EndpointOption,
    ) -> None:
        with service.report_mlflow_errors():
            models = service.registry(endpoint).list(filter_string, max_results)
        table = Table("name", "latest", "latest tags", "description")
        for model in models:
            latest = max(model.latest_versions or [], key=lambda v: int(v.version), default=None)
            table.add_row(model.name, str(latest.version) if latest else "-", str(latest.tags) if latest else "", model.description or "")
        service.console.print(table)

    return app


# ---------------------------------------------------------------------------------------------------------------------
# datasets (`dataset` group)
# ---------------------------------------------------------------------------------------------------------------------
def build_dataset_app(cfg: DictConfig) -> typer.Typer:
    """Build the `dataset` group: create, get, list, extend, update, delete, and snapshot evaluation datasets."""
    app = typer.Typer(help="Evaluation datasets: create, extend, tag, delete, and snapshot.", rich_markup_mode="rich")
    service = DatasetService(cfg)

    @app.command(name="create", help="Create a dataset, optionally seeded from a CSV/JSON/Excel file.")
    def create_dataset(
        name:         str                 = typer.Argument(..., help="Dataset name."),
        file:         Optional[Path]      = typer.Option(None, "--file", exists=True, dir_okay=False, help="CSV/JSON/Excel file of initial rows; omit for none."),
        input_key:    Optional[List[str]] = typer.Option(None, "--input-key", help="Column holding record inputs (required with --file); repeatable."),
        output_key:   Optional[List[str]] = service.OutputKeyOption,
        metadata_key: Optional[List[str]] = service.MetadataKeyOption,
        description:  Optional[str]       = typer.Option(None, help="Human-readable dataset description."),
        tag:          Optional[List[str]] = service.DatasetTagOption,
        endpoint:     Optional[str]       = service.EndpointOption,
    ) -> None:
        df_records = service.read_records(file) if file is not None else None
        options: Dict[str, Any] = {
            "output_keys":   output_key or None,
            "description":   description,
            "metadata_keys": metadata_key or None,
            "dataset_tags":  service.parse_tags(tag or []) or None,
        }
        with service.report_mlflow_errors():
            try:
                service.registry(endpoint).create(name, df_records, input_key or None, **{k: v for k, v in options.items() if v is not None})
            except ValueError as exc:
                raise typer.BadParameter(str(exc))
        service.console.print(f"[green]Created[/green] dataset {name} ({len(df_records) if df_records is not None else 0} records)")

    @app.command(name="get", help="Show a dataset's details and its first records.")
    def get_dataset(
        name:     str           = typer.Argument(..., help="Dataset name."),
        head:     int           = typer.Option(5, help="Number of records to show."),
        endpoint: Optional[str] = service.EndpointOption,
    ) -> None:
        with service.report_mlflow_errors():
            found = service.registry(endpoint).get(name)
            df_records = found.to_df()
        service.console.print(Panel(
            f"id={found.dataset_id}\ndigest={found.digest}\nexperiments={found.experiment_ids}\ntags={service.visible_tags(found.tags)}",
            title=name,
        ))
        service.print_records(df_records, head)

    @app.command(name="list", help="List evaluation datasets.")
    def list_datasets(
        filter_string: Optional[str] = typer.Option(None, "--filter", help="MLflow filter, e.g. \"name LIKE 'support%'\"."),
        max_results:   Optional[int] = typer.Option(None, help="Cap on the number of datasets returned."),
        endpoint:      Optional[str] = service.EndpointOption,
    ) -> None:
        options: Dict[str, Any] = {"filter_string": filter_string, "max_results": max_results}
        with service.report_mlflow_errors():
            found = service.registry(endpoint).list(**{k: v for k, v in options.items() if v is not None})
        table = Table("name", "id", "digest", "tags")
        for item in found:
            table.add_row(item.name, item.dataset_id, str(item.digest), str(service.visible_tags(item.tags)))
        service.console.print(table)

    @app.command(name="extend", help="Add rows from a CSV/JSON/Excel file to an existing dataset (upsert; no new version).")
    def extend_dataset(
        name:         str                 = typer.Argument(..., help="Dataset name."),
        file:         Path                = typer.Option(..., "--file", exists=True, dir_okay=False, help="CSV/JSON/Excel file of rows to add."),
        input_key:    List[str]           = typer.Option(..., "--input-key", help="Column holding record inputs; repeatable."),
        output_key:   Optional[List[str]] = service.OutputKeyOption,
        metadata_key: Optional[List[str]] = service.MetadataKeyOption,
        tag:          Optional[List[str]] = service.DatasetTagOption,
        endpoint:     Optional[str]       = service.EndpointOption,
    ) -> None:
        df_records = service.read_records(file)
        options: Dict[str, Any] = {
            "output_keys":   output_key or None,
            "metadata_keys": metadata_key or None,
            "dataset_tags":  service.parse_tags(tag or []) or None,
        }
        with service.report_mlflow_errors():
            service.registry(endpoint).extend(name, df_records, input_key, **{k: v for k, v in options.items() if v is not None})
        service.console.print(f"[green]Extended[/green] dataset {name} with {len(df_records)} rows")

    @app.command(name="update", help="Set or remove dataset tags without touching records.")
    def update_dataset(
        name:       str                 = typer.Argument(..., help="Dataset name."),
        tag:        Optional[List[str]] = service.DatasetTagOption,
        remove_tag: Optional[List[str]] = typer.Option(None, "--remove-tag", help="Dataset tag key to remove; repeatable."),
        endpoint:   Optional[str]       = service.EndpointOption,
    ) -> None:
        changes: Dict[str, Optional[str]] = {**service.parse_tags(tag or []), **{key: None for key in remove_tag or []}}
        if not changes:
            raise typer.BadParameter("Pass at least one of --tag or --remove-tag")
        with service.report_mlflow_errors():
            service.registry(endpoint).update(name, changes)
        service.console.print(f"[green]Updated[/green] tags on {name}")

    @app.command(name="delete", help="Delete a dataset, or only the given records and/or tags.")
    def delete_dataset(
        name:      str                 = typer.Argument(..., help="Dataset name."),
        record_id: Optional[List[str]] = typer.Option(None, "--record-id", help="Delete just this record; repeatable. The dataset is kept."),
        tag_key:   Optional[List[str]] = typer.Option(None, "--tag-key", help="Delete just this dataset tag key; repeatable. The dataset is kept."),
        yes:       bool                = typer.Option(False, "--yes", help="Skip the confirmation when deleting a whole dataset."),
        endpoint:  Optional[str]       = service.EndpointOption,
    ) -> None:
        if not record_id and not tag_key and not yes:
            typer.confirm(f"Delete dataset {name!r} and all its records?", abort=True)
        with service.report_mlflow_errors():
            service.registry(endpoint).delete(name, record_ids=record_id or None, dataset_tags=tag_key or None)
        service.console.print(f"[green]Deleted[/green] {'parts of ' if record_id or tag_key else ''}dataset {name}")

    @app.command(name="snapshot", help="Freeze the dataset's current records under a label.")
    def snapshot_dataset(
        name:     str           = typer.Argument(..., help="Dataset name."),
        label:    str           = typer.Argument(..., help="Snapshot label, e.g. v1."),
        endpoint: Optional[str] = service.EndpointOption,
    ) -> None:
        with service.report_mlflow_errors():
            written = service.registry(endpoint).snapshot(name, label)
        table = Table("tag", "value", title=f"snapshot {label} of {name}")
        for key, value in written.items():
            table.add_row(key, value)
        service.console.print(table)

    @app.command(name="list-snapshots", help="List a dataset's snapshot labels.")
    def list_snapshots(
        name:     str           = typer.Argument(..., help="Dataset name."),
        endpoint: Optional[str] = service.EndpointOption,
    ) -> None:
        with service.report_mlflow_errors():
            labels = service.registry(endpoint).list_snapshots(name)
        table = Table("snapshot")
        for label in labels:
            table.add_row(label)
        service.console.print(table)

    @app.command(name="load-snapshot", help="Show the first records of a snapshot.")
    def load_snapshot(
        name:     str           = typer.Argument(..., help="Dataset name."),
        label:    str           = typer.Argument(..., help="Snapshot label."),
        head:     int           = typer.Option(5, help="Number of records to show."),
        endpoint: Optional[str] = service.EndpointOption,
    ) -> None:
        with service.report_mlflow_errors():
            try:
                df_records = service.registry(endpoint).load_snapshot(name, label)
            except ValueError as exc:
                raise typer.BadParameter(str(exc))
        service.print_records(df_records, head)

    return app


def build_app(cfg: DictConfig) -> typer.Typer:
    """Build the Typer app: prompt commands at the top level, plus `artifact` and `dataset` groups."""
    app = typer.Typer(name=cfg.app.info.name, help=cfg.app.info.help, rich_markup_mode="rich", add_completion=False)
    app.add_typer(build_prompt_app(cfg))
    app.add_typer(build_artifact_app(cfg), name="artifact")
    app.add_typer(build_dataset_app(cfg), name="dataset")
    return app


if __name__ == "__main__":
    root = Path(read_root_dir())
    build_app(_load_cfg(str(root / "examples" / "_conf"), "tracking/mlflow/app_mlflow_registry.yaml", str(root / "config")))()
