# uv run python examples/ai/integrations/mlflow/registry_cli.py register ticket-triage --template "Classify: {{ticket_text}}" --tag team=support
# uv run python examples/ai/integrations/mlflow/registry_cli.py set-alias ticket-triage production 1
# uv run python examples/ai/integrations/mlflow/registry_cli.py load ticket-triage --alias production
# uv run python examples/ai/integrations/mlflow/registry_cli.py list --filter "name LIKE 'ticket%'"
# uv run python examples/ai/integrations/mlflow/registry_cli.py list --endpoint http://localhost:5001
# uv run python examples/ai/integrations/mlflow/registry_cli.py artifact register models:/m-abc123 billing-agent --tag team=support
# uv run python examples/ai/integrations/mlflow/registry_cli.py artifact load billing-agent --alias champion --input '[{"question": "..."}]'
# uv run python examples/ai/integrations/mlflow/registry_cli.py artifact list --filter "name LIKE 'billing%'"
# See examples/ai/integrations/mlflow/README.md for details.

import  json
import  logging
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
from    velari_core.core.io.partition.hydra import read_hydra_compose
from    velari_ai.integrations.mlflow.connection import Connector, ConnectorConfig
from    velari_ai.integrations.mlflow.registry import ArtifactLoader, PromptRegistry

logger  = logging.getLogger(__name__)
console = Console(stderr=True, color_system="auto", force_terminal=True)

EndpointOption = typer.Option(None, "--endpoint", help="Connect to an already-running MLflow tracking server instead of the local store.")


def _load_cfg(primary_config_dir: str, config_name: str, searchpath_dir: str) -> DictConfig:
    cfg, _ = read_hydra_compose(
        primary_config_dir,
        config_name,
        overrides=[f"hydra.searchpath=[file://{searchpath_dir}]"],
    )
    return cfg


def _connector(cfg: DictConfig, endpoint: Optional[str]) -> Connector:
    """Build a Connector to a remote tracking server (`endpoint`) or the config's local store."""
    connector = Connector(ConnectorConfig(remote=ConnectorConfig.Remote(endpoint=endpoint))) if endpoint else Connector.from_config(cfg)
    console.print(f"[dim]MLflow tracking → {connector.url}[/dim]")
    return connector


def _connect(cfg: DictConfig, endpoint: Optional[str]) -> PromptRegistry:
    return PromptRegistry(_connector(cfg, endpoint))


@contextmanager
def _report_mlflow_errors() -> Generator[None, None, None]:
    try:
        yield
    except MlflowException as exc:
        console.print(f"[red]{exc.message}[/red]")
        raise typer.Exit(code=1)


def _parse_tags(tags: List[str]) -> Dict[str, str]:
    try:
        return dict(tag.split("=", 1) for tag in tags)
    except ValueError:
        raise typer.BadParameter("--tag must be KEY=VALUE")


def build_app(cfg: DictConfig) -> typer.Typer:
    """Build the Typer app: one subcommand per PromptRegistry method, plus an `artifact` group for ArtifactLoader."""
    app = typer.Typer(name=cfg.app.info.name, help=cfg.app.info.help, rich_markup_mode="rich", add_completion=False)

    @app.command(help="Register a prompt, creating it or adding a new version.")
    def register(
        name:           str                = typer.Argument(..., help="Prompt name."),
        template:       Optional[str]      = typer.Option(None, help="Prompt text, with {{variables}}."),
        template_file:  Optional[Path]     = typer.Option(None, help="Read the prompt text from this file instead of --template."),
        commit_message: Optional[str]      = typer.Option(None, help="Description of this version's change."),
        tag:            Optional[List[str]] = typer.Option(None, help="Version tag as KEY=VALUE; repeatable."),
        endpoint:       Optional[str]      = EndpointOption,
    ) -> None:
        if (template is None) == (template_file is None):
            raise typer.BadParameter("Pass exactly one of --template or --template-file")
        text = template if template is not None else Path(str(template_file)).read_text()
        with _report_mlflow_errors():
            version = _connect(cfg, endpoint).register(name, text, commit_message=commit_message, tags=_parse_tags(tag or []) or None)
        console.print(f"[green]Registered[/green] {name} version {version.version}")

    @app.command(help="Load a prompt by version, by alias, or its latest version.")
    def load(
        name:     str           = typer.Argument(..., help="Prompt name."),
        version:  Optional[int] = typer.Option(None, help="Version number; not combinable with --alias."),
        alias:    Optional[str] = typer.Option(None, help="Alias to resolve, e.g. production."),
        endpoint: Optional[str] = EndpointOption,
    ) -> None:
        with _report_mlflow_errors():
            try:
                prompt = _connect(cfg, endpoint).load(name, version=version, alias=alias)
            except ValueError as exc:
                raise typer.BadParameter(str(exc))
        console.print(Panel(
            str(prompt.template),
            title=f"{prompt.name} v{prompt.version}",
            subtitle=f"aliases={prompt.aliases} · {prompt.commit_message or 'no commit message'}",
        ))

    @app.command(name="list", help="List registered prompts.")
    def list_prompts(
        filter_string: Optional[str] = typer.Option(None, "--filter", help="MLflow filter, e.g. \"name LIKE 'ticket%'\"."),
        max_results:   Optional[int] = typer.Option(None, help="Cap on the number of prompts returned."),
        endpoint:      Optional[str] = EndpointOption,
    ) -> None:
        with _report_mlflow_errors():
            prompts = _connect(cfg, endpoint).list(filter_string, max_results)
        table = Table("name", "tags", "description")
        for prompt in prompts:
            table.add_row(prompt.name, str({k: v for k, v in prompt.tags.items() if not k.startswith("_mlflow")}), prompt.description or "")
        console.print(table)

    @app.command(name="set-alias", help="Point an alias at a prompt version, moving it if it already exists.")
    def set_alias(
        name:     str           = typer.Argument(..., help="Prompt name."),
        alias:    str           = typer.Argument(..., help="Alias to set, e.g. staging."),
        version:  int           = typer.Argument(..., help="Version the alias should resolve to."),
        endpoint: Optional[str] = EndpointOption,
    ) -> None:
        with _report_mlflow_errors():
            _connect(cfg, endpoint).set_alias(name, alias, version)
        console.print(f"[green]Alias[/green] {alias} → {name} v{version}")

    @app.command(name="delete-alias", help="Remove an alias; the prompt version is kept.")
    def delete_alias(
        name:     str           = typer.Argument(..., help="Prompt name."),
        alias:    str           = typer.Argument(..., help="Alias to remove."),
        endpoint: Optional[str] = EndpointOption,
    ) -> None:
        with _report_mlflow_errors():
            _connect(cfg, endpoint).delete_alias(name, alias)
        console.print(f"[green]Deleted alias[/green] {alias} from {name}")

    @app.command(name="set-tag", help="Set a prompt-level tag.")
    def set_tag(
        name:     str           = typer.Argument(..., help="Prompt name."),
        key:      str           = typer.Argument(..., help="Tag key."),
        value:    str           = typer.Argument(..., help="Tag value."),
        endpoint: Optional[str] = EndpointOption,
    ) -> None:
        with _report_mlflow_errors():
            _connect(cfg, endpoint).set_tag(name, key, value)
        console.print(f"[green]Tagged[/green] {name} {key}={value}")

    @app.command(name="delete-tag", help="Remove a prompt-level tag.")
    def delete_tag(
        name:     str           = typer.Argument(..., help="Prompt name."),
        key:      str           = typer.Argument(..., help="Tag key."),
        endpoint: Optional[str] = EndpointOption,
    ) -> None:
        with _report_mlflow_errors():
            _connect(cfg, endpoint).delete_tag(name, key)
        console.print(f"[green]Removed tag[/green] {key} from {name}")

    artifact = typer.Typer(help="Registered GenAI apps/agents (logged models): register, load, and list.", rich_markup_mode="rich")
    app.add_typer(artifact, name="artifact")

    @artifact.command(name="register", help="Register a logged model, creating it or adding a new version.")
    def register_artifact(
        model_uri: str                 = typer.Argument(..., help="Logged model URI, e.g. models:/m-abc123 or runs:/<run_id>/agent."),
        name:      str                 = typer.Argument(..., help="Registered name."),
        tag:       Optional[List[str]] = typer.Option(None, help="Version tag as KEY=VALUE; repeatable."),
        endpoint:  Optional[str]       = EndpointOption,
    ) -> None:
        with _report_mlflow_errors():
            version = ArtifactLoader(_connector(cfg, endpoint)).register(model_uri, name, tags=_parse_tags(tag or []) or None)
        console.print(f"[green]Registered[/green] {name} version {version.version}")

    @artifact.command(name="load", help="Load a registered app by version, by alias, or its latest version; optionally run it.")
    def load_artifact(
        name:       str           = typer.Argument(..., help="Registered name."),
        version:    Optional[int] = typer.Option(None, help="Version number; not combinable with --alias."),
        alias:      Optional[str] = typer.Option(None, help="Alias to resolve, e.g. champion."),
        input_json: Optional[str] = typer.Option(None, "--input", help="JSON passed to the model's predict(), e.g. '[1, 2]'."),
        endpoint:   Optional[str] = EndpointOption,
    ) -> None:
        data: Any = None
        try:
            data = json.loads(input_json) if input_json is not None else None
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"--input must be valid JSON: {exc}")
        with _report_mlflow_errors():
            try:
                model = ArtifactLoader(_connector(cfg, endpoint)).load(name, version=version, alias=alias)
            except ValueError as exc:
                raise typer.BadParameter(str(exc))
            console.print(Panel(
                f"model_id={model.model_id}\nflavors={list(model.metadata.flavors)}\nsignature={model.metadata.signature}",
                title=f"{name} loaded",
            ))
            if input_json is not None:
                console.print(model.predict(data))

    @artifact.command(name="list", help="List registered apps.")
    def list_artifacts(
        filter_string: Optional[str] = typer.Option(None, "--filter", help="MLflow filter, e.g. \"name LIKE 'billing%'\"."),
        max_results:   Optional[int] = typer.Option(None, help="Cap on the number of apps returned."),
        endpoint:      Optional[str] = EndpointOption,
    ) -> None:
        with _report_mlflow_errors():
            models = ArtifactLoader(_connector(cfg, endpoint)).list(filter_string, max_results)
        table = Table("name", "latest", "latest tags", "description")
        for model in models:
            latest = max(model.latest_versions or [], key=lambda v: int(v.version), default=None)
            table.add_row(model.name, str(latest.version) if latest else "-", str(latest.tags) if latest else "", model.description or "")
        console.print(table)

    return app


if __name__ == "__main__":
    root = Path(read_root_dir())
    build_app(_load_cfg(str(root / "examples" / "_conf"), "tracking/mlflow/app_mlflow_registry.yaml", str(root / "config")))()
