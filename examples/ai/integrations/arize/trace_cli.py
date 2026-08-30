# uv run --extra evals python examples/ai/integrations/arize/trace_cli.py
# uv run --extra evals python examples/ai/integrations/arize/trace_cli.py --nested
# uv run --extra evals python examples/ai/integrations/arize/trace_cli.py --endpoint http://localhost:6006
# See examples/ai/integrations/arize/README.md for details.

import logging
from pathlib import Path
from typing import Optional

import typer
from omegaconf import DictConfig
from rich.console import Console
from rich.panel import Panel

from velari_core.core import read_root_dir
from velari_core.core.io.partition.hydra import read_hydra_compose
from velari_ai.integrations.arize.connection import Connector, ConnectorConfig

logger  = logging.getLogger(__name__)
console = Console(stderr=True, color_system="auto", force_terminal=True)


def _load_cfg(primary_config_dir: str, config_name: str, searchpath_dir: str) -> DictConfig:
    cfg, _ = read_hydra_compose(
        primary_config_dir,
        config_name,
        overrides=[f"hydra.searchpath=[file://{searchpath_dir}]"],
    )
    return cfg


def _configure_app(app: typer.Typer, examples_conf: str, config_dir: str, config_name: str) -> DictConfig:
    """Load config once, populate the Typer app's name/help from app.info.*, and return it for reuse."""
    cfg = _load_cfg(examples_conf, config_name, config_dir)
    app.info.name = cfg.app.info.name
    app.info.help = cfg.app.info.help
    return cfg


def main(cfg: DictConfig, nested: bool, endpoint: Optional[str]) -> None:
    """Start Phoenix from the given config and record an example trace span."""
    if endpoint:
        # Production pattern: connect to an already-running Phoenix instead of starting one
        # in-process — no ThreadSession, so the UI's lifetime isn't tied to this script's.
        config = ConnectorConfig(
            remote  = ConnectorConfig.Remote(endpoint=endpoint),
            project = ConnectorConfig.Project(project_name=cfg.app.info.name),
        )
        connector = Connector(config)
    else:
        connector = Connector.from_config(cfg)
    logger.info(f"Phoenix UI available at {connector.url}")

    console.print(Panel(f"[bold cyan]{cfg.app.info.name}[/bold cyan]", expand=False))
    console.print(f"[dim]Phoenix UI → {connector.url}[/dim]")

    tracer = connector.get_tracer(cfg.app.info.name, str(cfg.app.info.version))
    if nested:
        with tracer.start_as_current_span("example-parent-span") as parent_span:
            parent_span.set_attribute("example.attribute", "hello-phoenix-parent")
            with tracer.start_as_current_span("example-child-span") as child_span:
                child_span.set_attribute("example.attribute", "hello-phoenix-child")
        logger.info("Recorded a parent-child span pair — check the Phoenix UI to see the hierarchy.")
        console.print(
            "[green]Recorded a parent-child span pair[/green] — check the Phoenix UI to see the hierarchy."
        )
    else:
        with tracer.start_as_current_span("example-span") as span:
            span.set_attribute("example.attribute", "hello-phoenix")
        logger.info("Recorded one example span — check the Phoenix UI to see it.")
        console.print("[green]Recorded one example span[/green] — check the Phoenix UI to see it.")


if __name__ == "__main__":
    root          = Path(read_root_dir())
    examples_conf = str(root / "examples" / "_conf")
    config_dir    = str(root / "config")

    app = typer.Typer(rich_markup_mode="rich", add_completion=False)
    cfg = _configure_app(app, examples_conf, config_dir, "app_arize.yaml")

    def run(
        nested: bool = typer.Option(
            False, help="Record a parent-child span pair instead of a single flat span."
        ),
        endpoint: Optional[str] = typer.Option(
            None, help="Connect to an already-running Phoenix instance instead of starting one locally."
        ),
    ) -> None:
        main(cfg, nested, endpoint)

    app.command(help="Start Phoenix from the given config and record an example trace span.")(run)
    app(standalone_mode=False)
