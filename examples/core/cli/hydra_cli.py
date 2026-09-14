# uv run python examples/core/cli/hydra_cli.py
# uv run python examples/core/cli/hydra_cli.py workspace=demo count=2 mode=active

from __future__ import annotations

import  hydra
from    dataclasses             import dataclass
from    hydra.core.config_store import ConfigStore
from    omegaconf               import DictConfig, OmegaConf
from    typing                  import Any, Dict, cast


@dataclass
class CliConfig:
    """Configuration values accepted by the Hydra CLI example."""

    workspace: str = "demo"
    count:     int = 1
    mode:      str = "dry-run"


def render_status(cli_name: str, workspace: str, count: int, mode: str) -> str:
    """Render a workspace processing status message.

    Args:
        cli_name (str): CLI implementation name.
        workspace (str): Workspace or task name to process.
        count (int): Number of times to process the workspace.
        mode (str): Processing mode label.

    Returns:
        str: Human-readable processing status.
    """
    suffix = "time" if count == 1 else "times"
    return f"{cli_name} processed workspace '{workspace}' {count} {suffix} in {mode} mode."


def read_config(config: DictConfig) -> CliConfig:
    """Convert Hydra's DictConfig into the example dataclass.

    Args:
        config (DictConfig): Runtime Hydra config.

    Returns:
        CliConfig: Structured CLI configuration.
    """
    resolved = cast(Dict[str, Any], OmegaConf.to_container(config, resolve=True))
    return CliConfig(
        workspace=str(resolved["workspace"]),
        count=int(resolved["count"]),
        mode=str(resolved["mode"]),
    )


ConfigStore.instance().store(name="config", node=CliConfig)


@hydra.main(config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Process a Velari workspace with Hydra."""
    cli_config = read_config(config)
    print(render_status("hydra", cli_config.workspace, cli_config.count, cli_config.mode))


if __name__ == "__main__":
    main()
