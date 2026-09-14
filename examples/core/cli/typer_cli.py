# uv run python examples/core/cli/typer_cli.py --help
# uv run python examples/core/cli/typer_cli.py demo --count 2 --mode active

from __future__ import annotations

import  typer


app = typer.Typer(rich_markup_mode="rich", add_completion=False)


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


@app.command()
def main(
    workspace: str = typer.Argument(..., help="Workspace or task name to process."),
    count:     int = typer.Option(1, help="Number of processing passes."),
    mode:      str = typer.Option("dry-run", help="Processing mode label."),
) -> None:
    """Process a Velari workspace with Typer."""
    typer.echo(render_status("typer", workspace, count, mode))


if __name__ == "__main__":
    app()
