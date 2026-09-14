# uv run python examples/core/cli/argparse_cli.py demo
# uv run python examples/core/cli/argparse_cli.py demo --count 2 --mode active

from __future__ import annotations

import  argparse
from    typing import Optional, Sequence


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


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the example CLI.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    parser = argparse.ArgumentParser(description="Process a Velari workspace with argparse.")
    parser.add_argument("workspace", help="Workspace or task name to process.")
    parser.add_argument("--count", type=int, default=1, help="Number of processing passes.")
    parser.add_argument("--mode", default="dry-run", help="Processing mode label.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the argparse CLI example.

    Args:
        argv (Optional[Sequence[str]]): Command-line arguments after the script name.

    Returns:
        int: Process exit code.
    """
    args = build_parser().parse_args(argv)
    print(render_status("argparse", args.workspace, args.count, args.mode))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
