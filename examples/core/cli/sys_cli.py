# uv run python examples/core/cli/sys_cli.py demo
# uv run python examples/core/cli/sys_cli.py demo --count 2 --mode active

from __future__ import annotations

import  sys
from    typing import Sequence, Tuple


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


def parse_args(argv: Sequence[str]) -> Tuple[str, int, str]:
    """Parse command-line arguments with simple sys.argv-style lookups.

    Args:
        argv (Sequence[str]): Command-line arguments after the script name.

    Returns:
        Tuple[str, int, str]: Parsed workspace, count, and mode values.

    Raises:
        ValueError: If required or typed arguments are invalid.
    """
    if not argv or argv[0] in {"-h", "--help"}:
        raise ValueError("usage: sys_cli.py WORKSPACE [--count N] [--mode MODE]")

    # First positional value is the workspace name.
    workspace = argv[0]

    # Optional flags use list lookup, keeping the sys example intentionally small.
    try:
        count = int(argv[argv.index("--count") + 1]) if "--count" in argv else 1
        mode  = argv[argv.index("--mode") + 1] if "--mode" in argv else "dry-run"
    except (IndexError, ValueError) as exc:
        raise ValueError("usage: sys_cli.py WORKSPACE [--count N] [--mode MODE]") from exc

    return workspace, count, mode


def main(argv: Sequence[str]) -> int:
    """Run the sys.argv CLI example.

    Args:
        argv (Sequence[str]): Command-line arguments after the script name.

    Returns:
        int: Process exit code.
    """
    try:
        workspace, count, mode = parse_args(argv)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    print(render_status("sys", workspace, count, mode))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
