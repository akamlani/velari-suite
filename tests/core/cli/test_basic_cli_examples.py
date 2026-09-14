from __future__ import annotations

import  py_compile
import  subprocess
import  sys
from    pathlib import Path
from    typing  import Tuple


ROOT_DIR = Path(__file__).resolve().parents[3]
CLI_DIR  = ROOT_DIR / "examples" / "core" / "cli"


def test_basic_cli_examples_compile() -> None:
    """Compile each basic CLI example."""
    script_names: Tuple[str, ...] = ("sys_cli.py", "argparse_cli.py", "typer_cli.py", "hydra_cli.py")
    for script_name in script_names:
        py_compile.compile(str(CLI_DIR / script_name), doraise=True)


def test_sys_cli_processes_workspace() -> None:
    """Run the sys.argv example with a workspace name."""
    result = subprocess.run(
        [sys.executable, str(CLI_DIR / "sys_cli.py"), "demo"],
        capture_output=True,
        check=True,
        text=True,
    )

    assert "sys processed workspace 'demo' 1 time in dry-run mode." in result.stdout


def test_argparse_cli_processes_workspace_with_count() -> None:
    """Run the argparse example with an explicit count."""
    result = subprocess.run(
        [sys.executable, str(CLI_DIR / "argparse_cli.py"), "demo", "--count", "2"],
        capture_output=True,
        check=True,
        text=True,
    )

    assert "argparse processed workspace 'demo' 2 times in dry-run mode." in result.stdout
