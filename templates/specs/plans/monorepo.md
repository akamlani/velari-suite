# Monorepo Migration: velari-suite → uv Workspace (velari-core)

Executed migration plan for turning `velari-suite` from a single flat Python package into a `uv` workspace monorepo hosting multiple independently-installable packages, starting with `velari-core`. Kept as a reference for adding future workspace members (e.g. `velari-ai`, `velari-data`).

## Context

`velari-suite` was a single flat package (`velari-studios-suite`) with `velari/core/` and `velari/integrations/` at the repo root. The goal was to make this repo host multiple independently-versioned libraries, starting with `velari-core`. Modeled loosely on how LangChain structures `langchain-ai/langchain` (a `uv`-workspace monorepo where `langchain-core`, `langchain-community`, etc. are independent top-level distributions) — but for this first pass, `velari-core` consolidates **both** existing subpackages (`velari/core/` and `velari/integrations/{pydantic,fastapi}/`) into one distribution, rather than splitting them apart.

### Decisions made

- **One distribution for now**: `velari-core` owns the whole `velari.*` dotted namespace (`velari.core`, `velari.integrations.pydantic`, `velari.integrations.fastapi`). Since only one distribution claims the `velari` top-level name, no PEP 420 cross-distribution namespace-package handling is needed — `velari/__init__.py` stays a normal package with its auto-run `setup_logging()` behavior unchanged. Future genuinely-separate libraries (e.g. a future `velari-ai`) would follow LangChain's pattern of an independent top-level import name (`velari_ai`, not `velari.ai` merged in via namespace packages) — that's a decision for when that package is actually created, not resolved here.
- **Workspace directory name**: `packages/` (uv community convention), not `libs/` (which LangChain uses).
- **Root-detection fix**: `read_root_dir()` must not assume "root = N directories above this file." Instead it walks up parsing each candidate `pyproject.toml` for a `[tool.uv.workspace]` table (via `tomllib`) to find the true monorepo root — a nested per-package `pyproject.toml` under `packages/velari-core/` would otherwise falsely look like the root.

## Pre-migration state (verified)

- Root `pyproject.toml`: `name = "velari-studios-suite"`, single `[project].dependencies`, `[tool.setuptools.packages.find] where=["."], include=["velari*","examples*"]`.
- `velari/__init__.py`: auto-ran `setup_logging()` on import; computed `_CONFIG_PATH`/`_LOG_DIR` as `Path(__file__).parent.parent / "config"/"logs"` — hardcoded "repo root is 2 dirs above this file."
- `velari/version.py`: resolved `__version__` via `importlib.metadata.packages_distributions().get(module_name)` (first match) or by walking up for any `pyproject.toml` with a `[project].version`.
- `velari/core/__init__.py`: a re-export shim (`from .experiment import Experiment`, etc.) that violated `.claude/rules/PREFERENCES.md`'s "no `__init__.py` for re-exports" rule, independent of this migration. Depended on by `examples/imports_common.py` and `tests/core/test_hydra_config.py`.
- `velari/core/utils/env_utils.py::read_root_dir()`: walked up for *any* file named `pyproject.toml` — the root cause this migration had to fix.
- `Makefile`: a `uv_sync_project_name` target `sed`-rewrote the root `pyproject.toml`'s `name =` line to `$(PACKAGE_INSTALL_NAME)` on every install — a single-package-rename hack.

## Migration steps (as executed)

1. **Scaffold `packages/velari-core/`** with its own `pyproject.toml`: `[project]` name=`velari-core`, version=`0.1.0`, the dependency subset `velari.core`/`velari.integrations` actually import (numpy, pandas, pydantic, pydantic-core, httpx, pypdf, jinja2, hydra-core, omegaconf, pyyaml, rich, appdirs, python-dotenv, watermark). `[build-system]` = setuptools; `[tool.setuptools.packages.find] include=["velari*"]`. Added a minimal package-local `README.md` (required since `[project].readme` points at it and setuptools needs it to exist to build metadata).

2. **Move code**: `git mv velari/core packages/velari-core/velari/core`, `git mv velari/integrations packages/velari-core/velari/integrations`, `git mv velari/__init__.py packages/velari-core/velari/__init__.py`, `git mv velari/version.py packages/velari-core/velari/version.py`. Deleted the now-empty root `velari/`.

3. **Deleted the re-export shim**: removed `packages/velari-core/velari/core/__init__.py`; updated `examples/imports_common.py` and `tests/core/test_hydra_config.py` (plus a docstring example in `velari/core/io/partition/hydra.py`) to import directly from the defining modules (`velari.core.experiment`, `velari.core.utils.env_utils`) instead of the `velari.core` shim.

4. **Fixed root detection** in both places that depended on directory-depth assumptions:
   - `velari/core/utils/env_utils.py::read_root_dir()` — added a private `_is_workspace_root(pyproject_path)` helper that parses the candidate file with `tomllib` and checks for `"workspace" in pyproject.get("tool", {}).get("uv", {})`; `read_root_dir()` now walks up until this returns true, instead of matching on file existence alone.
   - `velari/__init__.py` — replaced the hardcoded `Path(__file__).parent.parent` for `_CONFIG_PATH`/`_LOG_DIR` with `_ROOT_DIR = Path(read_root_dir())` (imported from `.core.utils.env_utils`), so logging config/log-dir resolution is correct regardless of how deep the package lives under `packages/`.

5. **Converted root `pyproject.toml` to a virtual workspace root**: removed the `[project]` table and `[tool.setuptools.packages.find]` entirely; added:
   ```toml
   [tool.uv.workspace]
   members = ["packages/*"]
   ```
   Kept `[dependency-groups] dev=[...]`, `[tool.ruff]`/`[tool.ruff.lint]`/`[tool.ruff.lint.pydocstyle]`, and `[tool.pytest.ini_options] testpaths=["tests"]` centralized at the root — these apply repo-wide regardless of workspace member count.

6. **Makefile fixes**:
   - Deleted the `uv_sync_project_name` target entirely (and its `.PHONY` entry and call sites in `install_python`/`uv_install_python`) — there's no `name =` line left at the root to rewrite.
   - In `uv_install_python`, dropped `uv pip install -e .` (no root package to install); `uv sync --all-extras --active` installs all `[tool.uv.workspace]` members editable into the shared venv natively.
   - `install_setup`'s `mkdir -p` list: added `packages` so a fresh clone scaffolds the dir.
   - `PACKAGE_INSTALL_NAME` no longer named a real installable package once the root lost its `[project]` table — removed it entirely (from the `Makefile`'s `export` line, `help`/`info`/`install`/`clean_python` echo strings, and `config/runtime/runtime.env`), reworded those echoes to reference the workspace via `$(PACKAGE_NAME)` instead (e.g. `"install : create environment for workspace $(PACKAGE_NAME)"`).

7. **Docs**: updated `AGENTS.md`'s "Package Structure"/"Key Files" tables and `README.md`'s directory tree to describe the workspace model — root `pyproject.toml` as workspace config, `packages/velari-core/pyproject.toml` as package metadata, and the `packages/velari-core/velari/{core,integrations}/` layout.

Tests stayed centralized at root `tests/` (unchanged location/config) — imports (`velari.core.*`) didn't change, so no test file moves were required beyond the shim-import fixes in step 3.

8. **Fixed Pylance/Pyright import resolution** (follow-on issue discovered after the migration): once `velari/` moved under `packages/velari-core/`, Pylance stopped resolving `from velari...`/`import velari` in `tests/` and `examples/`, even though `uv sync` installed `velari-core` correctly and pytest/runtime imports worked fine. Root cause: `uv`/setuptools installs workspace members as PEP 660 "new-style" editable installs — a `MetaPathFinder`-based `.pth` (`__editable___velari_core_0_1_0_finder.py`) that remaps `velari` → `packages/velari-core/velari` at import time by executing Python code. Pyright/Pylance don't execute that hook; they only read plain `sys.path`-style directories, so they never learn the real package location once it moved out of the repo root. Fix: added
   ```toml
   [tool.pyright]
   extraPaths = ["packages/velari-core"]
   ```
   to the root `pyproject.toml` — not `.vscode/settings.json`, which is a symlink shared across projects via `_build/dotfiles/` and shouldn't carry project-specific paths.

## Verification (results)

- `uv sync` from repo root — workspace resolved, `velari-core` built and installed editable into `.venv`.
- `uv run pytest` — all 7 existing tests passed, including `tests/core/test_velari.py` (`import velari`, `from velari.version import __version__`, `from velari import logger`) and the updated `tests/core/test_hydra_config.py`.
- Manual check: `from velari.core.experiment import Experiment`, `from velari.integrations.pydantic import tools`, and `read_root_dir()` all resolved correctly — `read_root_dir()` returned the true repo root (not `packages/velari-core/`) even when called from a module nested three levels under `packages/velari-core/`.
- `make test`, `make help`, `make info` all ran cleanly with the updated wording.
- `uv run ruff check .` — one pre-existing long-docstring-line warning in `filesystem.py`, unrelated to this migration (file was moved, not edited).
- **Not run**: full `make install`/`make install_python` — those clone/pull the dotfiles repo and register a global ipykernel spec, side effects beyond what's needed to verify this change. Worth running manually before relying on a completely fresh clone.

## Applying this pattern to a new package

To add another workspace member (e.g. `velari-ai`):
1. Create `packages/velari-ai/pyproject.toml` with its own `[project]` name/deps and `[tool.setuptools.packages.find] include=["velari*"]`.
2. Decide namespace strategy up front: if `velari-ai` should share the `velari.*` dotted namespace with `velari-core` (i.e. `velari.ai`), note that this now requires real PEP 420 cross-distribution namespace-package handling — `velari-core`'s `velari/__init__.py` currently owns the real `velari/__init__.py`, so a second distribution can't also ship one. Resolve this before implementing (see the "Namespace strategy" discussion referenced in this migration — it was deferred here because only one distribution existed). Alternatively, follow the LangChain precedent and give it a fully independent top-level name (`velari_ai`) to sidestep the issue entirely.
3. No Makefile or root `pyproject.toml` `[tool.uv.workspace]` changes needed — `members = ["packages/*"]` already picks up any new `packages/<name>/`.
4. Add package-specific tests under `tests/<name>/` (tests stay centralized at the root).
5. Append the new package's path to `[tool.pyright] extraPaths` in the root `pyproject.toml` (see step 8 above) — Pyright's `extraPaths` doesn't support globs, so `packages/*` there won't auto-discover new members the way `[tool.uv.workspace]` does; each new package needs its own entry or Pylance won't resolve its imports in `tests/`/`examples/`.
