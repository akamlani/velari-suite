# Stage 1: Package creation and scaffolding

**Status: implemented and verified.**

Part of [plan.md](plan.md). Goal: an empty but installable, type-checked, lint-clean `velari-harness` package wired into the workspace. No harness logic yet.

## Files to create

Under `packages/velari-harness/`:

```
├── Makefile                  # header + help/info only (see below)
├── README.md                 # status, concepts, install, develop, layout (see below)
├── pyproject.toml
└── velari_harness/
    └── __init__.py           # __version__ from velari-core's shared helper, plus its own logger
```

### `pyproject.toml`

Mirror `velari-ai`'s metadata:

- `name = "velari-harness"`, version `0.1.0`, Python `>=3.12`, Apache-2.0, authors as in the sibling packages.
- `dependencies = ["velari-core", "velari-ai", "omegaconf>=2.3", "pandas>=2.0"]`.
- `[tool.uv.sources]`: `velari-core` and `velari-ai` set to `{ workspace = true }`.
- Setuptools build, `include = ["velari_harness*"]`.
- No extras yet. Runtime extras are added once the runtimes are defined (see the next subsection). Typer for the CLI is added in stage 6; it is already in the root dev group.

### Runtime extras (placeholder; optional for the first pass)

The optional-dependencies table is for **runtime backends of the harness itself**: the environment a harness executes in, such as a distributed or orchestrated runner. It is not for forwarding `velari-ai`'s dependencies; heavy third-party packages stay in `velari-ai`'s own extras, and harness types (`datascience`, `evaluation`) are modules inside the package, always available.

The runtimes are not defined yet, so stage 1 declares none. `pyproject.toml` carries a comment saying how to add one:

```toml
# Runtime extras are added here once the harness runtimes are defined, one extra per execution
# backend that owns its own third-party packages, e.g. `ray = ["ray>=2.0"]`, installed with
# `velari-harness[ray]`. The default local runner needs no extra.
```

- Each runtime extra lists only that backend's own packages, for example `ray = ["ray>=2.0"]`.
- The default local, in-process runner needs no extra.
- Code convention for later stages: a module that needs a runtime's package lives in its own module and imports it at its own top (no `TYPE_CHECKING` guards, per `import_guides.md`). The registry imports such a runtime on demand and raises an error that names the extra to install (`pip install velari-harness[<runtime>]`). Importing `velari_harness` itself never requires an extra.

### `__init__.py`

No per-package `version.py`. The version lookup lives once in `velari-core` (`velari_core/core/utils/package_utils.py`, `package_version`) and every package's `__init__.py` calls it:

```python
import logging

from velari_core.core.utils.package_utils import package_version

__version__ = package_version(__name__)
logger = logging.getLogger(__name__)

__all__ = ["__version__", "logger"]
```

- Importing from `velari_core` runs its `setup_logging()`, so the old `from velari_core import logger as _  # noqa: F401` trick is not needed.
- `logger` stays per package, named by `__name__`.
- No subpackage `__init__.py` files.

`package_version` resolves the version from installed metadata, then from the nearest `pyproject.toml` above the package's own source, then `"0.0.0"`. Core's `watermark` and `platform_sys` moved into the same module. The helper was introduced after the first scaffold, which had copied `version.py` from the sibling packages; `velari-core`, `velari-data` and `velari-ai` were switched to it at the same time.

### `README.md`

More than the one-line sibling READMEs, because this package will host several harnesses. Sections: a one-sentence description, a status note (scaffold, linking to this plan), planned concepts (harness, stage, registry, runtime), planned harnesses, install, develop commands, and the current layout. It describes only what exists or is marked as planned, and is updated as stages land.

### `Makefile`

Take only the header from `packages/velari-ai/Makefile`: the `runtime.env`/`python.env` includes, `PACKAGE_NAME := velari-harness`, shell detection, and the `help` and `info` targets. Leave out the `list_mcp` and `install_mcp_servers` targets, which don't apply here.

## Workspace wiring

`members = ["packages/*"]` already picks the package up, so no member change is needed. Update:

1. Root `pyproject.toml`: add `packages/velari-harness` to `[tool.pyright] extraPaths` and `include`, and to `[tool.ty.environment] extra-paths`.
2. Root `README.md`: add the package to the `packages/` tree and add `harness/` to the `tests/` tree.
3. `AGENTS.md`, three places:
   - the Project line that lists the workspace members;
   - the Key Files table (add `packages/velari-harness/pyproject.toml`);
   - the Package Structure paragraph with the import-name examples (add `velari-harness` → `velari_harness`), plus a short paragraph naming the shared bootstrap convention (see the next section).
4. Run `uv sync` with the same command `make install_python` uses, so extras match, to regenerate `uv.lock` and the shared `.venv`. The `uv.lock` diff should contain only the new package.

`AGENTS.md` and the root `README.md` already have unrelated uncommitted changes, so edit them carefully on top of those.

Out of scope:
- The root `Makefile` `install_kernel` target installs only `velari-core` and `velari-data` (not `velari-ai`), so adding the harness there would be inconsistent. Leave it alone.
- No `.github/workflows` change: no workflow references the sibling packages by name.

## Shared bootstrap in `velari-core`

The first scaffold copied `version.py` and `__init__.py` from `velari-ai`; all four packages then held a byte-identical `version.py`. That logic now lives once in `velari-core`, so new packages need only a few lines and fixes happen in one place.

| Change | Path |
|---|---|
| New helper: `package_version(package)`, plus `watermark` and `platform_sys` moved from `version.py` | `packages/velari-core/velari_core/core/utils/package_utils.py` |
| `velari_core/__init__.py` calls the helper through a relative import | `packages/velari-core/velari_core/__init__.py` |
| `velari_data`, `velari_ai`, `velari_harness` use the four-statement `__init__.py` shown above | `packages/velari-*/velari_*/__init__.py` |
| Delete the per-package `version.py` | all four packages |
| Tests: new helper tests; the four `test_version` tests import `__version__` from the package | `tests/core/test_package_utils.py`, `tests/{core,data,ai,harness}/test_velari_*.py` |
| Document the convention for new packages | `AGENTS.md` "Package Structure" |

- `package_version` anchors its `pyproject.toml` fallback on the calling package's own source location, so a package never reports `velari-core`'s version.
- `tests/core/test_package_utils.py` covers: installed metadata for core and ai, a dotted name using its top-level package, a source-only package reading its own `pyproject.toml`, and an unknown name returning `"0.0.0"`.
- Behavior of `__version__` and logging is unchanged for all four packages.

## Tests

Create `tests/harness/test_velari_harness.py` with the same four tests as `tests/core/test_velari_core.py` and `tests/ai/test_velari_ai.py`:

- the package imports;
- `velari_harness.__version__` is a `str`;
- `velari_harness.logger` is a `logging.Logger`;
- the logger emits a record, checked with `caplog`.

Module docstring: `"""Tests for the velari_harness package."""`. The sibling test directories have no `__init__.py`, so add none.

## Verification

- `uv sync` resolves cleanly.
- `uv tree --package velari-harness --depth 1` lists `velari-core`, `velari-ai`, pandas and omegaconf only.
- `uv run python -c "import velari_harness; print(velari_harness.__version__)"` prints `0.1.0`; the same check on `velari_core`, `velari_data` and `velari_ai` prints `0.1.0` too.
- `uv run pytest tests/core/test_package_utils.py` passes, and no `version.py` or `.version import` reference is left in `.py`, `.ipynb` or docs.
- `make typecheck` (or `uv run pyright packages/velari-harness`) is clean.
- `make lint` and `make format` are clean.
- `uv run pytest tests/harness` passes. The full `uv run pytest` shows only pre-existing failures.
- `make -C packages/velari-harness info` runs.

## Exit criteria

The package imports from a fresh `uv sync`, the workspace config lists it, and the docs mention it. No behavior beyond version and logger, and the version/logging bootstrap is shared from `velari-core`.
