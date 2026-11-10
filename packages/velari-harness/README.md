# velari-harness

Harnesses for the Velari suite (`velari_harness`). A harness is a repeatable, configurable run of a workflow, built from small reusable stages. Depends on `velari-core` and `velari-ai`.

> **Status: scaffold.** The package installs and imports (`velari_harness.__version__`), but the harness framework and the harnesses below are not built yet. See the [implementation plan](../../docs/packages/velari-harness/plan.md).

## Concepts (planned)

- **Harness**: an ordered list of stages that share one context. A new harness is a subclass plus a stage list.
- **Stage**: one named, reusable unit of work, such as `load`, `fit` or `score`. The same stage can appear in more than one harness.
- **Registry**: harnesses and stages register by name, so they can be built from an OmegaConf config and extended without editing this package.
- **Runtime**: the environment a harness executes in. The default is a local, in-process runner. Other runtimes will be optional extras.

## Planned harnesses

| Harness | Purpose |
|---|---|
| `datascience` | Load, profile, split, featurize, fit and score a dataset using the `velari-ai` drivers |
| `evaluation` | Run a target (agent, retriever or model) over cases, score the outputs and aggregate results |

## Install

This package is a member of the `velari-suite` `uv` workspace. From the repository root:

```bash
uv sync --all-extras
```

To use it from another project, install it as an editable path dependency:

```bash
uv pip install -e packages/velari-harness
```

No optional extras are defined yet. Each runtime that needs its own third-party packages will get one, installed as `velari-harness[<runtime>]`.

## Develop

From the repository root:

```bash
uv run pytest tests/harness      # tests
make lint                        # ruff
make typecheck                   # pyright
make -C packages/velari-harness info
```

Tests live under the root `tests/harness/`, not next to the source. Coding conventions are in `.claude/rules/guidelines/python/`.

## Layout

```
packages/velari-harness/
├── pyproject.toml
├── Makefile
└── velari_harness/
    └── __init__.py       # exposes __version__ (from velari-core) and logger
```

The `base/`, `datascience/` and `evaluation/` subpackages are added in the later stages of the plan.
