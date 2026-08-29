# AGENTS.md

AI agent context for the `velari-suite` repository. For the directory structure, GitHub Actions
workflows, and setup instructions, see [README.md](README.md) — this file covers agent-specific
working context instead of duplicating what's already there.

## Project

- **Package**: `velari-suite` — a `uv` workspace monorepo (`velari-core`, `velari-data`, `velari-ai`)
- **Language**: Python 3.12 (managed via `uv`)
- **License**: Apache-2.0

## Purpose

Monorepo for Core Operations, Data, and AI-native development and workflow orchestration, plus a
set of GitHub Actions workflows (shell runners, Python CI with uv, multi-job workflows with
environment variable scoping, JS actions, and Claude AI agent integration) that build and test it.

## Workflows

See README.md's [Workflows](README.md#workflows) section for the full table with links to each
workflow file.

## Key Files

Files an AI agent should read first to understand the project's conventions, configuration, and automation before writing or reviewing code.

| Path | Description |
|---|---|
| `Makefile` | Primary automation: install, python, agents, dotfiles |
| `config/runtime/runtime.env` | Package name, repo URLs, branch config |
| `config/runtime/python.env` | Python version and venv config |
| `pyproject.toml` | Workspace root config: `[tool.uv.workspace]` members, shared ruff/pytest config, dev dependency-group |
| `packages/velari-core/pyproject.toml` | `velari-core` package metadata and its runtime dependencies |
| `packages/velari-data/pyproject.toml` | `velari-data` package metadata; depends on `velari-core` via `[tool.uv.sources] workspace = true` |
| `packages/velari-ai/pyproject.toml` | `velari-ai` package metadata; depends on `velari-core` via `[tool.uv.sources] workspace = true` |
| `.claude/rules/guidelines/python/dev_guides.md` | Coding conventions: logging, type annotations, package structure |
| `.claude/rules/guidelines/python/import_guides.md` | Import conventions: ordering, alignment, grouping, relative vs. absolute, qualification |
| `.claude/rules/guidelines/python/doc_guides.md` | Docstring conventions: Google-style sections, spacing, test-code deviation |
| `.claude/rules/guidelines/python/test_guides.md` | Test-writing conventions: test placement, organization, fixtures, naming |
| `.claude/rules/guidelines/python/agent_guides.md` | Agent working habits: clean up side effects from verification/exploratory commands, including out-of-repo state |
| `.claude/rules/guidelines/python/refactor_guides.md` | Refactoring process/judgment: scoping the diff, consolidating duplication, typing vs. exception handling, verifying before claiming something isn't possible |
| `.claude/rules/guidelines/js/styles/style_frontend.md` | Frontend guidelines: technology stack preferences (Shadcn, NextJS, Tailwind CSS) |
| `.vscode/settings.json` | Editor and code-style configuration (docstring format, formatter settings, etc.) — consult when writing or reviewing code |

## Makefile Targets

| Target | Description |
|---|---|
| `install` | Full environment setup |
| `install_python` | Install Python 3.12 and sync dependencies via uv |
| `install_dotfiles` | Clone/update dotfiles repo |
| `install_agents` | Setup agent infrastructure and skills |
| `format` / `lint` / `typecheck` / `test` | Code quality and testing (`typecheck` runs `pyright`, matching the Pylance settings in `.vscode/settings.json`) |

## Package Structure

This repo is a `uv` workspace monorepo (`[tool.uv.workspace] members = ["packages/*"]` in the root `pyproject.toml`) hosting multiple independently-versioned packages under `packages/`. The root `pyproject.toml` has no `[project]` table of its own — it is a virtual workspace root; each package under `packages/<name>/` has its own `pyproject.toml` with its own name and dependencies. A single `uv.lock` and shared `.venv` cover the whole workspace. See README.md's [Directory Structure](README.md#directory-structure) for the current list of packages and their subpackages.

Each package's import name mirrors its own distribution name (`velari-core` → `velari_core`, `velari-data` → `velari_data`, `velari-ai` → `velari_ai`), following the LangChain/Dagster convention — bare `velari` is intentionally left unclaimed for a possible future top-level orchestration package. Future packages should follow the same convention: independent top-level import name mirroring the distribution name, not a nested `velari.<name>` namespace (which would require PEP 420 cross-distribution namespace-package handling).

## Structure Notes

- `daily-repo-status.lock.yml` is compiled from `daily-repo-status.md` using `gh aw compile` — do not edit directly
- Claude workflows require `ANTHROPIC_API_KEY` secret set in the repository
- `examples/ml/` notebooks that need extra libraries (e.g. `datasetsforecast`) intentionally do **not** use a shared `[dependency-groups]` entry — this workspace's root has no `[project]` name, so `[tool.uv.conflicts]` can't fence off a group's transitive constraints (confirmed empirically: it forced a project-wide `pandas` downgrade). Run those notebooks' dependencies via an isolated, ephemeral resolution instead: `uv run --isolated --with datasetsforecast <command>` — this leaves the shared `.venv`/`uv.lock` (and its `pandas` version) untouched.

For the full directory tree, dotfiles/agent configuration, external build dependencies, and external stores, see README.md's [Directory Structure](README.md#directory-structure) section.
