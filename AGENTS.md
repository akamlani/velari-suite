# AGENTS.md

AI agent context for the `project-template-velari` repository.

## Project

- **Package**: `velari-github-workflows`
- **Language**: Python 3.12 (managed via `uv`)
- **License**: Apache-2.0

## Purpose

Tutorial repository demonstrating GitHub Actions workflows — covering shell runners, Python CI with uv, multi-job workflows with environment variable scoping, JS actions, and Claude AI agent integration.

## Workflows

| Workflow | Trigger | Description |
|---|---|---|
| `basic-workflow.yml` | Manual | Multi-job workflow with JS action, env variable scoping |
| `python-workflow.yml` | Push/PR/Manual | Python CI with uv package manager and caching |
| `shell-workflow.yml` | Manual | First workflow — self-hosted and Ubuntu runners |
| `claude.yml` | Issue/PR comments, Issues | Claude AI agent integration via `@claude` mentions |
| `claude-code-review.yml` | Pull Request | Automated code review on PRs using Claude |
| `daily-repo-status.lock.yml` | Schedule (daily) | Daily repo status report generated as a GitHub issue |

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
| `.claude/rules/dev_preferences.md` | Coding conventions: logging, type annotations, package structure |
| `.claude/rules/test_preferences.md` | Test-writing conventions: test placement, organization, fixtures, naming |
| `.claude/rules/agent_preferences.md` | Agent working habits: clean up side effects from verification/exploratory commands, including out-of-repo state |
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

This repo is a `uv` workspace monorepo (`[tool.uv.workspace] members = ["packages/*"]` in the root `pyproject.toml`) hosting multiple independently-versioned packages under `packages/`. The root `pyproject.toml` has no `[project]` table of its own — it is a virtual workspace root; each package under `packages/<name>/` has its own `pyproject.toml` with its own name and dependencies. A single `uv.lock` and shared `.venv` cover the whole workspace.

| Package | Import path | Description |
|---|---|---|
| `packages/velari-core/` | `velari_core` | Core utilities: I/O, filesystem, partitioning, and experiment management |
| `packages/velari-core/` | `velari_core.integrations.pydantic` | Third-party integrations: Pydantic tooling (FastAPI integration planned) |
| `packages/velari-data/` | `velari_data` | Data utilities; depends on `velari-core` |

Each package's import name mirrors its own distribution name (`velari-core` → `velari_core`, `velari-data` → `velari_data`), following the LangChain/Dagster convention — bare `velari` is intentionally left unclaimed for a possible future top-level orchestration package. Future packages (e.g. `velari-ai`) should follow the same convention: independent top-level import name mirroring the distribution name, not a nested `velari.<name>` namespace (which would require PEP 420 cross-distribution namespace-package handling).

## Structure Notes

- `daily-repo-status.lock.yml` is compiled from `daily-repo-status.md` using `gh aw compile` — do not edit directly
- Claude workflows require `ANTHROPIC_API_KEY` secret set in the repository

### Project Directories

| Directory | Description |
|---|---|
| `.env` | Local environment variables and secrets; not tracked by git |
| `data/` | Sample and reference data files for examples and experiments |
| `docs/` | Documentation assets and prompt references |
| `examples/` | Runnable example scripts and application prototypes. `examples/ml/` notebooks that need extra libraries (e.g. `datasetsforecast`) intentionally do **not** use a shared `[dependency-groups]` entry — this workspace's root has no `[project]` name, so `[tool.uv.conflicts]` can't fence off a group's transitive constraints (confirmed empirically: it forced a project-wide `pandas` downgrade). Run those notebooks' dependencies via an isolated, ephemeral resolution instead: `uv run --isolated --with datasetsforecast <command>` — this leaves the shared `.venv`/`uv.lock` (and its `pandas` version) untouched. |
| `experiments/` | Experimental scripts and output snapshots |
| `logs/` | Runtime log output; not tracked by git |
| `outputs/` | Generated output artifacts, organized by date; not tracked by git |
| `templates/` | Reusable project templates including spec scaffolds |
| `tests/` | pytest test suite for the workspace, centralized regardless of package count (`testpaths = ["tests"]`) |
| `packages/` | `uv` workspace member packages, one subdirectory per installable package |

### Dotfiles & Agent Configuration

Configuration directories managed by `make install` and the dotfiles system. Some contents are symlinked from `_build/dotfiles/`; others are generated by Make targets.

| Directory | Description | Managed by |
|---|---|---|
| `.agents/` | Agent runtime directory | `make setup_agent` |
| `.claude/` | Claude Code configuration, plugin settings, and rules (e.g. `rules/dev_preferences.md`, `rules/test_preferences.md`, `rules/agent_preferences.md`) | `make setup_agent_claude` |
| `.github/` | GitHub Actions workflows (tracked source) and Copilot configuration (`copilot-instructions.md` symlinked from dotfiles) | Workflows tracked; config symlinked via `make link_dotfiles` |
| `.velari/` | Velari runtime configuration and cache directory | `make setup_agent` |
| `.vscode/` | VS Code editor and code-style configuration | Symlinked from `_build/dotfiles/` via `make link_dotfiles` |

### External Build Dependencies (`_build/`)

External repositories cloned into `_build/` by `make install`. Not tracked by git.

| Directory | Description |
|---|---|
| `_build/dotfiles/` | Shared dotfiles repo; provides `.vscode/` and `.github/copilot-instructions.md` as symlinks into the project root |
| `_build/agent-skills/` | AI agent skills and commands; skill toolkit lives at `_build/agent-skills/toolkit/` |

### External Stores (`stores/`)

Symlinked directories pointing to a shared external vault. Not tracked by git. Created by `make install` via `link_vaultspace`.

| Directory | Description |
|---|---|
| `stores/artifactlib/` | Artifact storage for equivalent projects and experiments |
| `stores/promptlib/` | Library of reusable prompts maintained outside this repository |
| `stores/contextlib/` | Common context library maintained outside this repository |

#### `stores/contextlib/`

| Directory | Description |
|---|---|
| `_personas/` | Persona definitions for AI roles and characters |
| `_rules/` | Rule sets governing analysis, guidelines, policies, styles, and tools |
| `_specs/` | Specification documents maintained outside the repository |
| `assets/` | Shared assets and media |
| `bio/` | Background and biographical context |
| `data/` | Data references and contextual data |
| `glossary/` | Terminology and concept definitions |
| `research/` | Research notes and references |
| `strategy/` | Strategic context and direction |
| `tech/` | Technology references and notes |
| `workspace/` | Workspace-level context and working notes |

#### `stores/contextlib/_rules/`

| Directory | Description |
|---|---|
| `datasets/` | Rules and guidelines for dataset handling |
| `policies/` | Operational and AI policy definitions |
| `styles/` | Style guide rules and conventions |
