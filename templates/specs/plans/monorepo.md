# Python Monorepo Workspace: Migration & Package-Addition Template

Reusable template for two related situations:
1. **Splitting a flat Python package into a multi-package workspace monorepo** (one distribution → several independently-installable distributions sharing a workspace).
2. **Adding a new package to an already-existing workspace monorepo.**

Written to be tool-agnostic where the underlying concept is (namespace strategy, root detection, test placement); commands shown use `uv` as the reference implementation since that's the most common modern choice, with the Poetry/PDM/plain-setuptools equivalents noted where they differ meaningfully. The packaging mechanics referenced (PEP 420 namespace packages, PEP 660 editable installs) are Python-specific, so "any monorepo design" here means *any Python monorepo tooling*, not language-agnostic advice.

## Key decisions to make up front

Make these decisions explicitly before moving any code — several are expensive to reverse once other packages depend on the choice.

1. **Workspace directory convention.** `packages/` (uv community convention) vs. `libs/` (LangChain's convention) vs. something else. Pick one and use it consistently for every member.

2. **Namespace strategy, per package.** Default recommendation: give each package an **independent top-level import name that mirrors its own distribution name** (`my-core` → `my_core`, `my-data` → `my_data`), following the convention used by LangChain (`langchain-core` → `langchain_core`) and Dagster (`dagster-aws` → `dagster_aws`). Reserve the *bare* top-level name (no suffix) for a possible future orchestration/meta package that sits above the others, the way `langchain` itself sits above `langchain-core`.
   The alternative — every package sharing one dotted namespace (`myapp.core`, `myapp.data`) — requires genuine **PEP 420 namespace packages**, which only works if it's designed *uniformly across every distribution from the start* (no distribution ships a plain `__init__.py` for the shared root). It is **not** something you can retrofit later once one distribution already owns a real `__init__.py` for that namespace — trying to bolt it on after the fact is exactly the friction the independent-name convention avoids. Real examples of the from-day-one namespace-package approach: OpenTelemetry Python (`opentelemetry.*`), Apache Airflow providers (`airflow.providers.*`).

3. **Root-path / repo-root detection.** Any code that assumes "the repo root is N directories above this file" will break the moment that file moves deeper into `packages/<name>/`. Replace hardcoded depth assumptions with logic that walks up the directory tree looking for an authoritative marker (e.g. a workspace-declaration table in a manifest file) rather than matching on file existence/name alone — a per-package manifest file can otherwise be mistaken for the workspace root.

4. **Editable-install static analysis.** Modern editable installs (PEP 660 "new-style," used by `uv`/setuptools by default) register a `MetaPathFinder`-based import hook that remaps the import name to its real source location *by executing Python code* at import time. Static analyzers (Pyright/Pylance and similar) don't execute that hook — they only read plain directory-based search paths — so they'll show phantom "unresolved import" errors for a package that works fine at runtime and under pytest. Fix by adding each package's source directory to the type-checker's own extra-source-roots config (e.g. `[tool.pyright] extraPaths` in `pyproject.toml`). This setting typically does **not** support globs, so it needs one explicit entry per workspace member, updated every time a package is added.

5. **Test placement.** Centralized `tests/<package-shortname>/` at the workspace root (simpler, one test-runner config) vs. per-package `packages/<name>/tests/` (better isolation if packages will ever be split into separate repos later). Pick one and apply it to every package.

6. **Cross-package dependencies.** Decide how a package depends on a sibling package in the same workspace *before* writing code that would otherwise duplicate shared utilities. With `uv`, declare the dependency normally in `[project.dependencies]` and point it at the local copy via `[tool.uv.sources] <dep-name> = { workspace = true }`. Poetry and PDM have their own equivalent path/workspace-dependency mechanisms.

## Generic steps: flat package → workspace (first split)

1. Scaffold `<workspace_root>/<packages_dir>/<package-dist-name>/` with its own manifest: distribution name, version, and only the dependency subset the code being moved actually imports. Add a package-local README if the manifest's readme field requires the file to exist for metadata to build.
2. Move the code with history preserved (`git mv <old_location> <packages_dir>/<package-dist-name>/<package_import_name>/`), deleting the now-empty original location.
3. Remove any re-export shim `__init__.py` that exists only to shorten import paths, and flatten any subpackage that stutters the distribution's own name (see the "redundant nesting" gotcha below) — update the handful of call sites that imported through the shim.
4. Fix root/path-detection code (decision 3) to walk up dynamically instead of assuming a fixed depth.
5. Convert the workspace root's manifest into a virtual/workspace-only root — no `[project]` table of its own, just the workspace-members declaration (e.g. `uv`'s `[tool.uv.workspace] members = [...]`) plus any repo-wide shared config (linting, test runner) that should apply regardless of member count.
6. Update any install/build tooling (Makefile targets, CI scripts) that assumed a single installable root package — drop root-level install steps that no longer apply, and remove any single-package-name rewriting hacks.
7. Add the static-analysis fix from decision 4.
8. Update repo-level docs describing the layout (package tables, directory trees) to reflect the new workspace structure.

Tests can generally stay wherever they already are (per decision 5) — only their imports need to change if the package's import path changed.

## Generic steps: adding a new package to an existing workspace

1. Create `<packages_dir>/<new-package-dist-name>/` with its own manifest — distribution name, dependencies, and package-discovery config scoped to `<new_package_import_name>*` only.
2. Decide the namespace strategy (decision 2) explicitly for this package before writing any code — default to an independent top-level name mirroring its distribution name.
3. If it depends on another workspace member, wire that up via your tool's intra-workspace dependency mechanism (decision 6) instead of duplicating shared utilities.
4. Usually no change needed to the workspace-members glob (e.g. `members = ["packages/*"]` auto-discovers any new `packages/<name>/`).
5. Add tests for the new package following the workspace's chosen test-placement convention (decision 5).
6. Append the new package's source path to the static-analysis `extraPaths`-equivalent config (decision 4) — remember it won't be picked up by a glob.
7. Update repo-level docs (package table, directory tree) to describe the new member.

## Gotchas checklist

- **Root-detection hardcoded on directory depth** silently breaks the moment a file moves one level deeper into a workspace subdirectory. Always resolve the root dynamically.
- **A subpackage that repeats its own distribution's name** (`mypackage/core/thing.py` inside a distribution already named `my-core`) is a smell once the distribution's whole purpose *is* "core" — flatten it (`mypackage/thing.py`) rather than keeping the stutter.
- **PEP 660 editable installs break static analyzers silently.** Imports work at runtime and under pytest; the editor shows a phantom "unresolved import" error. Fix it in the type-checker's config (extra source roots), not by changing the install mode.
- **Don't edit shared/symlinked editor config** (e.g. a dotfiles-managed `.vscode/settings.json`) to add project-specific paths — put that configuration in a project-tracked file instead (e.g. a `[tool.pyright]` table in the repo's own `pyproject.toml`).
- **A shared dotted namespace across distributions is a from-day-one design commitment**, not a later patch. If even one distribution already owns a plain `__init__.py` for the shared root, default every sibling to an independent top-level name instead of retrofitting PEP 420 namespace packages.

## Worked example (`velari-suite`)

This template was extracted from a real migration in this repo: `velari-suite` was split into a `uv` workspace under `packages/`, starting with `velari-core` (later renamed from a bare `velari` import name to `velari_core`, flattening a redundant nested `core/` subpackage in the process, once a second package — `velari-data`, importing as the independent `velari_data` and depending on `velari-core` via `[tool.uv.sources] { workspace = true }` — made the naming inconsistency visible). Root-path detection was fixed to walk up parsing for `[tool.uv.workspace]` rather than matching on any `pyproject.toml`. Pylance's editable-install blind spot was fixed via `[tool.pyright] extraPaths` in the root `pyproject.toml`, updated each time a package was added. The internal layout was refined twice more afterward: infrastructure-style subpackages (I/O, perf, statistics, transforms, utils) were first regrouped under a `velari_core/core/` subdirectory while `experiment.py`/`types.py` stayed flat at the package root, then those two were moved into `core/` as well — leaving only `__init__.py`, `version.py`, and the `integrations/` sibling at `velari_core/`'s own root. Internal layout preferences like this are cheap to iterate on precisely because moving a whole subtree together preserves every relative import within it; only the few imports crossing the moved boundary need updating each time.
