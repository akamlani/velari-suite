# Arize Phoenix Tracing Examples

Demonstrates `velari_ai.integrations.arize.connection` (`Connector`/`ConnectorConfig`) — recording
OpenTelemetry spans through [Arize Phoenix](https://github.com/Arize-ai/phoenix), either against a
Phoenix instance started in-process or one running standalone.

- **`serve_cli.py`** — starts a persistent, standalone Phoenix server (foreground; Ctrl+C to stop).
- **`trace_cli.py`** — records example trace span(s), either by starting its own local, in-process
  Phoenix (`ThreadSession`), or by connecting to an already-running Phoenix via `--endpoint`.

Run `serve_cli.py` in one terminal for a persistent, browser-viewable UI, then run
`trace_cli.py --endpoint http://localhost:6006` (in any other terminal, any number of times) to
send spans to it — this is the production-style pattern, decoupling the UI's lifetime from any
single script run. Without `--endpoint`, `trace_cli.py` starts and tears down its own local Phoenix
each run.

## Prerequisites

```
uv sync --extra evals
```

Installs `arize-phoenix`, `arize-phoenix-otel`, `arize-phoenix-client`, and `opentelemetry-api`
(defined under `evals` in `packages/velari-ai/pyproject.toml`).

## `trace_cli.py`

```
uv run --extra evals python examples/ai/integrations/arize/trace_cli.py
uv run --extra evals python examples/ai/integrations/arize/trace_cli.py --nested
uv run --extra evals python examples/ai/integrations/arize/trace_cli.py --endpoint http://localhost:6006
```

- No flags: starts a local, in-process Phoenix (`ThreadSession`) and records a single flat span.
- `--nested`: records a parent-child span pair instead, to see span hierarchy in the UI.
- `--endpoint <url>`: skips the local Phoenix entirely and connects to an already-running instance
  (the production pattern) — fires its span(s) and exits immediately; the UI's lifetime is
  independent of this script, since it was never hosted by this process to begin with. Pair with
  `serve_cli.py` (see below) for a browser-viewable UI without Docker.

## `serve_cli.py`

```
uv run --extra evals python examples/ai/integrations/arize/serve_cli.py
uv run --extra evals python examples/ai/integrations/arize/serve_cli.py phoenix.connection.port=6007
```

Runs in the foreground until Ctrl+C (or `SIGTERM`) — that's expected: this script *is* the
persistent server, unlike `trace_cli.py`, which intentionally never blocks. Built on Hydra's own
CLI entry point (`@hydra.main`), so any dotted config key can be overridden ad hoc on the command
line, as in the second usage line above.

Uses `phoenix.launch_app()` with `run_in_thread=True` (`ThreadSession`, a daemon thread) — the
script's own blocking `stop_event.wait()` is what keeps the process (and Phoenix) alive, since a
daemon thread alone wouldn't outlive the main thread.

### Known issue

`launch_app(..., run_in_thread=False)` (`ProcessSession`) is broken in this environment: Phoenix's
own `AppService` subprocess runs `main.py` with `cwd` set to `.../phoenix/server/`, whose local
`email/` subfolder shadows the stdlib `email` package, crashing with
`ModuleNotFoundError: No module named 'email.message'`. This is inside the installed
`arize-phoenix` package, not something fixable here — `run_in_thread=True` is used instead.

## Configuration

- **`config/tracing/phoenix.yaml`** — single source of truth for Phoenix's host, port, and storage
  location (`stores/databases/sqlite/phoenix/`, repo-relative). `serve_cli.py` composes this file
  directly as its Hydra root config (`config_path` computed via `read_root_dir()`), so its CLI
  overrides use `phoenix.*` dotted paths (e.g. `phoenix.connection.port=6007`).
- **`examples/_conf/app_arize.yaml`** — `trace_cli.py`'s Hydra config; composes
  `config/tracing/phoenix.yaml` via the `tracing/phoenix@_global_` package directive (so
  `cfg.phoenix.*` is usable directly, no `cfg.tracing.phoenix` indirection), plus this example's own
  `app.info.*` metadata (name/help/version, used to populate the Typer app and `get_tracer()`).
  `tracing/phoenix.yaml` itself lives under the root `config/` directory rather than
  `examples/_conf/`, so `trace_cli.py`'s `_load_cfg` passes `config/` as an explicit Hydra
  `hydra.searchpath` override to make it discoverable during composition.
