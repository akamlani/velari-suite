# MLflow Tracing Examples

Demonstrates `velari_ai.integrations.mlflow.connection` (`Connector`/`ConnectorConfig`) — recording
traces through [MLflow](https://mlflow.org/), either against a local file/sqlite store or a remote
tracking server. Deliberately mirrors `examples/ai/integrations/arize/` — same script names, same
`--nested`/`--endpoint` flags, same `get_tracer(...).start_as_current_span(...)` call pattern — both
`Connector` classes implement the shared `velari_ai.ai.tracing.trace.TracingConnector`
interface (`from_config`/`get_tracer`/`.url`/`.config`/`.client`), so the two integrations are used
identically even though the underlying SDKs differ.

- **`serve_cli.py`** — starts a persistent, standalone MLflow server (foreground; Ctrl+C to stop).
- **`trace_cli.py`** — records example trace span(s), either against a local file/sqlite store, or
  by connecting to an already-running MLflow tracking server via `--endpoint`.

Run `serve_cli.py` in one terminal for a persistent, browser-viewable UI, then run
`trace_cli.py --endpoint http://localhost:5001` (in any other terminal, any number of times) to
send spans to it. Without `--endpoint`, `trace_cli.py` logs to its own local sqlite store and exits
— **no UI is ever hosted by `trace_cli.py` itself**, local or remote (see "Architectural difference
from Arize" below); `serve_cli.py` is the only thing that ever serves a browsable UI.

## Prerequisites

MLflow is **not** part of this workspace's `evals` extra — the full `mlflow` package (needed for
`serve_cli.py`'s `mlflow server` command) pins `pandas<3`, which conflicts with this workspace's
`pandas>=3`. Adding it to the shared extras would force a workspace-wide pandas downgrade on every
`uv sync --all-extras`. Instead, run these scripts via `uv run --isolated --with mlflow`, which
resolves `mlflow` into a throwaway environment for that one invocation only — the shared
`.venv`/`uv.lock` is never touched:

```
uv run --isolated --with mlflow python examples/ai/integrations/mlflow/trace_cli.py
```

## `trace_cli.py`

```
uv run --isolated --with mlflow python examples/ai/integrations/mlflow/trace_cli.py
uv run --isolated --with mlflow python examples/ai/integrations/mlflow/trace_cli.py --nested
uv run --isolated --with mlflow python examples/ai/integrations/mlflow/trace_cli.py --endpoint http://localhost:5001
```

- No flags: sets the tracking URI to a local sqlite store (`stores/databases/sqlite/mlflow/`) and
  records a single flat span.
- `--nested`: records a parent-child span pair instead, to see span hierarchy in the UI.
- `--endpoint <url>`: connects to an already-running, externally-managed MLflow tracking server
  instead of a local store. Pair with `serve_cli.py` (see below) for a browser-viewable UI.

## `serve_cli.py`

```
uv run --isolated --with mlflow python examples/ai/integrations/mlflow/serve_cli.py
uv run --isolated --with mlflow python examples/ai/integrations/mlflow/serve_cli.py mlflow.connection.port=5002
```

Runs in the foreground until Ctrl+C — that's expected: this script *is* the persistent server,
unlike `trace_cli.py`, which intentionally never blocks. Built on Hydra's own CLI entry point
(`@hydra.main`), so any dotted config key can be overridden ad hoc on the command line, as in the
second usage line above.

Unlike Arize's `serve_cli.py`, there's no in-process session object or signal-handling wrapper here
— `os.execvp` replaces this process with the real `mlflow server` CLI process directly, so Ctrl+C/
`SIGTERM` are handled natively by MLflow's own server, not by anything in this script.

### Architectural difference from Arize

MLflow has no in-process "start a server" mode (no `ThreadSession` equivalent). Pointing at a local
file/sqlite path versus a remote `http(s)://` tracking server is the *same* call either way
(`mlflow.set_tracking_uri`) — MLflow doesn't distinguish them architecturally, unlike Phoenix's
separate local/remote code paths. Viewing the UI is *always* a separate `mlflow server` process,
whether the backend store is local or remote — so `trace_cli.py`'s local mode never itself hosts a
browsable UI (contrast with Arize's local `ThreadSession`, which does). Concretely: `Connector.url`
in local mode is the tracking-store URI (e.g. `sqlite:///.../mlflow.db`), not a browsable endpoint —
browsing requires separately running `serve_cli.py` pointed at that same store.

MLflow also has no native `get_tracer(name, version)` — span creation is the module-level
`mlflow.start_span()`. `Connector.get_tracer()` returns a small internal `_MlflowTracer` adapter so
the call site still matches Arize's (`get_tracer(...).start_as_current_span(...)`); it stamps
`name`/`version` onto the span as attributes, since MLflow has no instrumentation-scope concept to
hold them otherwise.

`ConnectorConfig.Remote.headers` is accepted (for shape-parity with Arize's `Remote` dataclass) but
not yet wired to anything — only `api_key` is applied (via the `MLFLOW_TRACKING_TOKEN` env var).
MLflow doesn't have as standardized a custom-header mechanism as `api_key`.

## Configuration

- **`config/tracing/mlflow.yaml`** — single source of truth for MLflow's host, port, and storage
  location (`stores/databases/sqlite/mlflow/`, repo-relative). `serve_cli.py` composes this file
  directly as its Hydra root config (`config_path` computed via `read_root_dir()`), so its CLI
  overrides use `mlflow.*` dotted paths (e.g. `mlflow.connection.port=5002`).
  `storage.artifacts_dir` (`stores/artifacts/mlflow/`, also repo-relative) is passed to
  `mlflow server --artifacts-destination`, so logged artifacts land in a deterministic location
  instead of MLflow's CWD-relative `./mlartifacts` default. `security.allowed_hosts`/
  `security.cors_allowed_origins` are empty by default (MLflow's own localhost-only defaults apply
  untouched); set them — e.g. via `mlflow.connection.host=0.0.0.0` plus
  `+mlflow.security.allowed_hosts=[...]` — when exposing the server beyond localhost.
- **`examples/_conf/app_mlflow.yaml`** — `trace_cli.py`'s Hydra config; composes
  `config/tracing/mlflow.yaml` via the `tracing/mlflow@_global_` package directive (so
  `cfg.mlflow.*` is usable directly), plus this example's own `app.info.*` metadata (name/help/
  version). `tracing/mlflow.yaml` itself lives under the root `config/` directory rather than
  `examples/_conf/`, so `trace_cli.py`'s `_load_cfg` passes `config/` as an explicit Hydra
  `hydra.searchpath` override to make it discoverable during composition.
- **`velari_ai/ai/tracing/trace.py`** — the shared `TracingConnector` ABC both this
  integration and Arize's implement, plus the `TracerLike`/`SpanLike` Protocols `get_tracer()`
  returns — this is what keeps the two integrations' call sites identical despite the underlying
  SDK differences described above.
