# Stage 2: Core kernel (`velari_harness/core/`)

Part of [plan.md](plan.md). Requires stage 1. Goal: the harness-agnostic layer (L0) that every category builds on. Nothing in `core/` knows about LLMs, agents, scikit-learn or data science.

## What the kernel does

A [harness](../glossary.md#harness-vocabulary) executes a Task against a Candidate, under an Environment, for N trials; it records every trial and reduces the records to a Result, reproducibly. The terms are defined in the [glossary](../glossary.md); the sources are in the research references of [plan.md](plan.md#research-references).

```
Task (cases + scorers) ─┐
Candidate (system under test) ─┼─> Runner ─> TrialRecord x (cases x trials) ─> Result (table, summary, save/load)
Environment (seed, limits) ─┘
```

## Build order

Stage 2 is built in small increments, each with its own tests, instead of all at once. The [glossary](../glossary.md#component-glossary) describes the finished kernel; this section says what exists today.

| Increment | Adds | Status |
|---|---|---|
| 1 | `Harness` (setup, execute, teardown lifecycle, `from_config`), `HarnessConfig` (`name`), a minimal `Result` (pydantic response schema), `Status` | Implemented |
| 2 | `Candidate` protocol and `CallableCandidate`, with generic input and output | Implemented |
| 3 | `Runner` (sequential, one trial per input), `TrialRecord`, `Status` | Implemented |
| 4 | `Task` (name, inputs, version) | Implemented |
| 5 | `Environment` (base seed, derived trial seeds) | Implemented |
| 6 | `TaskHarness`: a ready-made harness that wires `Harness`, `Runner`, `Task` and `Candidate` | Implemented |
| 7 | Hardening: validated frozen `HarnessConfig`, `teardown` in a `finally`, `TaskResult` with trial counts | Implemented |
| 8 | Scorers as plain functions `fn(output, reference)`; optional `references` on `Task`; `TrialRecord.scores`; `TaskHarness` scores each successful trial; `TaskResult.scores` (mean per scorer) | Implemented |
| 9 | Simplification: `Harness` becomes the container (`run()` chains run, score, package); the extension base, `TaskHarness`, harness hooks and the run-level status are removed; `Runner` only runs; one `Result` | Implemented |
| 10 onward | Not yet planned. The remaining modules are added one increment at a time, each re-planned when the previous one is done | |

### Increment 1: Harness and its config

> **Superseded in increment 9:** the `execute()`/`setup()`/`teardown()` hooks and `from_config` described here no longer exist. `Harness` is now a concrete container; see increment 9.

Files under `velari_harness/core/`:

| File | Contents |
|---|---|
| `schemas/types.py` | `Status`: `SUCCESS`, `ERROR` |
| `execution/config.py` | `HarnessConfig(name)`; `HarnessConfig.from_config(cfg, defaults, **overrides)` with precedence overrides, then `cfg`, then `defaults`; an empty `name` raises `ValueError`; unknown fields raise `TypeError` |
| `schemas/response/result.py` | `Result(name, outcome)`: a pydantic model, so it serialises to and from JSON; `outcome` is an `Outcome` (see increment 3), which rejects a negative duration |
| `execution/harness.py` | `Harness`: `from_config(cfg, **overrides)` (name defaults to the class name), hooks `setup()`, abstract `execute()`, `teardown()`, and `run() -> Result` |

Behavior:
- `run()` calls `setup`, `execute`, then `teardown`. It never raises: a failure in any hook becomes an `ERROR` result with `ExceptionType: message`, the first failure wins, and `teardown` always runs.
- A category harness adds settings by defining a frozen `HarnessConfig` subclass and passing an instance to its constructor (`from_config` builds only a plain `HarnessConfig`). `self.config` is typed as the base `HarnessConfig`, so a subclass narrows it with `isinstance` for now; making `Harness` generic over its config is left until a second harness needs it.
- The core does not seed global RNG state, and a harness has no seed of its own: the one seed belongs to the `Environment` (increment 5). `HarnessConfig` and `Result` do not carry one.
- `Harness` does not yet hold a task, candidate or scorers, and there is no tracking or output directory. Those arrive in later increments.

Tests: `tests/harness/core/execution/test_config.py`, `tests/harness/core/execution/test_harness.py` and `tests/harness/core/schemas/response/test_result.py` (20 tests with the package smoke tests). `pydantic>=2.9` is declared in the package's `pyproject.toml`.

### Increment 2: Candidate

A [Candidate](../glossary.md#component-glossary) is the thing a harness runs and measures: anything that takes an input and returns an output.

| File | Contents |
|---|---|
| `protocols/candidate.py` | `Candidate` (a `typing.Protocol`): `setup()` and `teardown()` that do nothing by default, and an abstract `run(input) -> output`. `CallableCandidate(fn)` wraps any function |

Behavior:
- **Lifecycle belongs to the caller.** The caller runs `setup()`, then `run()`, then `teardown()`, and must call `teardown()` even when `run()` raised. Nothing enforces that yet because there is no runner.
- **`run()` does not catch anything.** An exception propagates to the caller, which decides what it means. Classifying errors is the planned `ErrorPolicy`.
- **A candidate implements only `run()`.** Subclass `Candidate` and the other two hooks are inherited. (The default hooks have an explicit `return None` body so that `ty`, the IDE checker, does not treat them as abstract.) An object with all three methods also satisfies the protocol without subclassing. A subclass without `run()` cannot be instantiated.
- **Generic input and output.** Both are `Any` until `Case` and `Output` exist.
- **`Harness` is unchanged.** It does not hold a candidate. How a harness holds one (composition) next to today's subclass-and-`execute()` extension point is Q1 in the architecture spec, to be decided when the task and runner are planned. Until then a harness subclass can create a candidate itself and call it from its hooks.

Not in this increment: `CaseView`, `TrialContext`, `Sandbox`, adapters for agents, gateways or model drivers (stages 3 and 4).

Tests: `tests/harness/core/protocols/test_candidate.py` (5 test functions, 7 cases: one test is parametrized over three callables). The docstrings show an applied example (a `velari_ai` agent wrapped as a candidate); it is illustrative and not executed.

### Increment 3: Runner

A [Runner](../glossary.md#component-glossary) runs a candidate over inputs and records how each trial went.

| File | Contents |
|---|---|
| `schemas/types.py` | `Status` is used for trials too (one enum for runs and trials); `ReasonCode` (by cause: `INVALID_INPUT`, `NOT_FOUND`, `PERMISSION_DENIED`, `TIMEOUT`, `UNAVAILABLE`, `UNEXPECTED`) in `Outcome.failure` (`Failure(error, reason_code)`), which is `None` on success. The Runner classifies the exception type (`Failure.REASON_RULES`); a candidate can raise `TrialError(reason_code, message)` (`core/errors.py`) to declare its own |
| `schemas/response/record.py` | `Outcome` (pydantic, generic in its `output`): `output`, `failure` and `duration_sec`, with `status` derived (`SUCCESS` when there is no failure); `Failure(error, reason_code)` with the exception-to-reason-code rules; `TrialRecord(index, input, outcome, scores)` |
| `execution/runner.py` | `Runner().run(candidate, inputs) -> List[TrialRecord]` |

Behavior:
- **One trial per input, in order, one after another.** Each trial runs the candidate's `setup`, `run` and `teardown`, so the lifecycle runs once per input.
- **Failures are data.** An `Exception` in a trial becomes an `ERROR` record with `ExceptionType: message`, and the run carries on with the next input. The first failure in a trial wins. A `teardown` failure after a successful `run` makes the record `ERROR` but keeps the output.
- **`teardown` always runs**, including for `KeyboardInterrupt` and `SystemExit`: the Runner uses `try/finally`, so a `BaseException` runs `teardown` and then propagates out of `run()`. Records of earlier trials are lost in that case. (`Harness.run()` does not do this; see D2 in the architecture spec.)
- **The Runner is stateless** and takes no configuration yet. It is a class so that limits and an error policy have somewhere to attach later.
- **The result is a list of records**, not a `Result`: the name `Result` is taken by the run outcome from increment 1 (Q2 in the architecture spec), and aggregation does not exist yet.
- **`Harness` is unchanged** and does not call the Runner. How they connect is open (Q7).

Not in this increment: cases and datasets (inputs are plain values), `Task`, scorers, repeated trials and per-trial seeds, `Environment`, limits and `ErrorPolicy`, artifacts, `score=False`, concurrency, a result store.

Tests: `tests/harness/core/execution/test_runner.py` (8) and `tests/harness/core/schemas/response/test_record.py` (2). The docstrings show an applied example (an agent candidate over support questions); it is illustrative and not executed.

### Increment 4: Task

A [Task](../glossary.md#component-glossary) is what is measured: a named, versioned set of inputs, with no mention of who is measured.

| File | Contents |
|---|---|
| `schemas/request/task.py` | `Task(name, inputs, version="1")`: a frozen dataclass; an empty `name` raises `ValueError`; `inputs` is stored as a tuple |

Behavior:
- **Candidate-agnostic.** A task never names a candidate, so one task is run against several (a baseline and a variant) through the same inputs. This is the point of separating `Task` from `Candidate`.
- **Immutable.** The dataclass is frozen and `inputs` is copied into a tuple, so changing the list you passed in later does not change the task. The immutability is shallow: the input values themselves are not copied.
- **Plain inputs for now.** `inputs` stands in for the dataset of cases. It will be replaced by a dataset when `Case` and `Dataset` exist.
- **Not connected to the Runner.** `Runner.run(candidate, task.inputs)` is how they are used together today; whether the Runner takes a `Task` directly is open (Q8).

Not in this increment: `Case` and `Dataset`, scorers, preparation steps, repeated trials, the missing-score policy, a dataset fingerprint.

Tests: `tests/harness/core/schemas/request/test_task.py` (5). The docstring shows the baseline-versus-variant example (run) and an applied agent-evaluation task (illustrative).

### Increment 5: Environment

An [Environment](../glossary.md#component-glossary) holds the conditions of a run that must not vary between runs. For now that is the base seed. What an environment is, how the layers use it and how it grows are in the [environment note](../architecture/environment.md).

| File | Contents |
|---|---|
| `schemas/request/environment.py` | `Environment(seed=42)`: a frozen dataclass; a `seed` that is not an `int` (or is a `bool`) raises `TypeError`; `seed_for(key, trial=0) -> int` derives a trial seed |

Behavior:
- **Derived, repeatable seeds.** `seed_for` hashes `"{seed}:{key}:{trial}"` with SHA-256 and returns a value in `[0, 2**32)`. The same arguments give the same seed in any process (Python's `hash()` varies between processes, so it is not used); a different base seed, key or trial gives a different seed. The range suits NumPy and scikit-learn.
- **The core applies no seed.** Nothing seeds or reads global random state; `seed_for` only returns a number for a candidate to use.
- **`key` is the input's position today** and becomes the case id once `Case` exists. It is converted with `str`, so `1` and `"1"` give the same seed.
- **Immutable**, like `Task`.
- **Not connected.** Neither the Runner nor the `Harness` reads an `Environment` yet. It is the only place a seed lives: `HarnessConfig` and `Result` no longer carry one.

Not in this increment: the artifacts directory, limits, `ErrorPolicy` and `RetryPolicy`, fail-fast and error-ratio settings, a lazy `Experiment`, `from_config`.

Tests: `tests/harness/core/schemas/request/test_environment.py` (13 cases). The docstring shows the derived seeds (run) and an applied example of seeding a model driver per fold (illustrative).

### Increment 6: TaskHarness

> **Superseded in increment 9:** `TaskHarness` was merged into `Harness`, which now holds the task, candidate and scorers.

A **TaskHarness** wires the pieces: it holds a `Task` and a `Candidate` and runs them through the `Runner`. It is a subclass of `Harness`, so the `Harness` base is unchanged.

| File | Contents |
|---|---|
| `execution/harness.py` (beside `Harness`) | `TaskHarness(config, task, candidate)`; `TaskHarness.for_task(task, candidate, cfg=None, **overrides)`; `execute()` returns `Runner().run(candidate, task.inputs)`; read-only `task` and `candidate` |

Behavior:
- **`run()` returns a `Result` whose `outcome.output` is the list of `TrialRecord`**, one per task input, in order.
- **Status.** `SUCCESS` when the run completes, even if some trials failed. A failing input is in its record as `ERROR`; later inputs still run. (A decision recorded in the architecture spec; an error count on the `Result` is a possible follow-up.)
- **Named after its task** by default (`for_task` defaults `name` to `task.name`); a `name` override wins.
- **Two lifecycles.** The harness's own `setup()` and `teardown()` (inherited no-ops, for subclasses) wrap the whole run; the candidate's `setup`, `run` and `teardown` run once per input inside the Runner.
- **`for_task`, not `from_config`.** The inherited `from_config(cfg, **overrides)` cannot supply a task and a candidate, and adding required arguments to it would break the base signature, so `TaskHarness` has its own constructor method. Calling the inherited `from_config` on it raises `TypeError`.

**Harness or TaskHarness?** `Harness` is the extension base: you subclass it and write `execute()`. `TaskHarness` is the ready-made harness for the usual case, measuring a candidate on a task. The table in the [glossary](../glossary.md#harness-or-taskharness) says which to use. The two names are a known source of confusion; renaming the base to `BaseHarness` was deferred.

Not in this increment: `Environment` (not read yet), scorers (added in increment 8), repeated trials, an error count or summary, concurrency, building a task and candidate from config.

Tests: `tests/harness/core/execution/test_harness.py`, in the `TestTaskHarness` class (7). The docstring shows a task and candidate run for real, and an applied agent example (illustrative).

### Increment 8: Scorers

A trial record said whether the candidate ran, not whether its output was good. A scorer is deliberately just a function, with no new classes or types.

| Change | Where |
|---|---|
| A scorer is a plain function `fn(output, reference)` returning a number (a `bool` counts as 1.0 or 0.0); its `__name__` is the key in `scores`. The argument order matches `velari_ai.ai.evals`, so `exact_match` and `edit_distance` work unchanged. No built-in scorers | `execution/runner.py` |
| `Task(..., references=())`: the expected value per input, stored as a tuple, same length as `inputs` (`ValueError` otherwise); empty means none. A stopgap until `Case` and `Dataset` | `schemas/request/task.py` |
| `TrialRecord.scores: Dict[str, Optional[float]]`, keyed by scorer name; a scorer that raises gives `None`, not a zero | `schemas/response/record.py` |
| `Runner.run(candidate, inputs, scorers=(), references=())` scores each `SUCCESS` trial after its teardown; a failed trial gets no scores; duplicate scorer names, scorers without references and a length mismatch raise `ValueError` | `execution/runner.py` |
| `TaskHarness.for_task(task, candidate, cfg=None, scorers=())`; scorers on a task without references raise `ValueError` at construction | `execution/harness.py` |
| `TaskResult.scores`: the mean per scorer over its non-`None` values (also in the JSON) | `schemas/response/result.py` |

Tests: new cases in `test_task.py`, `test_runner.py`, `test_harness.py` and `test_result.py`. The example scores a CSV of customers against an `expected_email` column.

Not in this increment: a `Scorer` protocol and a `Score` object (a `passed` threshold, label, `valid` flag and `error` text), metrics and reducers beyond the mean, `Case` and `Dataset`, `rescore` and `score=False`, the missing-score policy, repeated trials, judge scorers.

### Increment 7: Hardening

> **Superseded in increment 9:** D2 and the `TaskResult` class are moot: the harness has no hooks, and one `Result` replaced `TaskResult`.

No new core component. It closes the defects D1 to D3 recorded in the architecture spec and two gaps found with `TaskHarness`.

| Change | Where |
|---|---|
| **D1.** `HarnessConfig` checks its name at construction: a non-`str` raises `TypeError`, an empty one `ValueError`, before any hook runs | `execution/config.py` |
| **D2.** `Harness.run()` calls `teardown` in a `finally`, so it also runs for `KeyboardInterrupt` and `SystemExit`, which then propagate (as the Runner already did). Resolves Q4 | `execution/harness.py` |
| **D3.** `HarnessConfig` is a frozen dataclass; a config subclass must also be `@dataclass(frozen=True)`. Resolves Q3 | `execution/config.py` |
| **Error count and JSON round trip.** `TaskResult(Result)` types `outcome` as `Outcome[List[TrialRecord]]` and adds `trials` (a count per `Status`) (also in the JSON), so the records survive a round trip and a run whose every trial failed is visible | `schemas/response/result.py` |
| **One `Outcome` and one `Failure`.** `Result` and `TrialRecord` each hold an `outcome: Outcome` (`output`, `failure`, `duration_sec`, derived `status`), instead of each declaring those fields, so a record cannot have a reason without an error or a status that disagrees. `Outcome` is generic in its `output`, so `TaskResult` types it as `Outcome[List[TrialRecord]]`. `Failure.of(exc)` holds the exception-to-reason-code rules. `Task.references` is `()` when absent (no `Optional`), and the Runner pairs inputs with references through a strict `zip` | `schemas/response/record.py`, `execution/runner.py`, `schemas/request/task.py` |
| **`TaskResult` is built directly.** `TaskHarness.run()` converts the base result with `TaskResult.model_validate(..., from_attributes=True)`; there is no `config_cls` or `result_cls` indirection on `Harness` | `execution/harness.py` |

Tests: 11 new cases across `test_config.py`, `test_harness.py` (classes `TestHarness` and `TestTaskHarness`) and `test_result.py`; the one existing config-subclass test became a frozen dataclass.

A breaking change worth knowing: any `@dataclass` subclass of `HarnessConfig` must now be frozen.

### Increment 9: Harness as the container

The harness and the Runner each wrapped a setup, work, teardown lifecycle, `execute()` existed only so a base class could guarantee that wrapper, and scoring was hidden in the Runner. This increment gives each piece one job.

| Change | Where |
|---|---|
| `Harness` is one concrete class holding a task, a candidate and scorers. `run()` chains three steps: `Runner().run(candidate, inputs)`, `score(records)`, then `Result(name, records, duration_sec)`. `score()` is public, so records can be scored again | `execution/harness.py` |
| `Runner.run(candidate, inputs)` only runs the trials; the scorers and references arguments moved to the harness | `execution/runner.py` |
| One `Result(name, records, duration_sec)` with computed `trials` (a count per status) and `scores`; `TaskResult` and the run-level `outcome` are gone. `Outcome` is no longer generic | `schemas/response/result.py`, `record.py` |
| `Failure(error, reason_code)` replaced the three loose failure fields; the `Step` enum is gone | `schemas/response/record.py`, `schemas/types.py` |
| Removed: the `Harness` ABC, `execute()`, harness-level `setup()`/`teardown()`, `Harness.from_config`, `TaskHarness` | `execution/harness.py` |

Tests: `test_harness.py` is one `TestHarness` class (the scoring cases moved here from `test_runner.py`); `test_result.py` and `test_record.py` follow the new shapes.

A breaking change worth knowing: a run has no `status` or `failure` of its own. Failed trials are data in their records, and anything else (a `KeyboardInterrupt`, a bad argument) raises.

## Modules

One folder per concern, matching how the rest of the repo is organised (`velari_ai/ai/{agents,evals,schemas,...}`). Subfolders have no `__init__.py`; import from the defining module.

```
velari_harness/core/
├── schemas/          # plain data, no behavior
│   ├── types.py          # StrEnums: Status (SUCCESS, ERROR; TIMEOUT planned)
│   ├── request/          # what you give a run
│   │   ├── task.py           # Task: candidate-agnostic spec (increment 4)
│   │   ├── environment.py    # Environment: seed, limits, failure policy, lazy Experiment (increment 5: seed)
│   │   └── case.py           # Case, CaseView, Dataset (planned)
│   └── response/         # what a run produces
│       ├── result.py         # Result: what one run returned (increment 1)
│       ├── record.py         # TrialRecord (increment 3); Event (planned)
│       ├── output.py         # Output: value, artifacts, usage, latency (planned)
│       └── score.py          # Score: value, passed, valid, error (planned)
├── protocols/        # extension points that users and later layers implement
│   ├── candidate.py        # Candidate protocol, TrialContext, CallableCandidate
│   ├── sandbox.py        # Sandbox protocol
│   └── scorer.py         # Scorer protocol
├── results/          # behavior over results; added when it exists (not yet created)
│   ├── metrics.py        # Metric and trial reducers
│   └── store.py          # save()/load()
├── execution/        # running things, and the config that builds them
│   ├── config.py         # HarnessConfig (increment 1)
│   ├── limits.py         # time limit, error budget, ErrorPolicy, RetryPolicy
│   ├── sandbox.py        # LocalSandbox: per-trial working directory
│   ├── runner.py         # Runner: cases x trials -> Result
│   └── harness.py        # Harness: the container (Task + Candidate + scorers) that chains run, score and package
└── registry/         # naming and construction
    ├── registry.py       # generic Registry[T]
    ├── registries.py     # the named instances: tasks, candidates, scorers, harnesses
    └── build.py          # from_config(cfg, **overrides), lazy runtime import
```

### Component definitions

The definition, core example, applied example and status of each component (Case, Candidate, Scorer, Task, Environment, Runner, Harness and the rest) are in the [glossary](../glossary.md#component-glossary). This document covers how they are built: the folders, the build order, the design rules and the tests.

### Sources

Where each idea comes from, and what the detailed read of each added:

- **Inspect AI**: `Task`, `Sample`, the solver and scorer split, the model chosen at run time, limits, a re-readable eval log, per-sample cleanup that runs even after an exception (`Candidate.teardown`).
- **lm-evaluation-harness**: declarative named tasks, a pluggable model interface, `--predict_only` (run without scoring, `score=False`).
- **openai/evals**: the completion-function candidate, a fresh solver per sample, per-sample seeded RNG, append-only events. Its index-based sample ids change when a dataset changes (content-hash ids here), and the base class has no per-sample error guard (the error policy here).
- **HELM**: task x candidate x metric runs, request states that hold everything a metric needs (re-scoring from records). Its README says it is in maintenance mode since June 2026, so it is a design reference only.
- **MLflow GenAI evaluation**: dataset, predict function, scorers, traces, and scorer errors stored as errors rather than scores (`Score.valid`).
- **Harbor**: separate agent, environment, verifier and dataset; the trial lifecycle with teardown always; per-exception error policy (retry, graded, reward valid); `regrade` on recorded artifacts; a verifier that does not see the agent's filesystem (the Sandbox seam and scorers that never touch it).
- **MLE-bench, DSBench**: hidden test labels held by the grader, a prepare step that splits public data, file submissions as output, 3+ seeds reported as mean +/- SEM.
- **AIDE ML**: an agent is a candidate that iterates internally, with a journal and token receipts (events and usage on the record).
- **DSAgentBench**: output files collected after the run and then scored deterministically; the continuous score kept apart from the pass threshold; the limits used are recorded.
- `ModelDriver.from_config` in `velari_ai`: the config build pattern.

### Dependency direction

One way, left to right, so there are no import cycles:

`schemas` <- `protocols` <- `results` <- `execution` <- `registry`

A folder imports only from folders to its left. Imports across folders are relative (`from ..schemas.case import Case`).

- `Harness` lives in `execution/` with its `HarnessConfig`: `Harness.run()` calls the `Runner`, which produces results, so it comes after `schemas`, `protocols` and `results`.
- There is no `definition/` folder. `Environment` sits beside `Task` in `schemas/request/` (both are things you give a run), and `HarnessConfig` sits beside `Harness` in `execution/` (a config belongs to what it configures).
- `Output` and `Score` are data, so they sit in `schemas/`; `protocols/` only describes behavior.
- `protocols/` is its own folder because it is the public contract for stages 3 and 4 and for third parties. Later layers implement `protocols/` and import `schemas/`; they do not reach into `execution/` internals.

## Design rules

- **Candidate is separate from Task.** The same task runs against several candidates. That is how baselines and variants are compared (the `variant_id` in `01-variants.csv`). The Task never imports or names a candidate.
- **Everything observable is recorded per trial**: output, usage, latency, seed, status, error, events. `Result.rescore()` recomputes scores from the records without calling the candidate again, so a new scorer never costs another run.
- **Determinism is explicit.** The environment hands each trial a seed derived from `(env.seed, case id, trial)`. The core never seeds or reads global RNG state.
- **Failures are data.** A trial that raises becomes a record with its status and the error text, classified by the error policy. `fail_fast` stops the run on the first error; `max_error_ratio` stops it when too many trials failed. The `Result.status` says which, so a partial run is never mistaken for a full one.
- **Limits are checked by the runner**, not trusted to the candidate: a wall-clock limit per trial and a total error budget. Token limits arrive with the evaluation category, which knows what usage looks like.
- **Steps prepare cases, they are not the harness.** `Task.steps` is the one place the pipeline idea (Kedro, ZenML) appears: ordered functions that turn raw data into cases. They run once, before any trial.
- **Tracking is lazy and opt-in.** `Experiment()` seeds Python's `random` and NumPy's global RNG with seed 42 and reads `.env`, and `Experiment.create(name, tags)` creates the output directory. The environment builds one only when tracking is enabled, so a default run touches neither.
- **`from_config` mirrors `ModelDriver.from_config`** (`velari_ai/integrations/sklearn/driver.py`): `from_config(cls, cfg: Optional[DictConfig], **overrides) -> Self`, where `None` means defaults and explicit overrides win. It calls `ConfigOmegaConf().resolve()` first. A missing Hydra file yields `None` from `read_hydra`, which the build path turns into a `ValueError`, as `Experiment.create` does.
- **Registry** is a small dict-backed `Registry[T]`. It raises `ValueError` for an unknown name (the same error type as `_AdapterRegistry.resolve` in `velari_ai`) and for a duplicate registration. Existing registries do not fit: `_AdapterRegistry` is private and built around provider/model resolution, and `velari_data.registry.Registry` is a static catalog.
- **The candidate never sees the reference.** The runner passes a `CaseView`; `target` and `grader_files` reach only the scorer. This is what makes hidden-label tasks (MLE-bench, DSBench) possible.
- **Artifacts are collected, then scored.** After `run`, the runner copies the candidate's declared artifacts into the trial directory. Scorers read those copies and never touch the live sandbox, so scoring can happen later, elsewhere, or again.
- **Invalid is not zero.** A score that could not be computed is marked invalid and excluded from metrics; the Task says whether a missing or invalid trial counts as `exclude` or `zero`. A pass threshold is kept apart from the continuous value.
- **Errors are classified, not just caught.** The `ErrorPolicy` decides per exception type whether to retry, whether the trial is scored and whether its score is valid. `Status` distinguishes `OK`, `ERROR` and `TIMEOUT`.
- **A result says what produced it.** The spec holds the resolved config, the task version, the dataset fingerprint, the candidate name and the limits actually applied. Case ids are content hashes unless given.
- **Teardown always runs.** `Candidate.teardown` and the sandbox `stop` run after a failed `run`, so a failed trial cannot leave state behind.
- **Runtimes that need an extra are imported lazily** by the registry, with an error naming the extra (stage 1 convention). The kernel itself needs none.

## Deliberately not in stage 2

The detailed read of the references turned up more than stage 2 should carry. These are deferred, not rejected:

| Finding | Source | Lands in |
|---|---|---|
| Run groups (`Job`: datasets x candidates x attempts) and a `ResultSet` for comparing variants | Harbor, HELM, MLE-bench | stage 2b or stage 3 |
| Incremental, append-only store and resume of an interrupted run | Harbor, `oaievalset` | stage 2b |
| Mergeable statistics and grouped summaries (per split, micro or macro) | HELM, lm-eval, MLE-bench | stage 2b |
| Paired comparison with confidence intervals and tests | MLflow `compare_evaluations` (experimental) | stage 3 |
| Concurrent trials (a thread pool) and async | Harbor (4 trials), openai/evals (10 threads) | stage 3 |
| Trajectory interchange format | Harbor ATIF | stage 3 |
| LLM-judge pinning and validation against human labels | AIDE, DSAgentBench, openai/evals, MLflow | stage 3 |
| Container and cloud sandboxes, GPU and network policy, secret stripping | MLE-bench, Harbor | stage 6 (L3) |
| Observation modality for computer-use agents | DSAgentBench | after stage 3 |

## Style

- Core schemas are pydantic models in the style of the `velari_ai` schemas: aligned fields with `Field(description=...)`. `HarnessConfig` is a dataclass because it is an OmegaConf config. Other dataclasses use `field()` and aligned annotations (`dev_guides.md`). Enums are `StrEnum` with `auto()` and right-aligned `=` columns across the file.
- Plain classes inherit from `object`. `Candidate` and `Scorer` are `typing.Protocol`s (a meaningful base, so exempt); `Registry` is `Generic[T]`.
- `from __future__ import annotations` is the first line of any file whose classmethods return their own class or `Self`.
- Imports follow the aligned-column and relative-import rules in `import_guides.md`. No `TYPE_CHECKING`.
- Google-style docstrings with a realistic `Examples:` that shows the setup flow (build a task, a candidate, run, read the summary). Vary scenarios across modules.

## Before writing code

Read `velari_core/core/experiment.py` (`Experiment`, `ConfigOmegaConf`), `velari_ai/integrations/sklearn/driver.py` (`ModelDriver.from_config`), `velari_core/core/io/partition/hydra.py` (`read_hydra`), and the shape of `examples/ai/evals/data/01-scores.csv`. Settle the open questions in plan.md (sync vs async, dataclasses vs pydantic, `core/` vs `base/`, duplicate registration) before the first file.

## Tests (`tests/harness/core/`)

High-signal only, per `test_guides.md`. Small local fakes, no `unittest.mock`. Test files mirror the source folders: `tests/harness/core/{schemas,protocols,results,execution,registry}/test_<module>.py` (`schemas/request/` and `schemas/response/` have their own subfolders). Pytest runs with `--import-mode=importlib`, so repeated basenames such as `test_registry.py` do not collide.

- The runner scores every case x trial, in order, and each trial gets its own derived seed.
- A candidate that raises produces an error record, not an exception; `fail_fast` stops on the first error and `max_error_ratio` stops the run, with the status saying so.
- One `Task` run against two fake candidates gives two independent results.
- `Result.to_frame()` returns one row per scored trial with the `01-scores.csv` columns that apply; `summary()` reduces trials per scorer.
- `rescore()` with a different scorer gives new scores and the candidate's call count does not change.
- `save()` then `load()` round-trips inside `tmp_path`, and nothing is written elsewhere.
- With tracking off, no `Experiment` is built and no directory is created.
- Registry: a registered name resolves; an unknown name and a duplicate each raise `ValueError`.
- Config build: an OmegaConf config produces the expected harness, and `None` uses defaults.
- A candidate cannot read `target` or `grader_files`: the `CaseView` it receives has neither.
- A file the candidate declares as an artifact is present in the trial directory when the scorer runs.
- An invalid score is left out of `summary()`; with the `zero` policy it counts as 0.
- A `TimeoutError` is recorded as `TIMEOUT` and a retryable error is retried per policy, while a plain error is recorded once as `ERROR`.
- `teardown` runs after a `run` that raises.
- `score=False` yields records without scores; `rescore` fills them in and the candidate's call count does not change.
- Changing one case input changes `Dataset.fingerprint`; the same cases in the same order give the same fingerprint.

## Verification

- Targeted tests, then the full suite, with only pre-existing failures.
- Run a toy harness for real (a fake candidate, two scorers, three trials) and compare the printed summary to the docstring `Examples:`.
- Typecheck and lint clean; `core/` imports nothing from `velari_ai` or scikit-learn (grep).
- Remove any scratch output created during verification.

## Exit criteria

A toy harness defined entirely in a test runs from an OmegaConf config through the registry and returns a `Result` whose table and summary match the documented shape, with trials recorded, re-scorable, and no files written outside `tmp_path`. A fake candidate cannot see hidden labels, produces an artifact that a scorer grades, and a failed trial is classified and does not leave state behind.
