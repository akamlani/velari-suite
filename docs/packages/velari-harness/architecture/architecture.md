# velari-harness architecture

Matches `velari_harness` as of stage 2, increment 13. Update it when an increment changes behavior. It describes the code that exists; planned components are in [../plan/](../plan/plan.md) and marked **Planned** in the [glossary](../glossary.md#component-glossary). Statements marked *verified* were checked by running the code; anything else is read from the source.

Audience: contributors who extend or review the harness core. Definitions of terms live in the glossary and are not repeated here.

## 1. Scope

Today the package provides these things. A `Harness`: the container for one measurement, which holds a task, a candidate and optional scorers and whose `run()` chains three steps (run the candidate on every example, score each successful trial, package a serialisable `Result`). The `Candidate` contract: the thing being measured, with a function wrapper. A `Runner` that runs a candidate over a sequence of examples and returns one `TrialRecord` per example. A `Example`: one input for the candidate and, optionally, the reference a scorer compares with. A `Task`: a named, versioned, immutable set of examples that says nothing about who is measured. Scorers as plain functions. An `Environment`: the base seed, with a repeatable seed derived for each trial; the Runner derives each trial's seed from it and records the seed on the trial.

Not implemented: examples, datasets, metrics and reducers, repeated trials, a registry, a result store, tracking, an output directory, retries, limits. The core does not seed global random state.

## 2. Components and dependencies

| Module | Responsibility |
|---|---|
| `__init__.py` | `__version__` (from installed metadata via `velari-core`) and the package `logger` |
| `core/schemas/types.py` | `Status` (`SUCCESS`, `ERROR`): how a run or one trial ended; `ReasonCode` (`INVALID_INPUT`, `NOT_FOUND`, `PERMISSION_DENIED`, `TIMEOUT`, `UNAVAILABLE`, `UNEXPECTED`): why a trial failed, by cause. `core/errors.py` has `TrialError`, which a candidate raises to declare its own code |
| `core/schemas/response/record.py` | `Output` (`value`, `usage`, `artifacts`): what a candidate may return; `Outcome` (`output`, an `Output`, `failure`, `duration_sec`, derived `status`) and `Failure(error, reason_code)` with the exception-to-reason-code rules; `Score` (`value`, `passed`, `label`, `explanation`, `failure`, `metadata`): a scorer's verdict; `TrialRecord(index, example, outcome, seed, scores)`: what one trial produced, with the seed derived for it |
| `core/schemas/response/result.py` | `Result(name, records, duration_sec)`: one record per task input in order, with computed `trials` (a count per status), `usage` and `summary` (per scorer a `ScoreSummary`: mean, standard error, valid count `n` and `invalid` count, computed with `velari_core`'s `calc_column_stats`) |
| `core/execution/config.py` | `HarnessConfig` (`name`) and building it from OmegaConf (frozen dataclass, checked at construction) |
| `core/schemas/dataset.py` | `Example` (`input`, `reference`, `uid`): one thing to measure; `Dataset(name, examples)`: an ordered, immutable collection with uids and a `fingerprint`; `Task(name, dataset, version)`: what is measured (all frozen dataclasses) |
| `core/schemas/request/environment.py` | `Environment`: the base seed and repeatable per-trial seeds (frozen dataclass) |
| `core/protocols/candidate.py` | `Candidate` (protocol with a per-trial lifecycle) and `CallableCandidate` |
| `core/execution/harness.py` | `Harness`: holds a task, a candidate, scorers and an `Environment`; `run()` chains run, score and package; `score()` scores records; `for_task()` builds one |
| `core/execution/runner.py` | `Runner(environment)`: runs a candidate over examples, one trial each, and records the outcome and the trial's seed; it does not score |

Field-level detail is in the source docstrings and the glossary, not here. Subfolders have no `__init__.py`.

```mermaid
graph LR
    harness["execution/harness.py"] --> config["execution/config.py"]
    harness --> result["schemas/response/result.py"]
    result --> types
    result --> corestats["velari_core: calc_column_stats"]
    result --> pydantic["pydantic"]
    config --> omegaconf["omegaconf"]
    config --> core["velari_core: ConfigOmegaConf"]
    harness --> omegaconf
    init["__init__.py"] --> core2["velari_core: package_version"]
    candidate["protocols/candidate.py"]
    dataset["schemas/dataset.py"]
    environment["schemas/request/environment.py"]
    record["schemas/response/record.py"] --> types
    record --> dataset
    record --> pydantic
    runner["execution/runner.py"] --> candidate
    runner --> record
    runner --> types
    harness --> runner
    harness --> dataset
    harness --> candidate
    harness --> record
```

`protocols/candidate.py` imports only the standard library; only `runner.py` imports it. `harness.py` imports `runner.py`, `Task` (from `dataset.py`), `Candidate`, `TrialRecord` and `Result`; the Runner does not import `harness.py`. `schemas/dataset.py` imports only the standard library; `record.py`, `runner.py` and `harness.py` import it. `schemas/request/environment.py` imports only the standard library and nothing imports it yet. `schemas/response/result.py` imports `calc_column_stats` from `velari_core` for the summary statistics, as `config.py` imports `ConfigOmegaConf`.

`velari-ai` and `pandas` are declared in `pyproject.toml` but not imported (*verified* by grep).

Folder convention: within `core/`, a folder imports only from folders to its left in `schemas` <- `protocols` <- `execution`. Nothing enforces it (no lint rule or test); it holds because the package is small. Imports inside the package are relative; imports of other packages are absolute.

## 3. Behavior

```mermaid
flowchart TD
    A["Harness.run()"] --> B["start timer"]
    B --> C["1. Runner.run: one TrialRecord per input"]
    C --> D["2. score(): each successful trial gets scores by scorer name"]
    D --> E["3. Result(name, records, duration_sec)"]
```

A task run has no run-level status or failure. A trial that raises (in `setup`, `run` or `teardown`) is recorded as an `ERROR` outcome and the run carries on, and a scorer that raises gives `None`. Only a `BaseException` (`KeyboardInterrupt`, `SystemExit`) or a bad argument stops a run: the candidate's `teardown` still runs, then the exception propagates and no `Result` is returned. `duration_sec` is measured with `time.perf_counter()` around the three steps; there are no retries, limits or timeouts.

### What holds, and how it is enforced

| Guarantee | Enforced by | Limit |
|---|---|---|
| A trial's `Exception` becomes an `ERROR` outcome and the run carries on | `try/except Exception` in `Runner._trial`; tests | A `BaseException` stops the run and the earlier records are lost |
| The first failure in a trial is the one reported | code; test | |
| `HarnessConfig.name` is a non-empty string, and the config cannot change | `__post_init__` (`TypeError`, `ValueError`); a frozen dataclass; tests | Other fields a subclass adds are not type-checked |
| `Outcome.duration_sec >= 0` and `Result.duration_sec >= 0` | pydantic `Field(ge=0)` | At construction only; pydantic does not validate on assignment |
| `Result` round-trips through JSON | test | Only for JSON-serialisable outputs and inputs |
| The core never seeds global RNG | absence in the code; a test for `Environment.seed_for` | No test for the other modules |
| A trial's recorded seed is `Environment.seed_for(example.uid)` (the index for an example without a uid), the same on every run | `Runner._trial`; tests | The candidate is not told it |
| A trial's seed is the same on every run, in any process | SHA-256 derivation; a pinned-value test | Nothing passes the seed to a candidate yet |
| A candidate's `teardown` runs after a failed `setup` or `run`, including `BaseException` | `try/finally` in `Runner._trial`; tests | |
| A failed trial is never scored, and a scorer that raises is `None`, never a zero | `Harness.score`; tests | |

### Task

`Task(name, dataset, version="1")` is a frozen dataclass; an empty `name` raises `ValueError` and assigning to a field raises `FrozenInstanceError`. The `Dataset(name, examples)` it holds stores the examples as a tuple (later changes to the list that was passed in do not reach it), gives an example with no `uid` a hash of its content, raises on an empty name, no examples or duplicate uids, and has a `fingerprint` that changes with any uid, input, reference or the order. All covered by tests in `test_dataset.py`, together with the central property: one task, run through the Runner against two different candidates, gives each its own outputs. The task knows nothing about either.

### Scorer

A scorer is a plain function `fn(output, reference)` returning a number (a `bool` counts as 1.0 or 0.0) or a `Score`, named by its `__name__`; the functions in `velari_ai.ai.evals` (`exact_match`, `edit_distance`) work as they are. `Harness.score(records)` applies every scorer to each `SUCCESS` record, comparing the output's value with `record.example.reference`, wraps a plain return in a `Score`, and returns copies with `scores` filled in (a `Dict[str, Score]` by scorer name). A failed trial has no scores, because it has no output to judge. A scorer that raises gets a `Score` with a `failure` (a `Failure`, so it has a reason code like a failed trial) and no value, never a zero, and later trials still run. Scorers when an example has no reference, and duplicate scorer names, raise `ValueError` when the harness is built. The candidate never receives a reference. `Result.summary` has, per scorer, the mean of each `Score.value` that is not `None`, its standard error, and the counts `n` (valid) and `invalid`. Covered by the scoring examples in `test_harness.py` and `test_result.py`.

### Environment

`Environment(seed=42)` is a frozen dataclass; a `seed` that is not an `int`, or is a `bool`, raises `TypeError`. `seed_for(key, trial=0)` returns `sha256("{seed}:{key}:{trial}")` reduced to `[0, 2**32)`. Covered by tests in `test_environment.py`: pinned values (so the derivation cannot change silently), identical results for identical arguments, a different result when the seed, key or trial changes, the 32-bit range, and global random state left untouched. The values are the same under a different `PYTHONHASHSEED` (*verified*), because SHA-256 is used rather than `hash()`. `1` and `"1"` give the same seed (*verified*), since the key goes through `str`. Since increment 14 the Harness holds an `Environment` (default seed 42) and passes it to the Runner, which calls `seed_for(example.uid)` for each trial (the example's index when it has no uid) and stores the result as `TrialRecord.seed`; the candidate is not told the seed.

### Harness

`Harness.for_task(task, candidate, scorers=[...])` builds a harness named after the task (a `name` override wins). `run()` reads top to bottom as three steps: `Runner().run(candidate, task.examples)`, `self.score(records)`, then `Result(name, records, duration_sec)`. `Result.trials` counts the trials per status, so a run where every trial failed is visible. There is no harness-level `setup` or `teardown`: the candidate's lifecycle runs once per input inside the Runner. `score()` is public so records can be scored again. Covered by `test_harness.py`.

### Runner

`Runner().run(candidate, examples)` returns one `TrialRecord` per example, in order; it only runs the trials and never scores them. For each example it runs `candidate.setup()`, `candidate.run(example.input)` and `candidate.teardown()` (the lifecycle runs once per input). A candidate may return an `Output` (a value plus `usage` and `artifacts`); a plain return is wrapped in one, so a record's `outcome.output` is an `Output`, or `None` for a failed trial. Outcomes, each covered by a test in `test_runner.py`:

| Scenario | Candidate calls | Record `outcome.status` | `outcome.output.value` | `outcome.failure.error` |
|---|---|---|---|---|
| Everything succeeds | `setup`, `run`, `teardown` | `SUCCESS` | what `run` returned | `None` |
| `run` raises on one input | all three; later inputs still run | `ERROR` | `None` | `ValueError: ...` |
| `setup` raises | `setup`, `teardown` (no `run`) | `ERROR` | `None` | `RuntimeError: setup failed` |
| `teardown` raises after success | all three | `ERROR` | what `run` returned | `OSError: teardown failed` |
| `teardown` raises after `run` raised | all three | `ERROR` | `None` | the `run` error (first failure wins) |
| `KeyboardInterrupt` in `run` | `setup`, `run`, `teardown`, then it propagates | no records returned | | |
| No inputs | none | empty list | | |

One `ERROR` log line is written per failed trial. There are no limits, timeouts or retries, and trials are sequential.

## 4. Configuration flow

```
Harness.for_task(task, candidate, cfg, **overrides)
  -> HarnessConfig.from_config(cfg, defaults={"name": task.name}, **overrides)
       ConfigOmegaConf().resolve()                  registers the as_tuple resolver (process-wide, idempotent)
       OmegaConf.to_container(cfg, resolve=True)    {} when cfg is None
       HarnessConfig(**{**defaults, **cfg_fields, **overrides})
  -> cls(config, task, candidate, scorers)
```

- **Precedence:** explicit overrides, then `cfg`, then defaults (the task name for `name`).
- **What `cfg` is:** exactly the harness's own section, holding only `HarnessConfig` fields (just `name`). A root config with sibling sections fails (*verified*): `TypeError: HarnessConfig.__init__() got an unexpected keyword argument 'harness'`. Callers must select the section first.
- **Errors:** non-mapping `cfg` raises `TypeError`; an unknown field raises `TypeError` (message is Python's, not specific to the harness); an empty `name` raises `ValueError`.
- **Types.** `HarnessConfig` is a frozen dataclass that checks `name` at construction: a non-`str` raises `TypeError`, an empty one `ValueError`, before anything runs. Other fields a subclass adds are not type-checked, and a subclass must also be a `@dataclass(frozen=True)`.
- **Related to** `ModelDriver.from_config` in `velari_ai` (`None` means defaults, overrides win), but not identical: this one takes a `defaults` argument at the config level, and it has no seed (the seed lives in `Environment`).

## 5. Extension

The extension points are the things a harness holds, not subclasses of it: write a `Candidate` (`run(input)` is required, `setup()` and `teardown()` optional; `CallableCandidate(fn)` wraps a plain function), a scorer function, or a `Task`. The caller owns the order `setup`, `run`, `teardown` and must call `teardown` even when `run` raised; the Runner does exactly that. To add settings, define a frozen `HarnessConfig` subclass and pass an instance to the harness's constructor (`for_task` builds only a plain `HarnessConfig`). To run candidates without a harness, use `Runner().run(candidate, inputs)`.

## 6. Design decisions

| Decision | Why | Alternative not taken | Cost |
|---|---|---|---|
| `Result` and `TrialRecord` each hold one shared `Outcome`, with one `Failure` | The two said the same thing twice (output or failure, plus a duration), and the reason-code rules lived only in the Runner | Nested `Outcome` classes inside each model (tried, more classes and a frozen-model workaround); `Outcome` as a base class that `Result` and `TrialRecord` subclass (tried, then composition preferred); separate copies | A trial and a run are the same shape (the setup, work, teardown block itself is still written twice, in `Harness.run()` and the Runner), so a change to one (for example a new field on `Outcome`) reaches both |
| Failures are data: a failed trial is a record, a raising scorer is `None`; nothing is raised | Later runs repeat many trials; one failure should not abort the rest, and the outcome must be recordable | Let exceptions propagate to the caller | Callers must look at `trials` or each record's `outcome.status`: a run where every trial failed still returns its records |
| `Harness` is the container that chains run, score and package; the `Runner` only runs | The roles were unclear: a harness base class with an `execute()` hook and a ready-made `TaskHarness` wrapped the same setup, work, teardown block the Runner wraps per trial, and scoring was hidden in the Runner. Now the harness reads top to bottom and each piece has one job | A generic extension base `Harness` with `execute()` plus a `TaskHarness` (built in increments 1 and 6, removed in increment 9); scoring inside the Runner | No hook for work that is not "candidate over task inputs"; a base class can be added back when a second kind of harness exists. A run no longer has a status of its own |
| Dataclass config, pydantic result | The config is built from OmegaConf and extended by subclasses; the result is serialised and validated, like the `velari_ai` response schemas | One type for both | Two validation regimes: the dataclass checks `name` itself, the pydantic result validates the rest |
| `Candidate` is a `typing.Protocol` with default `setup`/`teardown` and an abstract `run` | Any object with the three methods qualifies without subclassing (*verified* with pyright), and a subclass only has to write `run`; the base class gives the defaults | An `ABC`, as `Harness` is | Defaults are inherited only by explicit subclassing; a structural implementation must define all three methods. The default `setup` and `teardown` have an explicit `return None` body: the repo's IDE checker, `ty`, treats a Protocol method with a docstring-only (or `pass`) body as abstract, which made `CallableCandidate(...)` look like an abstract class being instantiated (pyright did not flag it) |
| `Environment` is a frozen dataclass holding only the seed, with `seed_for` | The one condition that can be used and tested today; per-trial seeds come from SHA-256 of the base seed, key and trial, because `hash()` differs between processes and a shared global RNG would couple trials | Add limits, policies and an artifacts directory now; seed the global RNG | A thin class (only the seed is read so far); `1` and `"1"` share a seed |
| A scorer that raises, or a failed trial, never becomes a zero: the first is `None`, the second has no scores. A scorer is a plain function, not a class or protocol | A score of 0 would be indistinguishable from a real mismatch; a plain function is the smallest thing a scorer can be | A `Scorer` class or protocol | A stateful scorer or one that needs the input has no home until a `Scorer` protocol is added |
| `Outcome.output` is an `Output`, which a candidate may return | One structure for what a candidate produced, so usage and files have a home on the record | Unwrapping it into fields on `Outcome` | `outcome.output` is no longer the plain value; scorers still get the value because the Harness unwraps it |
| `Score` is a structure with `metadata`, not a bare float | A float had nowhere to say why a score was given or to attach more; one shape can grow without another breaking change | A float plus parallel dictionaries for labels and notes | `record.scores[name]` is a `Score`, so the number is `.value` |
| `Example`, `Dataset` and `Task` live together in `schemas/dataset.py` | They are the data a run is given and are used together; one file, no import from `record.py` | `Example` in `record.py` (increment 12) and `Task` in `request/task.py` | `record.py` imports `dataset.py`, so the response side depends on the data it was given |
| An example carries its own reference, and a record keeps its example | A positional pairing of inputs and references could drift apart and needed a length check; with the example on the record, `Harness.score` needs no task and records can be scored again on their own | A parallel `references` sequence on `Task` (increment 8) | A task built from plain inputs needs `Example(...)` wrappers; `None` means no reference |
| `Task` holds a `Dataset`, not a sequence of examples | The task is a definition that must not change between runs, and the dataset gives it uids and a fingerprint | Accepting a plain list and wrapping it (more code) | Callers build a `Dataset` first; immutability is shallow, since the input values themselves are not copied |
| `HarnessConfig` is a frozen dataclass that checks its name at construction | A bad or changed config must not surface after the work is done; a dataclass keeps the OmegaConf and subclassing pattern | Pydantic config; only copying `name` before the hooks | A config subclass must also be a frozen dataclass; fields a subclass adds are not type-checked |
| `Runner` is a stateless class with `run(candidate, examples)` returning `List[TrialRecord]` | A class leaves room for limits and an error policy without changing callers; it only runs, so scoring can be applied or re-applied by the harness | A plain function; returning a `Result` | An empty class for now; no aggregate, so callers read raw records |
| The candidate lifecycle runs once per input | Isolation: each trial starts from a clean `setup` and always ends with `teardown` | Run `setup` once per run | A costly `setup` (building an agent, loading a model) repeats for every input |
| `schemas/` holds `dataset.py` (what a run is given), `request/` (`Environment`) and `response/` (what a run produces: `Result`, `TrialRecord`, `Score`) | `Example`, `Dataset` and `Task` are used together, so they share a file; `request/` keeps the one settings-like type until it has more | `Task` in `request/` (it moved out in increment 13); a flat `schemas/` | `request/` holds a single file |
| Folders `schemas`, `protocols`, `execution`; no `definition/` | Each file sits beside what it relates to: `Environment` in `schemas/request/` (what a run is run under), `HarnessConfig` with `Harness` in `execution/`. A `definition/` folder that held the two was removed because its name did not say what it held | A folder for settings (`settings/`) | Folders exist before most of their contents |
| `name` defaults to the task name; the harness has no seed | A name is needed for logs and results; the single seed belongs to the `Environment`, which is what must not vary between runs (it was once duplicated in `HarnessConfig` and `Result`, and removed) | A seed on both the config and the environment | Two harnesses of one class share a name unless overridden; each record reports its derived seed, but the `Result` reports no base seed until the run spec exists |

## 7. Quality attributes

| Attribute | State |
|---|---|
| Re-entrancy | `run()` can be called again on the same instance (*verified*: second run succeeds); it re-runs the candidate lifecycle for every input, and any state a candidate keeps persists between runs |
| Thread safety | Not designed or tested. The OmegaConf resolver registration is idempotent |
| Determinism | The only seed is `Environment.seed`; each record shows its derived seed, but a candidate is not told it yet |
| Observability | One `ERROR` log line per failed trial and per raising scorer; `duration_sec` on the outcome and the result |
| Consumer contract | `Result` fields as in the source; `trials` counts trials per status; each record's `outcome.status` is `success` or `error`, and an `error` record can still carry an `output` (a teardown failure after success), so check `status` first |

## 8. Defects and gaps

### Defects (behavior contradicted the code's own documentation)

None open. Three were found while reading the code and fixed in increment 7:

| ID | Was | Fix |
|---|---|---|
| D1 | `run()` raised a pydantic `ValidationError` for a mistyped config (`name=5`), after the hooks and `teardown` had run | `HarnessConfig` checks its name at construction (`TypeError` for a non-`str`), so a bad config fails before any work |
| D2 | `KeyboardInterrupt` or `SystemExit` in `setup` or `execute` skipped `teardown` | `run()` called `teardown` in a `finally`, as the Runner does. Moot since increment 9: the harness has no hooks, and the Runner's candidate `teardown` is in a `finally` |
| D3 | The config was mutable and read after the hooks, so a subclass could change the result's `name` | `HarnessConfig` is a frozen dataclass |

### Gaps and hygiene

- `config.py` defines a module `logger` that nothing uses.
- `velari-ai` and `pandas` are declared dependencies with no import.
- `Candidate.run(input)` names its parameter `input`, which shadows the builtin (the lint rule for it is not enabled), and an output's `value` is untyped (`Any`) until a schema check exists.
- A `BaseException` in a trial discards the records of the trials that already finished.
- Two examples with the same content and no `uid` raise (they hash to the same uid); repeated inputs need explicit uids.
- A uid or fingerprint is a hash of `json.dumps(..., default=str)`, so a non-JSON input hashes through its `str()`, which must be stable across runs (a DataFrame or an object with a default repr is not).
- The `Result` has no run spec, so the dataset fingerprint, task version and base seed are not saved with it; it comes with the record store, when something reads it.
- `outcome.output.value` and `example.input` are `Any`; the JSON round trip needs JSON-serialisable values.
- `Output.artifacts` are recorded, not copied; collecting files needs a trial working directory (`TrialContext`).
- A candidate cannot use its trial's seed yet (that needs `TrialContext`), the `Result` reports no base seed, and the `Environment` has no limits, error or retry policies, fail-fast settings, artifacts directory or lazy `Experiment`.
- `Task` has no dataset, preparation steps, trials or missing-score policy and the Runner does not accept a `Task`; callers pass `task.examples`.
- A score has no `Scorer` protocol, and there are no metrics or reducers beyond the mean, standard error and counts in `Result.summary` (`invalid` counts scores without a value, not failed trials). `Harness.score()` can score records again, but there is no `score=False` and no stored records to re-score; scorers receive only the output's value, not the whole `Output` or the example's input.
- The Runner does not limit, retry, time out, seed or run trials concurrently.
- Not verified: whether importing `velari_core` configures logging as a side effect.

## 9. Open architectural questions

1. **Inheritance or composition (Q1), resolved.** A harness is a container that holds a task, a candidate and scorers and chains run, score and package. There is no subclassing hook.
2. **`Result` and `run()` naming (Q2).** The implemented `Result` holds the records of one run; the glossary's `Result` also holds a spec and stats. `Harness.run()` and `Runner.run()` still share a verb: the Runner runs trials, the harness runs the whole measurement.
3. **Config validation (Q3), resolved.** The config is validated at construction and frozen, as a dataclass (not pydantic); a subclass must also be a frozen dataclass. Type-checking fields a subclass adds is left to the subclass.
4. **Teardown semantics (Q4), resolved.** `teardown` runs for a `BaseException` too, then the exception propagates.
5. **Value on error (Q5), moot.** A run has no `ERROR` result any more; a trial that failed only in `teardown` keeps its `output` in its record.
6. **`cfg` shape (Q6).** Should `from_config` accept a root config and select its own section, or stay section-only?
7. **Harness and Runner (Q7), resolved.** The Harness holds the candidate and calls the Runner; the Runner only runs trials. The Harness holds an `Environment` (increment 14).
8. **Runner and Task (Q8).** Should `Runner.run` take a `Task` (and read its examples and later its scorers), or stay `run(candidate, examples)` with the caller passing `task.examples`?
9. **Two seeds (Q9), resolved.** There is one seed, owned by `Environment`; `HarnessConfig` and `Result` no longer carry one. A harness gets its `Environment` as an argument (increment 14); each record reports its derived seed, and the `Result` reports no base seed until the run spec exists.
10. **Harness or TaskHarness (Q10), resolved.** One `Harness` class; `TaskHarness` and the extension base were removed in increment 9.

## 10. Testing

Run with `uv run pytest tests/harness`.

| File | Covers |
|---|---|
| `tests/harness/test_velari_harness.py` | Package imports; `__version__`; `logger` |
| `tests/harness/core/execution/test_config.py` | Defaults; precedence of overrides, `cfg` and defaults; resolver registration; unknown field and non-mapping `TypeError`; empty name `ValueError`; non-`str` name `TypeError`; the config is immutable |
| `tests/harness/core/execution/test_harness.py` | Named after the task with a `name` override; trials counted per status, including every trial failing; scoring through the harness, no scores for a failed trial, a raising scorer gives `None`; `score()` on records already run; invalid scorers `ValueError`, including a example missing its reference; JSON round trip; the harness's environment reaches every record. Records in order, the per-input lifecycle and one task against two candidates are covered in `test_runner.py` and `test_task.py` |
| `tests/harness/core/schemas/response/test_result.py` | Negative duration; `trials` counts, including in the JSON, and zero counts without records; `usage` totals per key in the JSON; `summary`: mean and counts in the JSON and a round trip, the standard error, no standard error below two valid scores, no mean when every score is invalid |
| `tests/harness/core/schemas/request/test_environment.py` | Pinned derived seeds; same arguments give the same seed; a changed seed, key or trial changes it; 32-bit range; global random state untouched; non-int seed `TypeError`; immutability |
| `tests/harness/core/schemas/test_dataset.py` | Examples kept in order and copied into a tuple; a given uid kept, a missing one hashed from the content; empty name, no examples or duplicate uids `ValueError`; the fingerprint changes with an input, a reference or the order and is stable for identical data; `Task` name check and immutability; one task run against two candidates |
| `tests/harness/core/execution/test_runner.py` | One record per input in order; lifecycle once per input; empty inputs; a failing input recorded while later inputs run; failing `setup` skips `run` and still tears down; `teardown` failure after success keeps the output; `teardown` failure not masking an earlier error; `KeyboardInterrupt` propagates after `teardown` ran; the candidate receives only the example's input, never its reference; a returned `Output` is kept and a plain return is wrapped in one; each record's seed is the environment's derived seed (for the example's uid, or its index), is repeatable, changes with the base seed, and defaults to seed 42 |
| `tests/harness/core/schemas/response/test_record.py` | `TrialRecord` JSON round trip; negative duration; `status` derived from the failure |
| `tests/harness/core/protocols/test_candidate.py` | `CallableCandidate` returns the function's result and propagates exceptions; a subclass with only `run` inherits no-op hooks; overridden hooks run in order; a subclass without `run` cannot be instantiated |

Not covered by any test: the global-RNG guarantee.

## 11. Where to look next

- [Glossary](../glossary.md): definitions, and implemented versus planned.
- [Environment note](environment.md): what an environment is, how the layers use it, and how it grows.
- [Plan](../plan/plan.md) and [stage 2](../plan/stage-2-base-framework.md): what comes next and why.
