# Harness Example (core)

A runnable example for `velari_harness`, using only the core that is implemented today: `Task`,
`Candidate`, a scorer function, `Harness` (the container that runs the candidate through the `Runner` and scores the
result), `TrialRecord`, `Result` and `Status`. It is deliberately not an applied harness (no LLM, agent or model): the task is validating the
rows of a customer CSV file, and the candidate is a small rule-based validator.

- **`harness_cli.py`** — the whole example in one short script: it builds a task from the config and the CSV,
  runs one candidate on it through a `Harness`, and prints the run and the outcome per row.
- **`examples/_conf/harness/app_harness.yaml`** — the `task:` section (name, version, and the path of the CSV
  file) and the `app.info` block that names the CLI.
- **`data/customers.csv`** — the task's inputs, one customer per row (`id,name,email,age,country`) plus an
  `expected_email` column, the reference for the scorer: six clean rows and six messy ones. One clean row's
  expected address deliberately differs, to show a successful trial with a score of `0.0`. Edit it to change the run.

## Structure

`CustomerValidator` is the candidate. `HarnessCli` is the only other class, in three separate steps that chain:
`cli.run().render()`.

1. **Create** (in `__init__`): read the CSV with pandas (every column as text, empty cells kept as `""`, so the
   validator sees the raw values), build the `Task` from the rows, the references and the config, and a `Harness` that pairs
   the candidate and the scorer with the task.
2. **Run** (`run()`): run the harness and keep its `Result`; nothing is printed.
3. **Render** (`render()`): print the run and the outcome per row. Call `run()` first.

The Typer command (no options) is built in the `__main__` block. Failed trials are data in the tables, so they do
not change the exit code.

## What it shows

| Core piece | Where in the example |
|---|---|
| `Task(name, inputs, version, references)` | Built from the `task:` section; the inputs are the rows of `data/customers.csv`, read with pandas, each a dict keyed by the header, and the references are its `expected_email` column (dropped from the inputs, so the candidate never sees it) |
| `Candidate` subclass | `CustomerValidator`: `setup()` compiles the email pattern, `run(row)` returns a cleaned record or raises, `teardown()` releases the pattern. The Runner calls them once per row |
| Scorer function | `email_match(output, reference)`: a plain function returning a `bool` (1.0 or 0.0), here reusing `exact_match` from `velari_ai.ai.evals.heuristics`; its name is the key in each record's `scores`. Score and status are separate: row 11 succeeds and scores `0.0`; a failed trial has no output, so it gets no score |
| `Harness.for_task(task, candidate, scorers=[...])` | The container, named after the task. Its `run()` chains three steps: run the candidate on every input, score each successful trial, package the result |
| `Runner` and `TrialRecord` (through `Harness`) | The result's `records` hold one trial record per row, in order |
| Failures are data | A bad row (an email with no domain, a negative or non-numeric age, an unknown country, a missing name or age) fails its own trial with the reasons; later rows still run, and the other rows are unaffected |
| `Result` and `Status` | The run table: harness, inputs, duration, trials success, trials error; and the scores table (`Result.scores`: the mean per scorer) |

## Prerequisites

No extras. From the repository root:

```
uv sync
```

## Run

```
uv run python examples/ai/harness/harness_cli.py
```

The command takes no options. The task comes from `examples/_conf/harness/app_harness.yaml`:

```
task:
  name: customer-validation
  version: "1"
  inputs_path: examples/ai/harness/data/customers.csv   # relative to the repository root
```

Output (the example silences the Runner's per-trial error logging, because the table already shows each failure;
durations vary):

```
                                  harness run                                  
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ harness             ┃ inputs ┃ duration_sec ┃ trials success ┃ trials error ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ customer-validation │ 12     │ 0.0035       │ 6              │ 6            │
└─────────────────────┴────────┴──────────────┴────────────────┴──────────────┘
        scores        
┏━━━━━━━━━━━━━┳━━━━━━┓
┃ scorer      ┃ mean ┃
┡━━━━━━━━━━━━━╇━━━━━━┩
│ email_match │ 0.83 │
└─────────────┴──────┘
               customer-validation (version 1): outcome per input               
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ input               ┃ status  ┃ reason code   ┃ score ┃ output / error       ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ 1,Ana               │ success │               │ 1.0   │ {'id': 1, 'name':    │
│ Pérez,ana.perez@ex… │         │               │       │ 'Ana Pérez',         │
│                     │         │               │       │ 'email':             │
│                     │         │               │       │ 'ana.perez@example.… │
│                     │         │               │       │ 'age': 34,           │
│                     │         │               │       │ 'country': 'ES'}     │
│ 2,Li                │ success │               │ 1.0   │ {'id': 2, 'name':    │
│ Wei,li.wei@example… │         │               │       │ 'Li Wei', 'email':   │
│                     │         │               │       │ 'li.wei@example.com… │
│                     │         │               │       │ 'age': 29,           │
│                     │         │               │       │ 'country': 'CN'}     │
│ 3,Sam               │ success │               │ 1.0   │ {'id': 3, 'name':    │
│ Carter,sam.carter@… │         │               │       │ 'Sam Carter',        │
│                     │         │               │       │ 'email':             │
│                     │         │               │       │ 'sam.carter@example… │
│                     │         │               │       │ 'age': 41,           │
│                     │         │               │       │ 'country': 'US'}     │
│ 4,Maria             │ success │               │ 1.0   │ {'id': 4, 'name':    │
│ Rossi,maria.rossi@… │         │               │       │ 'Maria Rossi',       │
│                     │         │               │       │ 'email':             │
│                     │         │               │       │ 'maria.rossi@exampl… │
│                     │         │               │       │ 'age': 52,           │
│                     │         │               │       │ 'country': 'IT'}     │
│ 5,John              │ error   │ invalid_input │       │ ValueError: invalid  │
│ Smith,john.smith@e… │         │               │       │ email:               │
│                     │         │               │       │ 'john.smith@example' │
│ 6,Aisha             │ error   │ invalid_input │       │ ValueError: invalid  │
│ Khan,aisha.khan@ex… │         │               │       │ age: '-4'            │
│ 7,Tom               │ error   │ invalid_input │       │ ValueError: invalid  │
│ Brown,tom.brown@ex… │         │               │       │ age: 'abc'           │
│ 8,Chloé             │ error   │ invalid_input │       │ ValueError: unknown  │
│ Martin,chloe.marti… │         │               │       │ country: 'XX'        │
│ 9,,empty.name@exam… │ error   │ invalid_input │       │ ValueError: missing  │
│                     │         │               │       │ name                 │
│ 10,Omar             │ error   │ invalid_input │       │ ValueError: invalid  │
│ Ali,omar.ali@examp… │         │               │       │ age: ''              │
│ 11,Eva              │ success │               │ 0.0   │ {'id': 11, 'name':   │
│ Novak,eva.novak@ex… │         │               │       │ 'Eva Novak',         │
│                     │         │               │       │ 'email':             │
│                     │         │               │       │ 'eva.novak@example.… │
│                     │         │               │       │ 'age': 67,           │
│                     │         │               │       │ 'country': 'CZ'}     │
│ 12,Noah             │ success │               │ 1.0   │ {'id': 12, 'name':   │
│ Kim,noah.kim@examp… │         │               │       │ 'Noah Kim', 'email': │
│                     │         │               │       │ 'noah.kim@example.c… │
│                     │         │               │       │ 'age': 23,           │
│                     │         │               │       │ 'country': 'KR'}     │
└─────────────────────┴─────────┴───────────────┴───────┴──────────────────────┘
```

Six trials failed, and the run still returned all twelve records: each trial record says whether its own row
worked, and the run table counts them. The command exits 0 either way.

## The candidate lifecycle

Only the candidate has one: its `setup`, `run` and `teardown` run once per row inside the Runner. In this example it
does the visible work (compiling and releasing the pattern, once per row). The harness has no lifecycle hooks of its
own; it only chains run, score and package.

## Notes

- Compiling the pattern in `setup()` repeats for every row because the Runner runs the candidate's lifecycle
  per input. It is cheap here; a costly `setup` would be a problem (listed as a gap in the architecture spec).
- The example uses the supported path only. It does not demonstrate the known gaps listed in
  `docs/packages/velari-harness/architecture/architecture.md`.

## Not here yet

- **Comparing candidates.** A harness is one candidate on one task. Lining several runs up (two candidates,
  or two environments) needs a run group and result set, which the core does not have yet.
- No metrics or reducers beyond the mean in the scores summary, no repeated trials, and no `Environment`
  (nothing reads one yet).
- The candidate is defined in the script, not built from config.
- The evaluation and data science categories and the planned core components (cases, datasets, metrics, a
  registry) are not implemented.

See `docs/packages/velari-harness/plan/` for the plan and `docs/packages/velari-harness/glossary.md` for
what is implemented versus planned.
