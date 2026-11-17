# uv run python examples/ai/harness/harness_cli.py
# Edit examples/_conf/harness/app_harness.yaml (the task's inputs) to change the run.
# See examples/ai/harness/README.md for details.

import  logging
import  numpy as np
import  pandas as pd
import  re
import  typer
from    pathlib       import Path
from    typing        import Any, ClassVar, Optional, Self
from    omegaconf     import DictConfig
from    rich.console  import Console
from    rich.table    import Table
# package modules
from    velari_ai.ai.evals.heuristics import exact_match
from    velari_core.core import read_root_dir
from    velari_core.core.io.partition.hydra import read_hydra
from    velari_harness.core.execution.harness import Harness
from    velari_harness.core.protocols.candidate import Candidate
from    velari_harness.core.schemas.dataset import Dataset, Example, Task
from    velari_harness.core.schemas.response.record import Score
from    velari_harness.core.schemas.request.environment import Environment
from    velari_harness.core.schemas.response.result import Result
from    velari_harness.core.schemas.types import Status

# Objective: run a customer-row validator (the candidate) on every row of a CSV file (the task), record each trial, and show which rows pass or fail and why.

# Terms used below, each with what it is in this example (full definitions: docs/packages/velari-harness/glossary.md)
#   Task       what is measured: a name, a version and a dataset of examples
#              here: "customer-validation", whose examples are the 12 rows of data/customers.csv
#   Environment  what is held fixed for every candidate: the base seed each trial's seed is derived from
#              here: Environment with base seed 42 (also set as NumPy's seed); each trial record shows its derived seed
#   Candidate  the thing being measured: it takes one input and returns an output
#              here: CustomerValidator, which takes one customer row and returns a cleaned record, or says what is wrong with it
#   Trial      one run of the candidate on one input; its record is "success" with an output, or "error" with a message
#              here: row 1 gives a success record; row 5 (an email with no domain) gives an error record
#   Harness    the container: it runs the candidate on every input, then scores each successful trial
#              here: a Harness pairing CustomerValidator and email_match with the task (a Runner runs the trials)
#   Scorer     judges one output against the reference (the expected value), without the candidate ever seeing the reference
#              here: the function email_match, which compares the cleaned email with the expected_email column (a scorer can be a plain function);
#              a failed trial has no output, so it gets no score
#   Result     what one harness run returns: its duration, one trial record per input, and counts and a summary per scorer (mean, stderr, counts)
#              here: status "success" (the run completed), with 6 trials success and 6 trials error


class CustomerValidator(Candidate):
    """The candidate: checks one customer row (a dict from the CSV) and returns a cleaned record."""
    EMAIL     = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    COUNTRIES = {"US", "GB", "CA", "ES", "IT", "FR", "DE", "CN", "KR", "AE", "CZ", "JP"}

    def setup(self) -> None:
        # the resource this candidate holds: the compiled email pattern (the Runner calls setup once per input)
        self.email = re.compile(self.EMAIL)

    def run(self, input: Any) -> Any:
        name, email = (input.get("name") or "").strip(), (input.get("email") or "").strip()
        age, country = (input.get("age") or "").strip(), (input.get("country") or "").strip()
        # collect every problem with the row; raising is how a candidate reports a failed trial
        problems = []
        if not name:
            problems.append("missing name")
        if not self.email.match(email):
            problems.append(f"invalid email: {email!r}")
        if not age.isdigit() or not 0 <= int(age) <= 120:
            problems.append(f"invalid age: {age!r}")
        if country not in self.COUNTRIES:
            problems.append(f"unknown country: {country!r}")
        if problems:
            raise ValueError("; ".join(problems))
        return {"id": int(input["id"]), "name": name, "email": email.lower(), "age": int(age), "country": country}

    def teardown(self) -> None:
        del self.email


def email_match(output: Any, reference: Any) -> Score:
    # a scorer is a plain function of (output, reference); it may return a number, a bool or a Score, and its name is the key in the record's scores
    # score is separate from status: a trial can succeed and still score 0.0
    matched = exact_match(output["email"], reference)
    return Score(value=float(matched), passed=matched, label="match" if matched else "mismatch", metadata={"expected": reference})


class HarnessCli(object):
    """Run one candidate on the configured task through a Harness, then print its trial records. Calls chain: `cli.run().render()`."""
    CANDIDATE: ClassVar[Candidate] = CustomerValidator()

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        # the Environment's base seed: set NumPy's global seed, as Experiment.seed_init does; each trial's seed is derived from it
        np.random.seed(42)
        # the Task is what is measured: a name, a version and the dataset of examples the candidate is run on.
        # Each example is a CSV row: its input is the row as a dict keyed by the header (id, name, email, age, country), and its reference
        # is the expected_email column, which goes to the scorer and never reaches the candidate.
        # pandas reads every column as text (dtype=str) with empty cells kept as "", so the validator sees the raw values.
        df_customers = pd.read_csv(Path(read_root_dir()) / cfg.task.inputs_path, dtype=str, keep_default_na=False)
        df_inputs = df_customers.drop(columns="expected_email")
        examples = [Example(row, reference, uid=row["id"]) for row, reference in zip(df_inputs.to_dict("records"), df_customers["expected_email"])]
        self.task = Task(cfg.task.name, Dataset(Path(cfg.task.inputs_path).stem, examples), version=str(cfg.task.version))
        # step 1, create: the harness pairs the candidate and the scorer with the task (it is named after the task)
        self.harness = Harness.for_task(self.task, self.CANDIDATE, scorers=[email_match], environment=Environment(seed=42))
        self.result: Optional[Result] = None   # set by run()

    @classmethod
    def from_config(cls, path: Path) -> Self:
        cfg = read_hydra(str(path))
        if cfg is None:
            raise ValueError(f"failed to load config: {path}")
        return cls(cfg)

    def run(self) -> Self:
        # step 2, run: the result holds one trial record per input; a failed trial is an error record, not an exception
        self.result = self.harness.run()
        return self

    def render(self) -> Self:
        # step 3, render: print what run() produced
        if self.result is None:
            raise RuntimeError("call run() before render()")
        result = self.result

        # table 1: the harness run: how many inputs, how long, and how many trials ended in each status
        run = Table(title="harness run")
        for column in ("harness", "inputs", "duration_sec", "trials success", "trials error"):
            run.add_column(column)
        run.add_row(result.name, str(len(result.records)), f"{result.duration_sec:.4f}",
                    str(result.trials[Status.SUCCESS]), str(result.trials[Status.ERROR]))

        # table 2: the scorer's summary over the run (mean, standard error and counts of the scores)
        summary = Table(title="scores")
        for column in ("scorer", "mean", "stderr", "n", "invalid"):
            summary.add_column(column)
        for name, stats in result.summary.items():
            summary.add_row(name, f"{stats.mean:.2f}" if stats.mean is not None else "",
                            f"{stats.stderr:.2f}" if stats.stderr is not None else "", str(stats.n), str(stats.invalid))

        # table 3: one row per input, with the trial's seed, status, reason code (empty on success), its score (empty on failure) and its output or error message
        trials = Table(title=f"{self.task.name} (version {self.task.version}, data {self.task.dataset.fingerprint}): outcome per input")
        for column in ("input", "seed", "status", "reason code", "score", "output / error"):
            trials.add_column(column)
        for record in result.records:
            failure = record.outcome.failure   # None when the trial succeeded
            output = record.outcome.output     # None when the trial failed
            color = "red" if failure else "green"
            score = ",".join(f"{s.value:.1f} {s.label}" for s in record.scores.values() if s.value is not None)
            trials.add_row(",".join(str(v or "") for v in record.example.input.values()), str(record.seed), f"[{color}]{record.outcome.status.value}[/{color}]",
                           str(failure.reason_code) if failure else "", score, failure.error if failure else repr(output.value if output else None))

        Console().print(run)
        Console().print(summary)
        Console().print(trials)
        return self


if __name__ == "__main__":
    # the Runner logs every failed trial at ERROR level; the table below already shows them, so silence the package logger
    logging.getLogger("velari_harness").setLevel(logging.CRITICAL)
    # build the command-line app: one command, no options; its name and help text come from the config
    path = Path(read_root_dir()) / "examples" / "_conf" / "harness" / "app_harness.yaml"
    cli = HarnessCli.from_config(path)
    app = typer.Typer(rich_markup_mode="rich", add_completion=False)
    app.info.name = cli.cfg.app.info.name
    app.info.help = cli.cfg.app.info.help

    @app.command(help="Run the task through a Harness and print its trial results.")
    def run() -> None:
        # failed trials are data in the printed tables, so they do not change the exit code
        cli.run().render()

    app()
