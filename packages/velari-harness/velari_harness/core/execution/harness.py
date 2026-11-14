import  logging
import  time
from    omegaconf  import DictConfig
from    typing     import Any, Callable, Dict, List, Optional, Self, Sequence, Union
# package modules
from    .config                    import HarnessConfig
from    .runner                    import Runner
from    ..protocols.candidate      import Candidate
from    ..schemas.request.task     import Task
from    ..schemas.response.record  import Failure, Score, TrialRecord
from    ..schemas.response.result  import Result

logger = logging.getLogger(__name__)


class Harness(object):
    """The container for one measurement: a candidate, the task it is measured on, and the scorers that judge it.

    `run()` chains the Runner, `score()` and the `Result`; a failed trial stays in its record and is not scored.

    Args:
        config (HarnessConfig): The name.
        task (Task): What is measured; its examples are what the candidate runs on.
        candidate (Candidate): What is measured.
        scorers (Sequence[Callable[[Any, Any], Union[float, Score]]]): `fn(output, reference)` returning a number, bool or `Score`; keyed by distinct `__name__`.

    Raises:
        ValueError: If `scorers` are given but an example has no `reference`, or two scorers share a name.

    Examples:
        One task and one candidate, scored against the expected churn-risk bands:

        >>> task = Task("churn-risk", [Example(3, "low"), Example(45, "high")])
        >>> candidate = CallableCandidate(lambda days_inactive: "high" if days_inactive > 30 else "low")
        >>> def same_band(output, reference):
        ...     return output == reference
        >>> result = Harness.for_task(task, candidate, scorers=[same_band]).run()
        >>> result.name, [(r.outcome.status.value, r.outcome.output.value) for r in result.records]
        ('churn-risk', [('success', 'low'), ('success', 'high')])
        >>> result.scores
        {'same_band': 1.0}

        Applied (illustrative): an agent candidate on an evaluation task:

        >>> task = Task("support-agent-quality", [Example("What's the balance on ACC-10293?"), Example("Summarize ticket ORG-4471.")])
        >>> result = Harness.for_task(task, BillingAgentCandidate()).run()
    """
    def __init__(self, config: HarnessConfig, task: Task, candidate: Candidate,
                 scorers: Sequence[Callable[[Any, Any], Union[float, Score]]] = ()) -> None:
        if scorers and any(example.reference is None for example in task.examples):
            raise ValueError(f"scorers need a reference on every example of task {task.name!r}")
        self._scorers = {getattr(scorer, "__name__", repr(scorer)): scorer for scorer in scorers}
        if len(self._scorers) != len(scorers):
            raise ValueError(f"scorer names must be distinct, got {list(self._scorers)}")
        self._config = config
        self._task = task
        self._candidate = candidate

    @classmethod
    def for_task(cls, task: Task, candidate: Candidate, cfg: Optional[DictConfig] = None,
                 scorers: Sequence[Callable[[Any, Any], Union[float, Score]]] = (), **overrides: Any) -> Self:
        """Build a harness for a task and a candidate; it is named after the task unless `name` is given.

        Args:
            task (Task): What is measured.
            candidate (Candidate): What is measured on it.
            cfg (Optional[DictConfig]): `HarnessConfig` fields (just `name`); `None` keeps the defaults.
            scorers (Sequence[Callable[[Any, Any], Union[float, Score]]]): Functions `fn(output, reference)`; every example must have a `reference`.
            **overrides (Any): Fields that take precedence over `cfg`.

        Returns:
            Self: A new harness, not yet run.
        """
        return cls(HarnessConfig.from_config(cfg, defaults={"name": task.name}, **overrides), task, candidate, scorers)

    @property
    def config(self) -> HarnessConfig:
        return self._config

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def task(self) -> Task:
        return self._task

    @property
    def candidate(self) -> Candidate:
        return self._candidate

    def run(self) -> Result:
        """Run the candidate on every example, score the successful trials, and return the records with their timing.

        Returns:
            Result: One record per task example, in order, with the time the whole run took.
        """
        start = time.perf_counter()
        records = self.score(Runner().run(self._candidate, self._task.examples))
        return Result(name=self.name, records=records, duration_sec=time.perf_counter() - start)

    def score(self, records: Sequence[TrialRecord]) -> List[TrialRecord]:
        """Return `records` with each successful trial scored by every scorer; a failed trial is returned as it is.

        Args:
            records (Sequence[TrialRecord]): Records from running this harness's task.

        Returns:
            List[TrialRecord]: Copies of the records; the originals are not changed.
        """
        return [
            record if record.outcome.failure is not None or record.outcome.output is None
            else record.model_copy(update={"scores": self._scores(record.outcome.output.value, record.example.reference)})
            for record in records
        ]

    def _scores(self, output: Any, reference: Any) -> Dict[str, Score]:
        # a scorer that raises is data too: a Score with its failure and no value, never a zero
        scores: Dict[str, Score] = {}
        for name, scorer in self._scorers.items():
            try:
                returned = scorer(output, reference)
                scores[name] = returned if isinstance(returned, Score) else Score(value=float(returned), passed=returned if isinstance(returned, bool) else None)
            except Exception as exc:
                logger.error(f"scorer {name} failed: {type(exc).__name__}: {exc}")
                scores[name] = Score(failure=Failure.of(exc))
        return scores
