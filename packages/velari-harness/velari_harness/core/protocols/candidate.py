import  copy
import  time
from    abc      import abstractmethod
from    pathlib  import Path
from    typing   import Any, Callable, Dict, List, Optional, Protocol
# package modules
from    ..schemas.response.record import Event


class TrialContext(object):
    """What a candidate gets for one trial: its seed, a private working directory, and a way to record events.

    Built by the runner, one per trial; not global state.

    Args:
        seed (int): The seed derived for this trial.
        workdir (Path): A directory only this trial uses; it exists for the whole trial.

    Examples:
        A priority classifier noting which rule fired, then reading what it recorded:

        >>> ctx = TrialContext(seed=4121104260, workdir=Path("/tmp/trial-0"))
        >>> ctx.emit("rule_matched", {"rule": "outage"})
        >>> [event.name for event in ctx.events]
        ['rule_matched']
    """
    def __init__(self, seed: int, workdir: Path) -> None:
        self.seed = seed
        self.workdir = workdir
        self._start = time.perf_counter()
        self._events: List[Event] = []

    @property
    def events(self) -> List[Event]:
        return [event.model_copy(deep=True) for event in self._events]

    def emit(self, name: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Record that `name` happened, with optional details, stamped with the time since the trial began.

        Args:
            name (str): What happened, such as `skill_triggered`.
            data (Optional[Dict[str, Any]]): Its details, copied when emitted; keep to values that serialize to JSON.
        """
        self._events.append(Event(name=name, data=copy.deepcopy(data or {}), elapsed_sec=time.perf_counter() - self._start))


class Candidate(Protocol):
    """The thing a harness runs and measures: anything that takes an input and returns an output.

    The caller owns the order `setup(ctx)`, `run(input, ctx)`, `teardown(ctx)` and must call `teardown(ctx)` even when `run()` raised.
    `ctx` is the trial's `TrialContext` (its seed, a private working directory and `emit()`); a candidate that needs none of it ignores it.

    Examples:
        A plain function as a candidate (the core case), here a rule that finds an account's plan tier:

        >>> def plan_tier(account_id: str) -> str:
        ...     return "enterprise" if account_id.startswith("ORG") else "team"
        >>> ctx = TrialContext(seed=0, workdir=Path("/tmp/trial-0"))
        >>> candidate = CallableCandidate(plan_tier)
        >>> candidate.run("ORG-4471", ctx)
        'enterprise'

        A candidate that uses its context: it records which queue it sent a request to:

        >>> class RefundRouter(Candidate):
        ...     def run(self, input: Any, ctx: TrialContext) -> Any:
        ...         queue = "billing" if "refund" in input.lower() else "general"
        ...         ctx.emit("queue_selected", {"queue": queue})
        ...         return queue
        >>> RefundRouter().run("Please refund the duplicate charge on invoice INV-2207", ctx)
        'billing'

        An applied candidate that wraps a `velari_ai` agent, built once in `setup()`:

        >>> class BillingAgentCandidate(Candidate):
        ...     def setup(self, ctx: TrialContext) -> None:
        ...         agent = Agent(agent_config=AgentConfig(name="billing-support-agent"))
        ...         self._agent = agent.build([lookup_account_balance], system_prompt="You are a billing support assistant.")
        ...     def run(self, input: Any, ctx: TrialContext) -> Any:
        ...         return self._agent.run(input).response.content
        >>> candidate = BillingAgentCandidate()
        >>> candidate.setup(ctx)
        >>> candidate.run("What's the balance on ACC-10293?", ctx)
        >>> candidate.teardown(ctx)
    """
    def setup(self, ctx: TrialContext) -> None:
        """Prepare for a trial (load a model, build an agent, open a client); the default does nothing."""
        return None

    @abstractmethod
    def run(self, input: Any, ctx: TrialContext) -> Any:
        """Take one input and return a plain value or an `Output`; exceptions propagate to the caller."""

    def teardown(self, ctx: TrialContext) -> None:
        """Release what `setup()` and `run()` acquired; the default does nothing."""
        return None


class CallableCandidate(Candidate):
    """Wrap any function as a candidate, so a plain callable needs no class of its own.

    Args:
        fn (Callable[[Any], Any]): Called with each input; its return value is the candidate's output.

    Examples:
        >>> ctx = TrialContext(seed=0, workdir=Path("/tmp/trial-0"))
        >>> candidate = CallableCandidate(str.upper)
        >>> candidate.setup(ctx)
        >>> candidate.run("acc-10293", ctx)
        'ACC-10293'
        >>> candidate.teardown(ctx)
    """
    def __init__(self, fn: Callable[[Any], Any]) -> None:
        self._fn = fn

    def run(self, input: Any, ctx: TrialContext) -> Any:
        return self._fn(input)
