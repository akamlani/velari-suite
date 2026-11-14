from    abc     import abstractmethod
from    typing  import Any, Callable, Protocol


class Candidate(Protocol):
    """The thing a harness runs and measures: anything that takes an input and returns an output.

    The caller owns the order `setup()`, `run()`, `teardown()` and must call `teardown()` even when `run()` raised.

    Examples:
        A plain function as a candidate (the core case), here a rule that finds an account's plan tier:

        >>> def plan_tier(account_id: str) -> str:
        ...     return "enterprise" if account_id.startswith("ORG") else "team"
        >>> candidate = CallableCandidate(plan_tier)
        >>> candidate.run("ORG-4471")
        'enterprise'

        An applied candidate that wraps a `velari_ai` agent, built once in `setup()`:

        >>> class BillingAgentCandidate(Candidate):
        ...     def setup(self) -> None:
        ...         agent = Agent(agent_config=AgentConfig(name="billing-support-agent"))
        ...         self._agent = agent.build([lookup_account_balance], system_prompt="You are a billing support assistant.")
        ...     def run(self, input: Any) -> Any:
        ...         return self._agent.run(input).response.content
        >>> candidate = BillingAgentCandidate()
        >>> candidate.setup()
        >>> candidate.run("What's the balance on ACC-10293?")
        >>> candidate.teardown()
    """
    def setup(self) -> None:
        """Prepare for a trial (load a model, build an agent, open a client); the default does nothing."""
        return None

    @abstractmethod
    def run(self, input: Any) -> Any:
        """Take one input and return the candidate's output; exceptions propagate to the caller."""

    def teardown(self) -> None:
        """Release what `setup()` and `run()` acquired; the default does nothing."""
        return None


class CallableCandidate(Candidate):
    """Wrap any function as a candidate, so a plain callable needs no class of its own.

    Args:
        fn (Callable[[Any], Any]): Called with each input; its return value is the candidate's output.

    Examples:
        >>> candidate = CallableCandidate(str.upper)
        >>> candidate.setup()
        >>> candidate.run("acc-10293")
        'ACC-10293'
        >>> candidate.teardown()
    """
    def __init__(self, fn: Callable[[Any], Any]) -> None:
        self._fn = fn

    def run(self, input: Any) -> Any:
        return self._fn(input)
