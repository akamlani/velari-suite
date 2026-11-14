from    dataclasses  import dataclass, field
from    typing       import Sequence
# package modules
from    ..response.record import Example


@dataclass(frozen=True)
class Task(object):
    """What is measured: a named, versioned set of examples. It says nothing about who is measured.

    Immutable: `examples` is stored as a tuple.

    Args:
        name (str): Identifies the task in logs and results; must not be empty.
        examples (Sequence[Example]): What the candidate is run on, one trial each, with the reference a scorer may use.
        version (str): Bump it when the examples or their meaning change, so old and new results are not mixed.

    Raises:
        ValueError: If `name` is empty.

    Examples:
        One task, two candidates (a baseline rule and a variant), run through the same examples:

        >>> task = Task("ticket-routing", [Example("Card was charged twice for invoice INV-2291", "billing"),
        ...                                Example("Login fails with a 500 error", "general")])
        >>> baseline = CallableCandidate(lambda text: "general")
        >>> variant  = CallableCandidate(lambda text: "billing" if "invoice" in text.lower() else "general")
        >>> [record.outcome.output.value for record in Runner().run(baseline, task.examples)]
        ['general', 'general']
        >>> [record.outcome.output.value for record in Runner().run(variant, task.examples)]
        ['billing', 'general']

        Applied (illustrative): a task for an agent evaluation, run against a `velari_ai` agent candidate:

        >>> task = Task("support-agent-quality", [Example("What's the balance on ACC-10293?"), Example("Summarize ticket ORG-4471.")], version="2")
    """
    name:      str
    examples:  Sequence[Example]
    version:   str                 = field(default="1")

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Task.name must not be empty")
        object.__setattr__(self, "examples", tuple(self.examples))
