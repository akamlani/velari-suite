from    dataclasses  import dataclass, field
from    typing       import Sequence
# package modules
from    .case import Case


@dataclass(frozen=True)
class Task(object):
    """What is measured: a named, versioned set of cases. It says nothing about who is measured.

    Immutable: `cases` is stored as a tuple.

    Args:
        name (str): Identifies the task in logs and results; must not be empty.
        cases (Sequence[Case]): What the candidate is run on, one trial each, with the reference a scorer may use.
        version (str): Bump it when the cases or their meaning change, so old and new results are not mixed.

    Raises:
        ValueError: If `name` is empty.

    Examples:
        One task, two candidates (a baseline rule and a variant), run through the same cases:

        >>> task = Task("ticket-routing", [Case("Card was charged twice for invoice INV-2291", "billing"),
        ...                                Case("Login fails with a 500 error", "general")])
        >>> baseline = CallableCandidate(lambda text: "general")
        >>> variant  = CallableCandidate(lambda text: "billing" if "invoice" in text.lower() else "general")
        >>> [record.outcome.output for record in Runner().run(baseline, task.cases)]
        ['general', 'general']
        >>> [record.outcome.output for record in Runner().run(variant, task.cases)]
        ['billing', 'general']

        Applied (illustrative): a task for an agent evaluation, run against a `velari_ai` agent candidate:

        >>> task = Task("support-agent-quality", [Case("What's the balance on ACC-10293?"), Case("Summarize ticket ORG-4471.")], version="2")
    """
    name:    str
    cases:   Sequence[Case]
    version: str             = field(default="1")

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Task.name must not be empty")
        object.__setattr__(self, "cases", tuple(self.cases))
