from    dataclasses  import dataclass, field
from    typing       import Any, Sequence


@dataclass(frozen=True)
class Task(object):
    """What is measured: a named, versioned set of inputs. It says nothing about who is measured.

    A task never names a candidate, so one task can be run against several candidates, such as a
    baseline and a variant, and the results compared. It is immutable: `inputs` is stored as a tuple.
    The inputs are plain values for now; they become a dataset of cases, with scorers, later.

    Args:
        name (str): Identifies the task in logs and results; must not be empty.
        inputs (Sequence[Any]): Values a candidate is run on, one trial each.
        version (str): Bump it when the inputs or their meaning change, so old and new results are not mixed.
        references (Sequence[Any]): The expected value per input, for scorers; the candidate never sees them. Empty means none, else one per input.

    Examples:
        One task, two candidates (a baseline rule and a variant), run through the same inputs:

        >>> task = Task("ticket-routing", ["Card was charged twice for invoice INV-2291", "Login fails with a 500 error"])
        >>> baseline = CallableCandidate(lambda text: "general")
        >>> variant  = CallableCandidate(lambda text: "billing" if "invoice" in text.lower() else "general")
        >>> [record.outcome.output for record in Runner().run(baseline, task.inputs)]
        ['general', 'general']
        >>> [record.outcome.output for record in Runner().run(variant, task.inputs)]
        ['billing', 'general']

        With references, so a scorer can judge each output:

        >>> task = Task("ticket-routing", ["Card was charged twice for invoice INV-2291", "Login fails with a 500 error"], references=["billing", "general"])

        Applied (illustrative): a task for an agent evaluation, run against a `velari_ai` agent candidate:

        >>> task = Task("support-agent-quality", ["What's the balance on ACC-10293?", "Summarize ticket ORG-4471."], version="2")
    """
    name:       str
    inputs:     Sequence[Any]
    version:    str            = field(default="1")
    references: Sequence[Any]  = field(default=())

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Task.name must not be empty")
        if self.references and len(self.references) != len(self.inputs):
            raise ValueError(f"Task.references has {len(self.references)} items for {len(self.inputs)} inputs")
        for name in ("inputs", "references"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
