from __future__ import annotations
import  hashlib
import  json
from    dataclasses  import dataclass, field, replace
from    typing       import Any, Iterator, Optional, Sequence


def _canonical(payload: Any) -> Any:
    # json's sort_keys compares raw keys, so mixed key types raise; non-str keys get a type prefix so 1 and "1" stay distinct.
    if isinstance(payload, dict):
        return {(k if isinstance(k, str) else f"{type(k).__name__}:{k}"): _canonical(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_canonical(item) for item in payload]
    return payload


def _digest(payload: Any) -> str:
    # default=str keeps non-JSON values (dates, paths) hashable, but their str() must be stable across runs
    return hashlib.sha256(json.dumps(_canonical(payload), sort_keys=True, default=str).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Example(object):
    """One thing to measure: an input for the candidate and, optionally, the expected value for a scorer.

    The candidate sees only `input`; `reference` is for scorers.

    Args:
        input (Any): What the candidate is run on.
        reference (Any): The expected value a scorer compares the output with; `None` when there is none.
        uid (Optional[str]): Stable unique id of the example; a `Dataset` derives one from the content when `None`.

    Examples:
        A support ticket with the queue it should be routed to:

        >>> example = Example("Refund not received for order 8812", reference="billing")
    """
    input:      Any
    reference:  Any            = field(default=None)
    uid:        Optional[str]  = field(default=None)


@dataclass(frozen=True)
class Dataset(object):
    """An ordered, immutable collection of examples with stable uids, so a run says exactly which data it used.

    Args:
        name (str): Identifies the dataset in logs and results; must not be empty.
        examples (Sequence[Example]): The examples, in run order; one without a `uid` gets a hash of its content.

    Raises:
        ValueError: If `name` is empty, there are no examples, or two examples share a uid.

    Examples:
        Two tickets, then the same tickets in another order, which is a different dataset:

        >>> tickets = Dataset("ticket-routing", [Example("Card charged twice", "billing"), Example("Login fails", "general")])
        >>> [example.uid for example in tickets]
        ['082c349c73eb', '05cf28d9c8af']
        >>> reordered = Dataset("ticket-routing", list(reversed(tickets.examples)))
        >>> tickets.fingerprint == reordered.fingerprint
        False
    """
    name:      str
    examples:  Sequence[Example]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Dataset.name must not be empty")
        if not self.examples:
            raise ValueError(f"Dataset {self.name!r} needs at least one example")
        examples = tuple(
            example if example.uid is not None else replace(example, uid=_digest([example.input, example.reference]))
            for example in self.examples
        )
        uids = [example.uid for example in examples]
        if len(set(uids)) != len(uids):
            raise ValueError(f"Dataset {self.name!r} has duplicate example uids; give repeated examples a `uid`")
        object.__setattr__(self, "examples", examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[Example]:
        return iter(self.examples)

    @property
    def fingerprint(self) -> str:
        """A hash of every example's uid, input and reference in order; any change to them, or to their order, changes it."""
        return _digest([[example.uid, example.input, example.reference] for example in self.examples])


@dataclass(frozen=True)
class Task(object):
    """What is measured: a named, versioned dataset. It says nothing about who is measured.

    Args:
        name (str): Identifies the task in logs and results; must not be empty.
        dataset (Dataset): What the candidate is run on, one trial per example, with the reference a scorer may use.
        version (str): Bump it when the dataset or its meaning change, so old and new results are not mixed.

    Raises:
        ValueError: If `name` is empty.

    Examples:
        One task, two candidates (a baseline rule and a variant), run through the same examples:

        >>> tickets = Dataset("tickets", [Example("Card was charged twice for invoice INV-2291", "billing"),
        ...                               Example("Login fails with a 500 error", "general")])
        >>> task = Task("ticket-routing", tickets)
        >>> baseline = CallableCandidate(lambda text: "general")
        >>> variant  = CallableCandidate(lambda text: "billing" if "invoice" in text.lower() else "general")
        >>> [record.outcome.output.value for record in Runner().run(baseline, task.dataset)]
        ['general', 'general']
        >>> [record.outcome.output.value for record in Runner().run(variant, task.dataset)]
        ['billing', 'general']

        Applied (illustrative): a task for an agent evaluation, run against a `velari_ai` agent candidate:

        >>> questions = Dataset("support-questions", [Example("What's the balance on ACC-10293?"), Example("Summarize ticket ORG-4471.")])
        >>> task = Task("support-agent-quality", questions, version="2")
    """
    name:     str
    dataset:  Dataset
    version:  str       = field(default="1")

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Task.name must not be empty")
