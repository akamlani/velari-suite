from    dataclasses  import dataclass, field
from    typing       import Any


@dataclass(frozen=True)
class Case(object):
    """One thing to measure: an input for the candidate and, optionally, the expected value for a scorer.

    The candidate sees only `input`; `reference` is for scorers.

    Args:
        input (Any): What the candidate is run on.
        reference (Any): The expected value a scorer compares the output with; `None` when there is none.

    Examples:
        A support ticket with the queue it should be routed to:

        >>> case = Case("Refund not received for order 8812", reference="billing")
    """
    input:     Any
    reference: Any  = field(default=None)
