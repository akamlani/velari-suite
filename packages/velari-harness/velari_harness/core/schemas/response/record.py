from __future__ import annotations
from    dataclasses  import dataclass, field
from    typing       import Any, ClassVar, Dict, Optional, Type
from    pydantic     import BaseModel, Field, computed_field
# package modules
from    ...errors    import TrialError
from    ..types      import ReasonCode, Status


@dataclass(frozen=True)
class Example(object):
    """One thing to measure: an input for the candidate and, optionally, the expected value for a scorer.

    The candidate sees only `input`; `reference` is for scorers.

    Args:
        input (Any): What the candidate is run on.
        reference (Any): The expected value a scorer compares the output with; `None` when there is none.

    Examples:
        A support ticket with the queue it should be routed to:

        >>> example = Example("Refund not received for order 8812", reference="billing")
    """
    input:     Any
    reference: Any  = field(default=None)


class Failure(BaseModel):
    """The first failure in a piece of work: the `ExceptionType: message` text, and why it failed."""
    # exception type to reason code, first match wins, so the OSError subclasses come before OSError
    REASON_RULES: ClassVar[Dict[Type[Exception], ReasonCode]] = {
        TimeoutError:       ReasonCode.TIMEOUT,
        PermissionError:    ReasonCode.PERMISSION_DENIED,
        LookupError:        ReasonCode.NOT_FOUND,
        FileNotFoundError:  ReasonCode.NOT_FOUND,
        ConnectionError:    ReasonCode.UNAVAILABLE,
        OSError:            ReasonCode.UNAVAILABLE,
        ValueError:         ReasonCode.INVALID_INPUT,
        TypeError:          ReasonCode.INVALID_INPUT,
    }

    error:       str
    reason_code: ReasonCode

    @classmethod
    def of(cls, exc: Exception) -> Failure:
        """Describe `exc`; a `TrialError` declares its own reason code, anything else is matched against `REASON_RULES`.

        Args:
            exc (Exception): What a candidate, setup or teardown raised.

        Returns:
            Failure: The `ExceptionType: message` text and the reason code; `UNEXPECTED` when no rule matches.
        """
        reason_code = exc.reason_code if isinstance(exc, TrialError) else next(
            (code for kind, code in cls.REASON_RULES.items() if isinstance(exc, kind)), ReasonCode.UNEXPECTED)
        return cls(error=f"{type(exc).__name__}: {exc}", reason_code=reason_code)


class Output(BaseModel):
    """What a candidate returns when it has more to report than a value: the value, and how it was produced.

    Args:
        value (Any): What the candidate produced for the example.
        usage (Dict[str, float]): Counts and costs of producing it, such as tokens or seconds.
        artifacts (Dict[str, str]): Files it wrote, by name to path; recorded, not copied.

    Examples:
        An agent candidate that reports its token usage with the answer:

        >>> Output(value="enterprise", usage={"input_tokens": 812, "output_tokens": 96})
    """
    value:      Any # what the candidate returned
    usage:      Dict[str, float]  = Field(default_factory=dict)
    artifacts:  Dict[str, str]    = Field(default_factory=dict)


class Outcome(BaseModel):
    """How one trial ended: its `output`, or the `failure` of its first failing step, and how long it took.

    `output` is `None` when the work did not complete; a failed teardown still keeps it.
    """
    output:       Optional[Output]   = None
    failure:      Optional[Failure]  = None
    duration_sec: float              = Field(ge=0)

    @computed_field
    @property
    def status(self) -> Status:
        """`SUCCESS`, or `ERROR` when setup, the work or teardown raised."""
        return Status.SUCCESS if self.failure is None else Status.ERROR


class Score(BaseModel):
    """One scorer's verdict on one output; `value` is `None` when the scorer could not judge, which is not a zero.

    Args:
        value (Optional[float]): How good the output is.
        passed (Optional[bool]): Whether it met the scorer's threshold, if it has one.
        label (Optional[str]): A short name for the verdict, such as `match`.
        explanation (Optional[str]): Why the scorer gave this verdict, such as an LLM judge's reasoning.
        failure (Optional[Failure]): Why the scorer could not judge, when it raised.
        metadata (Dict[str, Any]): Anything else to record: the scorer's name or version, a judge model, per-part metrics.

    Examples:
        A judge's verdict on a retrieved answer, with its reasoning:

        >>> Score(value=0.5, passed=False, label="partial", explanation="cites 1 of 2 sources", metadata={"judge": "relevance-v2"})
    """
    value:        Optional[float]    = None # how good it was
    passed:       Optional[bool]     = None # whether it met the scorer's threshold
    label:        Optional[str]      = None # short name for the verdict
    explanation:  Optional[str]      = None # why it got this verdict, such as an LLM judge's reasoning
    failure:      Optional[Failure]  = None # why the scorer could not judge, if it raised
    metadata:     Dict[str, Any]     = Field(default_factory=dict)


class TrialRecord(BaseModel):
    """One trial: the example and its position, how it ended (`outcome`), and its `scores`.

    `scores` holds each scorer's `Score` by name (its `failure` is set when it raised), and is empty when the trial failed.
    """
    index:    int
    example:  Example
    outcome:  Outcome
    scores:   Dict[str, Score]            = Field(default_factory=dict)
