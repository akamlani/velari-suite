from __future__ import annotations
from    typing    import Any, ClassVar, Dict, Optional, Type
from    pydantic  import BaseModel, Field, computed_field
# package modules
from    ...errors       import TrialError
from    ..request.case  import Case
from    ..types         import ReasonCode, Status


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


class Outcome(BaseModel):
    """How one trial ended: its `output`, or the `failure` of its first failing step, and how long it took.

    `output` is `None` when the work did not complete; a failed teardown still keeps it.
    """
    output:       Any                = None
    failure:      Optional[Failure]  = None
    duration_sec: float              = Field(ge=0)

    @computed_field
    @property
    def status(self) -> Status:
        """`SUCCESS`, or `ERROR` when setup, the work or teardown raised."""
        return Status.SUCCESS if self.failure is None else Status.ERROR


class TrialRecord(BaseModel):
    """One trial: the case and its position, how it ended (`outcome`), and its `scores`.

    `scores` is `None` for a scorer that raised, and empty when the trial failed.
    """
    index:    int
    case:     Case
    outcome:  Outcome
    scores:   Dict[str, Optional[float]]  = Field(default_factory=dict)
