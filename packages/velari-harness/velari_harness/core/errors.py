# package modules
from    .schemas.types  import ReasonCode


class TrialError(Exception):
    """Raised by a candidate to declare why its trial failed, instead of leaving the Runner to classify the exception.

    Args:
        reason_code (ReasonCode): Why the trial failed.
        message (str): What went wrong, for the record's `error` text.

    Examples:
        >>> raise TrialError(ReasonCode.UNAVAILABLE, "billing API returned 503 for ACC-10293")
    """
    def __init__(self, reason_code: ReasonCode, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code
