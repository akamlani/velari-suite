from    enum import StrEnum, auto


class Status(StrEnum):
    """How a harness run or one trial ended; a trial is one candidate run on one input."""
    SUCCESS            = auto()
    ERROR              = auto()


class ReasonCode(StrEnum):
    """Why a trial ended in `ERROR`, by cause and not by use case; see `Failure.REASON_RULES` for the mapping from exceptions."""
    INVALID_INPUT      = auto()
    NOT_FOUND          = auto()
    PERMISSION_DENIED  = auto()
    TIMEOUT            = auto()
    UNAVAILABLE        = auto()
    UNEXPECTED         = auto()

