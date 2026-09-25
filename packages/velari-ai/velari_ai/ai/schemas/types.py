from    enum import StrEnum, auto


class Priority(StrEnum):
    LOW        = auto()
    MEDIUM     = auto()
    HIGH       = auto()
    URGENT     = auto()


class Sentiment(StrEnum):
    POSITIVE   = auto()
    NEUTRAL    = auto()
    NEGATIVE   = auto()
