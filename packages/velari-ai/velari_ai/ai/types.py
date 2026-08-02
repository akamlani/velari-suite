
from enum import StrEnum, auto
from typing import Any, Optional, TypedDict, Literal

class ProviderMode(StrEnum):
    CHAT                    = auto()
    RESPONSES               = auto()


class ProviderName(StrEnum):
    OPENAI                  = auto()
    HUGGINGFACE             = auto()
    ANTHROPIC               = auto()
    SENTENCE_TRANSFORMERS   = auto()

class Message(TypedDict):
    role:    Literal["user", "assistant"]
    content: str
