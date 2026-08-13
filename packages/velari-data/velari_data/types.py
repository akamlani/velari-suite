import time
from enum import StrEnum, auto
from dataclasses import dataclass, field
from typing import Any
# package modules
from velari_core.core.types import StrEnumBase
from velari_core.core.io.types import ArtifactFormat, ArtifactKind, ArtifactProperties


class SourceType(StrEnumBase):
    DATABASE        = auto()
    DOCUMENT        = auto()
    CACHE           = auto()
    API             = auto()
    OBJECT_STORAGE  = auto()

class DocType(StrEnumBase):
    GLOSSARY        = auto()
    NOTES           = auto()
    CONTENT         = auto()
    METADATA        = auto()

@dataclass
class TraceEvent:
    step:       int
    event_type: str
    payload:    Any
    timestamp:  float = field(default_factory=time.time)
