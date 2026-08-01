from enum import StrEnum, auto
from velari_core.core.types import StrEnumBase
from velari_core.core.io.types import ArtifactFormat, ArtifactKind, ArtifactProperties


class SourceType(StrEnumBase):
    DATABASE = auto()
    DOCUMENT = auto()
    CACHE = auto()
    API = auto()
    OBJECT_STORAGE = auto()


class DocType(StrEnumBase):
    GLOSSARY = auto()
    NOTES = auto()
    CONTENT = auto()
    METADATA = auto()
