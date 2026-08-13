from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path
from typing import Any, Optional


class ArtifactKind(StrEnum):
    FILE      = auto()
    DIRECTORY = auto()
    ARCHIVE   = auto()
    UNKNOWN   = auto()


# Merged IoFromat into this
class ArtifactFormat(StrEnum):
    PDF         = auto()
    DOCX        = auto()
    JSON        = auto()
    TXT         = auto()
    MARKDOWN    = auto()
    HTML        = auto()
    YAML        = auto()
    PY          = auto()
    BLOB        = auto()
    DUCKDB      = auto()
    DICT        = auto()
    DICTCONFIG  = auto()
    DATAFRAME   = auto()
    EXCEL       = auto()
    WEB         = auto()

    @classmethod
    def from_ext(cls, extension: str) -> Optional[ArtifactFormat]:
        mapping = {
            ".pdf":  cls.PDF,
            ".json": cls.JSON,
            ".txt":  cls.TXT,
            ".md":   cls.TXT,
            ".yaml": cls.YAML,
            ".yml":  cls.YAML,
            ".py":   cls.PY,
            ".xlsx": cls.EXCEL,
            ".xls":  cls.EXCEL,
        }
        return mapping.get(extension.lower())

    @classmethod
    def from_path(cls, path: Any) -> ArtifactFormat:
        path_str = str(path)
        if isinstance(path, list) or path_str.startswith(("http://", "https://")):
            return cls.WEB
        ext = Path(path_str).suffix.lower()
        fmt = cls.from_ext(ext)
        if fmt is None:
            raise ValueError(f"Cannot infer format from extension '{ext}'. Pass fmt= explicitly.")
        return fmt


@dataclass
class ArtifactProperties:
    @dataclass
    class Location:
        is_local: bool
        is_remote: bool
        path: Optional[str] = None
        uri: Optional[str] = None

    @dataclass
    class Kind:
        exists: bool
        type: ArtifactKind

    @dataclass
    class Name:
        parent: str
        base_name: str
        extension: str
        mime_type: str

    @dataclass
    class Stats:
        size: Optional[int] = None
        disk_total: Optional[int] = None
        disk_used: Optional[int] = None
        disk_free: Optional[int] = None

    location: Location
    kind: Kind
    name: Name
    stats: Stats
