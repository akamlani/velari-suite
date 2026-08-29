from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict
from omegaconf import DictConfig, OmegaConf
from typing import ClassVar


class ServiceSettings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        case_sensitive=False,  # env var names matched case-insensitively
        arbitrary_types_allowed=True,
        extra="forbid",  # reject any unrecognised fields
    )

    cfg: DictConfig

    @classmethod
    def from_yaml(cls, path: str) -> ServiceSettings:
        loaded = OmegaConf.load(path)
        if not isinstance(loaded, DictConfig):
            raise TypeError(f"Expected a mapping at the root of {path!r}, got {type(loaded).__name__}")
        return cls(cfg=loaded)
