from __future__ import annotations

from    dataclasses import dataclass, field, fields
from    typing       import Any, Dict, Self, Union

from    omegaconf import DictConfig


@dataclass
class ConfigBase:
    """Base for config dataclasses built from a raw YAML/dict mapping.

    Subclasses declare their own known fields (each needs a default, since `extra` below already
    has one — dataclass inheritance requires every field after the first defaulted one to have a
    default too) and inherit `from_config()` for free. Override `_coerce_kwargs()` only when a known
    field needs converting from its raw YAML form (e.g. a string into an enum) before construction.
    """
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, entry: Union[DictConfig, Dict[str, Any]]) -> Self:
        """Build this config from a raw YAML/dict mapping.

        Args:
            entry (Union[DictConfig, Dict[str, Any]]): Raw mapping — this class's own declared
                fields (other than `extra`) are read directly as kwargs; anything else lands in
                `extra`, for the consumer to forward on as it sees fit.

        Returns:
            Self: Ready to pass into whatever this config configures.

        Examples:
            >>> model_config = ModelConfig.from_config(cfg.model_config)
            >>> chunking_config = ChunkingConfig.from_config(cfg.retrieval.chunking)
        """
        known  = {f.name for f in fields(cls) if f.name != "extra"}
        kwargs = {str(k): v for k, v in entry.items() if k in known}
        extra  = {str(k): v for k, v in entry.items() if k not in known}
        kwargs = cls._coerce_kwargs(kwargs)
        return cls(**kwargs, extra=extra)

    @classmethod
    def _coerce_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert raw YAML values (e.g. a string into an enum) before construction; no-op by default."""
        return kwargs
