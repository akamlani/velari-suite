from __future__ import annotations
import  logging
from    dataclasses  import dataclass
from    omegaconf    import DictConfig, OmegaConf
from    typing       import Any, Dict, Optional, Self
# velari-core
from    velari_core.core.experiment import ConfigOmegaConf

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HarnessConfig(object):
    """Settings every harness shares; category harnesses subclass it to add their own fields.

    The seed is not a harness setting: it belongs to the `Environment`. The config is immutable, so a subclass
    must also be a `@dataclass(frozen=True)`.

    Args:
        name (str): Identifies the harness in logs and results; must be a non-empty string.
    """
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError(f"HarnessConfig.name must be a str, got {type(self.name).__name__}")
        if not self.name:
            raise ValueError("HarnessConfig.name must not be empty")

    @classmethod
    def from_config(cls, cfg: Optional[DictConfig], defaults: Optional[Dict[str, Any]] = None, **overrides: Any) -> Self:
        """Build a config from an OmegaConf section; precedence is overrides, then `cfg`, then `defaults`.

        Args:
            cfg (Optional[DictConfig]): Config fields (e.g. `name`); `None` keeps the defaults.
            defaults (Optional[Dict[str, Any]]): Values used only when neither `cfg` nor `overrides` sets them.
            **overrides (Any): Explicit field values that take precedence over `cfg`.

        Returns:
            Self: A validated config.

        Raises:
            TypeError: If `cfg` is not a mapping, or sets a field the config does not have.
            ValueError: If the resulting `name` is empty.

        Examples:
            >>> cfg = OmegaConf.create({"name": "ticket-triage"})
            >>> config = HarnessConfig.from_config(cfg, name="ticket-triage-v2")
            >>> config.name
            'ticket-triage-v2'
        """
        ConfigOmegaConf().resolve()
        params = {} if cfg is None else OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(params, dict):
            raise TypeError(f"cfg must be a mapping of config fields, got {type(params).__name__}")
        fields: Dict[str, Any] = {str(k): v for k, v in params.items()}
        return cls(**{**(defaults or {}), **fields, **overrides})
