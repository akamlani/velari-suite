from __future__ import annotations
import  numpy as np
from    abc       import ABC, abstractmethod
from    omegaconf import DictConfig, OmegaConf
from    typing    import Any, Dict, Optional, Self


class ModelDriver(ABC):
    """Base class for any model driver: config-based construction, fit interface, and effective settings.

    Args:
        seed (int): Random state, for reproducible fits; ignored by deterministic models.
    """
    def __init__(self, seed: int = 42) -> None:
        self._seed   = seed
        self._fitted = False

    @classmethod
    def from_config(cls, cfg: Optional[DictConfig], seed: int = 42, **overrides: Any) -> Self:
        """Build a driver from an OmegaConf section; explicit overrides win over config values.

        Args:
            cfg (Optional[DictConfig]): Constructor arguments (e.g. `min_cluster_size`); `None` keeps the driver defaults.
            seed (int): Random state passed to the driver.
            **overrides (Any): Constructor arguments that take precedence over `cfg`.

        Returns:
            Self: A new, unfitted driver.

        Examples:
            >>> cfg = OmegaConf.create({"min_cluster_size": 3})
            >>> driver = HDBSCANDriver.from_config(cfg, seed=7, min_samples=1)
        """
        params = {} if cfg is None else OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(params, dict):
            raise TypeError(f"cfg must be a mapping of constructor arguments, got {type(params).__name__}")
        kwargs: Dict[str, Any] = {str(k): v for k, v in params.items()}
        return cls(seed=seed, **{**kwargs, **overrides})

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[Any] = None) -> Self:
        """Fit on `X` of shape (n_samples, n_features) and return `self`, so calls chain."""

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Effective settings after any small-sample clamping, plus a fit summary once fitted."""

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} not fitted — call fit() first")
