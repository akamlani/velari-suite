"""velari_core package — configures logging from the `logging` entry in config/config.yaml."""

import  logging
import  logging.config
from    importlib.util import find_spec
from    pathlib import Path
from    typing import Any, Dict, Optional
from    hydra.errors import MissingConfigException
from    omegaconf import OmegaConf
# package modules
from    .version import __version__
from    .core import read_root_dir
from    .core.io.partition.hydra import read_hydra, read_hydra_defaults


def _in_notebook() -> bool:
    """Return True when running under Jupyter/IPython."""
    try:
        from IPython.core.getipython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def setup_logging(config_path: Optional[Path] = None) -> None:
    """Configure the root logger from a Hydra-managed logging config; skipped in notebooks unless `logging.notebook` is true.

    Args:
        config_path (Optional[Path]): Logging YAML to load; defaults to the file named by `logging.name` in config/config.yaml.

    Examples:
        >>> setup_logging()  # called automatically on package import
        >>> logger = logging.getLogger("velari.ai.retrieval")
        >>> logger.info("Indexed 1,204 support-ticket embeddings.")
    """
    root_dir   = Path(read_root_dir())
    config_dir = root_dir / "config"
    try:
        main_cfg = read_hydra_defaults(str(config_dir), "config")
    except MissingConfigException:
        main_cfg = OmegaConf.create()
    if _in_notebook() and not OmegaConf.select(main_cfg, "logging.notebook", default=False):
        return

    try:
        config_path = config_path or config_dir / OmegaConf.select(main_cfg, "logging.name", default="logging.yaml")
        container   = OmegaConf.to_container(read_hydra(str(config_path)), resolve=True)
    except FileNotFoundError:
        logging.basicConfig(level=logging.INFO)
        return

    cfg: Dict[str, Any] = {str(k): v for k, v in container.items()} if isinstance(container, dict) else {}
    # RichHandler is an optional dependency — drop it from the config rather than fail dictConfig.
    if find_spec("rich") is None:
        cfg.get("handlers", {}).pop("console_rich_handler", None)
        cfg.get("root", {})["handlers"] = [h for h in cfg.get("root", {}).get("handlers", []) if h != "console_rich_handler"]
    for handler in cfg.get("handlers", {}).values():
        if "filename" in handler:
            log_file = (root_dir / handler["filename"]).resolve()
            log_file.parent.mkdir(parents=True, exist_ok=True)
            handler["filename"] = str(log_file)
    logging.config.dictConfig(cfg)


setup_logging()
# TODO: Consider adding a mechanism to namespace loggers by package, root, etc...
logger = logging.getLogger(__name__)

__all__ = ["__version__", "logger"]
