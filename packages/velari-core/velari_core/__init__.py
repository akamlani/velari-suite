"""velari_core package — sets up logging from config/logging.yaml."""

import logging
import logging.config
from pathlib import Path

import yaml

from .version import __version__
from .core.utils.env_utils import read_root_dir

_ROOT_DIR = Path(read_root_dir())
_CONFIG_PATH = _ROOT_DIR / "config" / "logging.yaml"
_LOG_DIR = _ROOT_DIR / "logs"


def setup_logging(config_path: Path = _CONFIG_PATH) -> None:
    _LOG_DIR.mkdir(exist_ok=True)
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        # Remove rich handler if rich is not installed
        try:
            import rich  # noqa: F401
        except ImportError:
            root_handlers: list = cfg.get("root", {}).get("handlers", [])
            if "console_rich_handler" in root_handlers:
                root_handlers.remove("console_rich_handler")
                cfg.setdefault("root", {})["handlers"] = root_handlers
            cfg.get("handlers", {}).pop("console_rich_handler", None)
        for handler in cfg.get("handlers", {}).values():
            if "filename" in handler:
                handler["filename"] = str(_LOG_DIR / Path(handler["filename"]).name)
        logging.config.dictConfig(cfg)
    else:
        logging.basicConfig(level=logging.INFO)


setup_logging()
logger = logging.getLogger(__name__)

__all__ = ["__version__", "logger"]
