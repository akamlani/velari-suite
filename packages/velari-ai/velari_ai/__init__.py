import logging

from velari_core import logger as _  # noqa: F401  ensures velari-core's setup_logging() has run
from .version import __version__

logger = logging.getLogger(__name__)

__all__ = ["__version__", "logger"]
