import  logging
from    velari_core.core.utils.package_utils import package_version

__version__ = package_version(__name__)
logger = logging.getLogger(__name__)

__all__ = ["__version__", "logger"]
