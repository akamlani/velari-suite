import  json
import  logging
from    typing import Union
from    pathlib import Path

logger = logging.getLogger(__name__)

def write_text(
    text:      str,
    path:      Union[str, Path],
    encoding:  str = "utf-8",
    append:    bool = False,
) -> None:
    """Write text to a file."""
    try:
        mode = "a" if append else "w"
        with open(path, mode, encoding=encoding) as f:
            f.write(text)
    except Exception as e:
        logger.error(f"Error writing to file {path}: {e}")


def write_json(
    data:      dict,
    path:      Union[str, Path],
    encoding:  str = "utf-8",
    append:    bool = False,
) -> None:
    """Write JSON data to a file."""
    try:
        mode = "a" if append else "w"
        with open(path, mode, encoding=encoding) as f:
            json.dump(data, f)
    except Exception as e:
        logger.error(f"Error writing JSON to file {path}: {e}")
