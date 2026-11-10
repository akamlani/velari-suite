import platform
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, packages_distributions, version
from importlib.util import find_spec
from pathlib import Path
from typing import Optional

# format: ('_major', '_minor', '_patch')
watermark    = dict(python=f"{sys.version_info.major}.{sys.version_info.minor}")
platform_sys = platform.system()


def _version_from_metadata(package: str) -> Optional[str]:
    """Look up the installed distribution that provides `package` and return its version.

    Args:
        package (str): Top-level import name, e.g. `velari_ai`.

    Returns:
        Optional[str]: The installed version, or None if no installed distribution provides it.
    """
    for distribution_name in packages_distributions().get(package, []):
        try:
            return version(distribution_name)
        except PackageNotFoundError:
            continue
    return None


def _version_from_pyproject(package: str) -> Optional[str]:
    """Read the version from the nearest `pyproject.toml` above `package`'s source location.

    Args:
        package (str): Top-level import name, e.g. `velari_ai`.

    Returns:
        Optional[str]: The `[project] version`, or None if the package can't be located or no parent declares one.
    """
    try:
        spec = find_spec(package)
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None

    for parent in Path(spec.origin).resolve().parents:
        pyproject_file = parent / "pyproject.toml"
        try:
            with pyproject_file.open("rb") as stream:
                declared = tomllib.load(stream).get("project", {}).get("version")
        except (OSError, tomllib.TOMLDecodeError):
            continue
        if isinstance(declared, str) and declared.strip():
            return declared.strip()
    return None


def package_version(package: str) -> str:
    """Resolve a package's version from installed metadata, else its own `pyproject.toml`, else `"0.0.0"`.

    The `pyproject.toml` fallback starts from `package`'s own source location, so each workspace
    package reports its own version, not `velari-core`'s.

    Args:
        package (str): Import name of the package; a dotted name uses its top-level package.

    Returns:
        str: The package version.

    Examples:
        >>> __version__ = package_version("velari_harness")   # in velari_harness/__init__.py: package_version(__name__)
        >>> __version__
        '0.1.0'
    """
    top_level = package.split(".")[0]
    return _version_from_metadata(top_level) or _version_from_pyproject(top_level) or "0.0.0"
