import sys
import platform
from pathlib import Path
import importlib.metadata
import tomllib

module_name = (__package__ or __name__).split(".")[0]
# format: ('_major', '_minor', '_patch')
watermark = dict(python=f"{sys.version_info.major}.{sys.version_info.minor}")
platform_sys = platform.system()


def _read_version_from_installed_metadata(module_name: str) -> str | None:
    distribution_names = importlib.metadata.packages_distributions().get(module_name, [])
    for distribution_name in distribution_names:
        try:
            return importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _read_version_from_pyproject() -> str | None:
    for parent in Path(__file__).resolve().parents:
        pyproject_file = parent / "pyproject.toml"
        if not pyproject_file.is_file():
            continue
        try:
            with pyproject_file.open("rb") as stream:
                pyproject = tomllib.load(stream)
        except OSError:
            continue
        project = pyproject.get("project", {})
        version = project.get("version")
        if isinstance(version, str) and version.strip():
            return version.strip()
    return None


__version__ = _read_version_from_installed_metadata(module_name) or _read_version_from_pyproject() or "0.0.0"
