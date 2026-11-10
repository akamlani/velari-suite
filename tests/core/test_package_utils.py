"""Tests for velari_core.core.utils.package_utils."""

from importlib.metadata import version

import pytest


@pytest.mark.parametrize("package, distribution", [("velari_core", "velari-core"), ("velari_ai", "velari-ai")])
def test_package_version_installed_returns_metadata_version(package, distribution):
    from velari_core.core.utils.package_utils import package_version

    assert package_version(package) == version(distribution)


def test_package_version_dotted_name_uses_top_level_package():
    from velari_core.core.utils.package_utils import package_version

    assert package_version("velari_core.core.utils") == package_version("velari_core")


def test_package_version_source_only_reads_own_pyproject(tmp_path, monkeypatch):
    from velari_core.core.utils.package_utils import package_version

    (tmp_path / "pyproject.toml").write_text('[project]\nname = "velari-fake"\nversion = "9.9.9"\n')
    pkg_dir = tmp_path / "velari_fake"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))

    assert package_version("velari_fake") == "9.9.9"


def test_package_version_unknown_package_returns_default():
    from velari_core.core.utils.package_utils import package_version

    assert package_version("velari_no_such_package") == "0.0.0"
