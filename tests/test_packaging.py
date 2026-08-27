"""
Test Suite for Package Data Declarations.

Guards against shipping wheels that omit non-Python runtime assets:
every data file living inside the ``orchard`` package must be matched
by a ``[tool.setuptools.package-data]`` glob, otherwise the installed
distribution raises ``FileNotFoundError`` at import time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import orchard

PACKAGE_ROOT = Path(orchard.__file__).parent
PYPROJECT_PATH = PACKAGE_ROOT.parent / "pyproject.toml"

DATA_SUFFIXES = (".yaml", ".yml", ".json")


def _package_data_globs() -> list[str]:
    """
    Read the declared package-data patterns for the ``orchard`` package.
    """
    import tomllib

    with PYPROJECT_PATH.open("rb") as handle:
        config = tomllib.load(handle)
    return list(config["tool"]["setuptools"]["package-data"]["orchard"])


def _shipped_data_files() -> set[Path]:
    """
    Collect every runtime data file under the ``orchard`` package tree.
    """
    return {
        path
        for suffix in DATA_SUFFIXES
        for path in PACKAGE_ROOT.rglob(f"*{suffix}")
        if "__pycache__" not in path.parts
    }


@pytest.mark.unit
def test_medical_registry_yaml_is_present() -> None:
    """
    The MedMNIST metadata YAML ships inside the installed package tree.
    """
    yaml_path = PACKAGE_ROOT / "core" / "metadata" / "domains" / "classification" / "medical.yaml"
    assert yaml_path.is_file(), f"missing runtime asset: {yaml_path}"


@pytest.mark.unit
@pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib requires Python 3.11+")
def test_all_package_data_files_are_declared() -> None:
    """
    Every data file under ``orchard`` matches a declared package-data glob.
    """
    if not PYPROJECT_PATH.is_file():
        pytest.skip("pyproject.toml unavailable (running against an installed distribution)")

    globs = _package_data_globs()
    declared = {path for pattern in globs for path in PACKAGE_ROOT.glob(pattern)}
    undeclared = sorted(
        str(path.relative_to(PACKAGE_ROOT)) for path in _shipped_data_files() - declared
    )

    assert not undeclared, (
        f"data files not covered by [tool.setuptools.package-data] {globs}: {undeclared} "
        "-- they would be dropped from the built wheel"
    )
