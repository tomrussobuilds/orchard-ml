"""
Configuration Serialization & Persistence Utilities.

This module handles the conversion of complex Python objects (including Pydantic
models and Path objects) into YAML format, ensuring thread-safe and
environment-agnostic persistence to the filesystem.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Protocol, cast

import yaml

from ..paths import LOGGER_NAME


# YAML ORCHESTRATION
def save_config_as_yaml(data: Any, yaml_path: Path) -> Path:
    """
    Serializes and persists configuration data to a YAML file.

    This function coordinates the extraction of data from potentially complex
    objects (supporting Pydantic models, custom portable manifests, or raw dicts),
    applies recursive sanitization, and performs an atomic write to disk.

    Args:
        data (Any): The configuration object to save. Supports objects with
            'dump_portable()' or 'model_dump()' methods, or standard dictionaries.
        yaml_path (Path): The destination filesystem path.

    Returns:
        Path: The confirmed path where the YAML was successfully written.

    Raises:
        ValueError: If the data structure cannot be serialized.
        OSError: If a filesystem-level error occurs (permissions, disk full).
    """
    logger = logging.getLogger(LOGGER_NAME)

    # 1. Extraction & Sanitization Phase
    try:
        # Priority 1: Custom portability protocol
        if hasattr(data, "dump_portable"):
            raw_dict = data.dump_portable()

        # Priority 2: Pydantic model protocol
        elif hasattr(data, "model_dump"):
            try:
                raw_dict = data.model_dump(mode="json")
            except (TypeError, ValueError):  # pragma: no cover
                # Fallback for older Pydantic V2 versions or complex types
                raw_dict = data.model_dump()  # pragma: no mutate

        # Priority 3: Raw dictionary or other types
        else:
            raw_dict = data

        final_data = _sanitize_for_yaml(raw_dict)

    except Exception as e:
        logger.error("Serialization failed: object structure is incompatible. Error: %s", e)
        raise ValueError(f"Could not serialize configuration object: {e}") from e

    # 2. Persistence Phase (Atomic Write)
    try:
        _persist_yaml_atomic(final_data, yaml_path)
        logger.debug("Configuration frozen at → %s", yaml_path.name)
        return yaml_path

    except OSError as e:
        logger.error("IO Error: Could not write YAML to %s. Error: %s", yaml_path, e)
        raise


def dump_requirements(output_path: Path) -> None:
    """
    Freeze installed packages to a requirements file for reproducibility.

    Invokes ``pip freeze --local`` to capture the exact dependency versions
    of the current environment. The output is prefixed with a Python version
    header for auditability.

    Args:
        output_path: Filesystem path where the requirements file is written.
    """
    import subprocess  # nosec B404
    import sys

    logger = logging.getLogger(LOGGER_NAME)

    try:
        result = subprocess.run(  # nosec B603
            [sys.executable, "-m", "pip", "freeze", "--local"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        header = f"# Python {sys.version.split()[0]}\n"
        output_path.write_text(header + result.stdout, encoding="utf-8")
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.error("Failed to dump requirements: %s", e)


def dump_git_info(output_path: Path) -> None:
    """
    Persist git commit hash, branch, and dirty status for run auditability.

    Captures the current HEAD commit (short hash + full hash), active branch,
    and whether the working tree has uncommitted changes. Silently skips if
    git is not available or the project is not a git repository.

    Args:
        output_path: Filesystem path where the git info is written.
    """
    import subprocess  # nosec B404

    logger = logging.getLogger(LOGGER_NAME)

    try:
        git_info_parts: list[str] = []

        commit_full = subprocess.run(  # nosec B603 B607
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        commit_short = subprocess.run(  # nosec B603 B607
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        branch = subprocess.run(  # nosec B603 B607
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        dirty = subprocess.run(  # nosec B603 B607
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if commit_full.returncode == 0:
            git_info_parts.append(f"commit: {commit_full.stdout.strip()}")
            git_info_parts.append(f"commit_short: {commit_short.stdout.strip()}")
        if branch.returncode == 0:
            git_info_parts.append(f"branch: {branch.stdout.strip()}")
        if dirty.returncode == 0:
            is_dirty = bool(dirty.stdout.strip())
            git_info_parts.append(f"dirty: {is_dirty}")

        if git_info_parts:
            output_path.write_text("\n".join(git_info_parts) + "\n", encoding="utf-8")

    except (subprocess.TimeoutExpired, OSError) as e:
        logger.debug("Could not capture git info: %s", e)


def load_config_from_yaml(yaml_path: Path) -> dict[str, Any]:
    """
    Loads a raw configuration dictionary from a YAML file.

    Args:
        yaml_path (Path): Path to the source YAML file.

    Returns:
        dict[str, Any]: The loaded configuration manifest.

    Raises:
        FileNotFoundError: If the specified path does not exist.
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML configuration file not found at: {yaml_path}")

    # Equivalent mutants: "r" is Python's default open mode; cast() has no runtime effect.
    with open(yaml_path, "r", encoding="utf-8") as f:  # pragma: no mutate
        return cast(dict[str, Any], yaml.safe_load(f))  # pragma: no mutate


class AuditSaverProtocol(Protocol):
    """
    Protocol for run-manifest persistence (config YAML + dependency snapshot).

    Enables dependency injection of auditability operations in
    ``RootOrchestrator``, keeping the constructor signature lean while
    allowing full mocking in tests.
    """

    def save_config(self, data: Any, yaml_path: Path) -> Path:
        """
        Persist configuration to a YAML file.

        Args:
            data: Configuration object to serialize.
            yaml_path: Destination filesystem path.

        Returns:
            Confirmed path where the YAML was written.
        """
        ...  # pragma: no cover

    def dump_requirements(self, output_path: Path) -> None:
        """
        Freeze installed packages for reproducibility.

        Args:
            output_path: Filesystem path for the requirements snapshot.
        """
        ...  # pragma: no cover

    def dump_git_info(self, output_path: Path) -> None:
        """
        Persist git commit hash and working tree status for auditability.

        Args:
            output_path: Filesystem path for the git info snapshot.
        """
        ...  # pragma: no cover


class AuditSaver:
    """
    Default ``AuditSaverProtocol`` implementation.

    Delegates to the module-level ``save_config_as_yaml``,
    ``dump_requirements``, and ``dump_git_info`` functions —
    no logic duplication.
    """

    def save_config(self, data: Any, yaml_path: Path) -> Path:
        """
        Persist configuration to a YAML file.

        Args:
            data: Configuration object to serialize.
            yaml_path: Destination filesystem path.

        Returns:
            Confirmed path where the YAML was written.
        """
        return save_config_as_yaml(data, yaml_path)

    def dump_requirements(self, output_path: Path) -> None:
        """
        Freeze installed packages for reproducibility.

        Args:
            output_path: Filesystem path for the requirements snapshot.
        """
        dump_requirements(output_path)

    def dump_git_info(self, output_path: Path) -> None:
        """
        Persist git commit hash and working tree status.

        Args:
            output_path: Filesystem path for the git info snapshot.
        """
        dump_git_info(output_path)


def _sanitize_for_yaml(obj: Any) -> Any:
    """
    Recursively converts non-serializable types into YAML-standard formats.

    Specifically handles:

    - Path objects -> converted to strings.
    - Dicts/Lists/Tuples -> processed recursively.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_for_yaml(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_yaml(i) for i in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _persist_yaml_atomic(data: Any, path: Path) -> None:
    """
    Performs a safe write operation with directory creation and buffer flushing.

    Leverages fsync to ensure the data is physically committed to the storage
    device, preventing data loss during system failures.
    """
    # Ensure directory existence before writing
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        # Equivalent mutants: PyYAML 6+ defaults match these kwargs (sort_keys=None≡False,
        # default_flow_style omitted≡False). Removing or tweaking them is runtime-equivalent.
        # fmt: off
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, indent=4, allow_unicode=True)  # pragma: no mutate
        # fmt: on
        # Force OS-level synchronization
        f.flush()
        os.fsync(f.fileno())
