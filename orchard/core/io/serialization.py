"""
Configuration Serialization & Persistence Utilities.

This module handles the conversion of complex Python objects (including Pydantic
models and Path objects) into YAML format, ensuring thread-safe and
environment-agnostic persistence to the filesystem.
"""

# Standard Imports
import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml

# Internal Imports
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
            except Exception:
                # Fallback for older Pydantic V2 versions or complex types
                raw_dict = data.model_dump()

        # Priority 3: Raw dictionary or other types
        else:
            raw_dict = data

        final_data = _sanitize_for_yaml(raw_dict)

    except Exception as e:
        logger.error(f"Serialization failed: object structure is incompatible. Error: {e}")
        raise ValueError(f"Could not serialize configuration object: {e}") from e

    # 2. Persistence Phase (Atomic Write)
    try:
        _persist_yaml_atomic(final_data, yaml_path)
        logger.info(f"Configuration frozen successfully at → {yaml_path.name}")
        return yaml_path

    except (OSError, PermissionError) as e:
        logger.error(f"IO Error: Could not write YAML to {yaml_path}. Error: {e}")
        raise


def load_config_from_yaml(yaml_path: Path) -> Dict[str, Any]:
    """
    Loads a raw configuration dictionary from a YAML file.

    Args:
        yaml_path (Path): Path to the source YAML file.

    Returns:
        Dict[str, Any]: The loaded configuration manifest.

    Raises:
        FileNotFoundError: If the specified path does not exist.
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML configuration file not found at: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, indent=4, allow_unicode=True)
        # Force OS-level synchronization
        f.flush()
        os.fsync(f.fileno())
