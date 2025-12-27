"""
Input/Output Utilities Module

This module provides low-level file handling utilities, including YAML 
serialization for configurations and validation for dataset archives.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import logging
import yaml
import hashlib
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import numpy as np

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
if TYPE_CHECKING:
    from .config import Config

# =========================================================================== #
#                                  I/O UTILITIES                              #
# =========================================================================== #

logger = logging.getLogger("medmnist_pipeline")

def save_config_as_yaml(config: "Config", yaml_path: Path) -> Path:
    """
    Serializes a Pydantic configuration object to a clean YAML file.

    Uses Pydantic's json-mode dumping to ensure complex types (Paths, Enums, 
    Tuples) are converted to standard primitives, avoiding Python-specific 
    tags in the output.

    Args:
        config (Config): The validated configuration object.
        yaml_path (Path): Destination path for the YAML file.

    Returns:
        Path: The path where the configuration was saved.
    """
    try:
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Converts to standard types: Path -> str, Tuple -> list
        cleaned_data = config.model_dump(mode='json')
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                cleaned_data, 
                f, 
                default_flow_style=False, 
                sort_keys=False,
                indent=4,
                allow_unicode=True
            )

        logger.info(f"Configuration frozen successfully at → {yaml_path.name}")
        return yaml_path
    except Exception as e:
        logger.error(f"Critical failure during YAML serialization: {e}")
        raise


def load_config_from_yaml(yaml_path: Path) -> Dict[str, Any]:
    """
    Loads a configuration dictionary from a YAML file.

    Args:
        yaml_path (Path): Path to the source YAML file.

    Returns:
        Dict[str, Any]: The raw configuration dictionary.

    Raises:
        FileNotFoundError: If the specified path does not exist.
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML configuration file not found at: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def validate_npz_keys(data: np.lib.npyio.NpzFile) -> None:
    """
    Validates that the loaded NPZ dataset contains all required MedMNIST keys.

    Args:
        data (np.lib.npyio.NpzFile): The loaded NPZ file object.

    Raises:
        ValueError: If any required key (images/labels) is missing.
    """
    required_keys = {
        "train_images", "train_labels",
        "val_images", "val_labels",
        "test_images", "test_labels",
    }

    missing = required_keys - set(data.files)
    if missing:
        found = list(data.files)
        raise ValueError(
            f"NPZ archive is corrupted or invalid. Missing keys: {missing}"
            f" | Found keys: {found}"
        )


def md5_checksum(path: Path) -> str:
    """
    Calculates the MD5 checksum of a file using buffered reading.

    Args:
        path (Path): Path to the file to verify.

    Returns:
        str: The calculated hexadecimal MD5 hash.
    """
    hash_md5 = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()