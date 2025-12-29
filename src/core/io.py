"""
Input/Output & Persistence Utilities.

This module manages the pipeline's interaction with the filesystem, handling 
configuration serialization (YAML), model checkpoint restoration, and dataset 
integrity verification via MD5 checksums and schema validation.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import logging
import yaml
import hashlib
from pathlib import Path
from typing import Any, Dict

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import numpy as np
import torch

# =========================================================================== #
#                                  I/O Utilities                              #
# =========================================================================== #

logger = logging.getLogger("medmnist_pipeline")

def save_config_as_yaml(data: Dict[str, Any], yaml_path: Path) -> Path:
    """
    Serializes a configuration dictionary to a YAML file.
    Converts complex types (like Path) to strings for YAML compatibility.
    """
    try:
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        def stringify_paths(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: stringify_paths(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [stringify_paths(i) for i in obj]
            if isinstance(obj, Path):
                return str(obj)
            return obj

        cleaned_data = stringify_paths(data)

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


def load_model_weights(
        model: torch.nn.Module, 
        path: Path, 
        device: torch.device
) -> None:
    """
    Restores model state from a checkpoint using secure weight-only loading.
    
    Args:
        model (torch.nn.Module): The model instance to populate.
        path (Path): Filesystem path to the checkpoint file.
        device (torch.device): Target device for mapping the tensors.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {path}")
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)