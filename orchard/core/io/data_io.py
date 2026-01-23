"""
Data Integrity & Dataset I/O Utilities.

Provides tools for verifying file integrity via checksums and validating
the structure of NPZ dataset archives.
"""

# Standard Imports
import hashlib
from pathlib import Path

# Third-Party Imports
import numpy as np


# DATA VERIFICATION
def validate_npz_keys(data: np.lib.npyio.NpzFile) -> None:
    """
    Validates that the loaded NPZ dataset contains all required MedMNIST keys.

    Args:
        data (np.lib.npyio.NpzFile): The loaded NPZ file object.

    Raises:
        ValueError: If any required key (images/labels) is missing.
    """
    required_keys = {
        "train_images",
        "train_labels",
        "val_images",
        "val_labels",
        "test_images",
        "test_labels",
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
    hash_md5 = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as f:
        # 8192 byte chunk size for memory efficiency
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
