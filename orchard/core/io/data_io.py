"""
Data Integrity & Dataset I/O Utilities.

Provides tools for verifying file integrity via checksums and validating
the structure of NPZ dataset archives.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ...exceptions import OrchardDatasetError


# DATA VERIFICATION
def validate_npz_keys(data: np.lib.npyio.NpzFile) -> None:
    """
    Validates that the loaded NPZ dataset contains all required dataset keys.

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
        raise OrchardDatasetError(
            f"NPZ archive is corrupted or invalid. Missing keys: {missing} | Found keys: {found}"
        )


def md5_checksum(path: Path, chunk_size: int = 8192) -> str:
    """
    Calculates the MD5 checksum of a file using buffered reading.

    Args:
        path (Path): Path to the file to verify.
        chunk_size (int): Read buffer size in bytes.

    Returns:
        str: The calculated hexadecimal MD5 hash.
    """
    hash_md5 = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
