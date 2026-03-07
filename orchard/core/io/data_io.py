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

_REQUIRED_NPZ_KEYS: frozenset[str] = frozenset(
    {
        "train_images",
        "train_labels",
        "val_images",
        "val_labels",
        "test_images",
        "test_labels",
    }
)
_MD5_CHUNK_SIZE: int = 8192  # pragma: no mutate


# DATA VERIFICATION
def validate_npz_keys(data: np.lib.npyio.NpzFile) -> None:
    """
    Validates that the loaded NPZ dataset contains all required dataset keys.

    Args:
        data (np.lib.npyio.NpzFile): The loaded NPZ file object.

    Raises:
        OrchardDatasetError: If any required key (images/labels) is missing.
    """
    missing = _REQUIRED_NPZ_KEYS - set(data.files)
    if missing:
        found = list(data.files)
        raise OrchardDatasetError(
            f"NPZ archive is corrupted or invalid. Missing keys: {missing} | Found keys: {found}"
        )


def md5_checksum(path: Path, chunk_size: int = _MD5_CHUNK_SIZE) -> str:
    """
    Calculates the MD5 checksum of a file using buffered reading.

    Args:
        path (Path): Path to the file to verify.
        chunk_size (int): Read buffer size in bytes.

    Returns:
        str: The calculated hexadecimal MD5 hash.
    """
    hash_md5 = hashlib.md5(usedforsecurity=False)  # pragma: no mutate
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):  # pragma: no mutate
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
