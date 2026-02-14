"""
Dataset Fetching and Loading Module

This module handles the physical retrieval of datasets (MedMNIST, Galaxy10),
including robust download logic, MD5 verification, and metadata preparation
for lazy loading. Automatically detects dataset type and applies appropriate
conversion pipeline.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

from orchard.core import DatasetMetadata, md5_checksum, validate_npz_keys
from orchard.core.paths import LOGGER_NAME


# DATA CONTAINERS
@dataclass(frozen=True)
class DatasetData:
    """
    Metadata container for dataset (MedMNIST, Galaxy10, etc.).
    Stores path and format info instead of raw arrays to save RAM.
    """

    path: Path
    name: str
    is_rgb: bool
    num_classes: int


logger = logging.getLogger(LOGGER_NAME)


def _retry_delay(exc: Exception, base_delay: float, attempt: int) -> float:
    """Compute retry delay with quadratic backoff for 429 responses."""
    if hasattr(exc, "response") and exc.response is not None and exc.response.status_code == 429:
        delay = base_delay * (attempt**2)
        logger.warning(f"Rate limited (429). Waiting {delay}s before retrying...")
        return delay
    return base_delay


# FETCHING LOGIC
def ensure_dataset_npz(
    metadata: DatasetMetadata,
    retries: int = 5,
    delay: float = 5.0,
) -> Path:
    """
    Downloads dataset NPZ file robustly with retries and MD5 validation.

    Automatically detects dataset type (MedMNIST vs Galaxy10) and applies
    appropriate download/conversion pipeline.

    Args:
        metadata (DatasetMetadata): Metadata containing URL, MD5, name and target path.
        retries (int): Max number of download attempts.
        delay (float): Delay (seconds) between retries.

    Returns:
        Path: Path to the successfully validated .npz file.
    """
    # Check if this is Galaxy10 (requires HDF5 download and conversion)
    if metadata.name == "galaxy10":
        from .galaxy10_converter import ensure_galaxy10_npz

        return ensure_galaxy10_npz(metadata)

    # Standard NPZ download for MedMNIST datasets
    target_npz = metadata.path

    # 1. Validation of existing file
    if _is_valid_npz(target_npz, metadata.md5_checksum):
        logger.info(f"Valid dataset '{metadata.name}' found at: {target_npz}")
        return target_npz

    # 2. Cleanup corrupted file
    if target_npz.exists():
        logger.warning(f"Corrupted dataset found, deleting: {target_npz}")
        target_npz.unlink()

    # 3. Download logic with retries
    logger.info(f"Downloading {metadata.name} from {metadata.url}")
    target_npz.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_npz.with_suffix(".tmp")

    for attempt in range(1, retries + 1):
        try:
            _stream_download(metadata.url, tmp_path)

            if not _is_valid_npz(tmp_path, metadata.md5_checksum):
                actual_md5 = md5_checksum(tmp_path)
                logger.error(f"MD5 mismatch: expected {metadata.md5_checksum}, got {actual_md5}")
                raise ValueError("Downloaded file failed MD5 or header validation")

            # Atomic move
            tmp_path.replace(target_npz)
            logger.info(f"Successfully downloaded and verified: {metadata.name}")
            return target_npz

        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()

            if attempt == retries:
                logger.error(f"Download failed after {retries} attempts")
                raise RuntimeError(f"Could not download {metadata.name}") from e

            actual_delay = _retry_delay(e, delay, attempt)
            logger.warning(
                f"Attempt {attempt}/{retries} failed: {e}. Retrying in {actual_delay}s..."
            )
            time.sleep(actual_delay)

    raise RuntimeError("Unexpected error in dataset download logic.")  # pragma: no cover


# LOADING INTERFACE
def load_dataset(metadata: DatasetMetadata) -> DatasetData:
    """
    Ensures the dataset is present and returns its metadata container.

    Handles both MedMNIST (direct NPZ download) and Galaxy10 (HDF5 conversion).
    """
    path = ensure_dataset_npz(metadata)

    with np.load(path) as data:
        validate_npz_keys(data)

        train_shape = data["train_images"].shape
        is_rgb = len(train_shape) == 4 and train_shape[-1] == 3

        num_classes = len(np.unique(data["train_labels"]))

        return DatasetData(path=path, name=metadata.name, is_rgb=is_rgb, num_classes=num_classes)


def load_dataset_health_check(metadata: DatasetMetadata, chunk_size: int = 100) -> DatasetData:
    """
    Loads a small "chunk" of data (e.g., the first 100 images and labels)
    for an initial health check, while retaining the download and verification logic.

    Args:
        metadata (DatasetMetadata): Metadata containing URL, MD5, name, and path for the dataset.
        chunk_size (int): Number of samples to load for the health check.

    Returns:
        DatasetData: Metadata of the dataset, including info about the loaded data.
    """
    path = ensure_dataset_npz(metadata)

    with np.load(path) as data:
        validate_npz_keys(data)

        images_chunk = data["train_images"][:chunk_size]
        labels_chunk = data["train_labels"][:chunk_size]

        is_rgb = images_chunk.ndim == 4 and images_chunk.shape[-1] == 3

        num_classes = len(np.unique(labels_chunk))

        return DatasetData(path=path, name=metadata.name, is_rgb=is_rgb, num_classes=num_classes)


# PRIVATE HELPERS
def _is_valid_npz(path: Path, expected_md5: str) -> bool:
    """Checks file existence, header (ZIP/NPZ), and MD5 checksum."""
    if not path.exists():
        return False
    try:
        # Check for ZIP header (NPZ files are ZIP archives)
        with open(path, "rb") as f:
            if f.read(2) != b"PK":
                return False
    except IOError:
        return False

    return md5_checksum(path) == expected_md5


def _stream_download(url: str, tmp_path: Path):
    """Executes the streaming GET request and writes to a temporary file."""
    headers = {
        "User-Agent": "Wget/1.0",
        "Accept": "application/octet-stream",
        "Accept-Encoding": "identity",
    }

    with requests.get(url, headers=headers, timeout=60, stream=True, allow_redirects=True) as r:
        r.raise_for_status()

        content_type = r.headers.get("Content-Type", "")
        if "text/html" in content_type:
            raise ValueError("Downloaded file is an HTML page, not the expected NPZ file.")

        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
