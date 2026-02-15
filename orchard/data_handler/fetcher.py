"""
Dataset Fetching Dispatcher and Loading Interface

Central entry point for dataset retrieval. Routes each dataset to its
dedicated fetch module inside the ``fetchers/`` sub-package and exposes the
loading functions that return ``DatasetData`` containers.

Adding a new domain only requires a new branch in ``ensure_dataset_npz``
and a corresponding module in ``fetchers/``.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from orchard.core import DatasetMetadata, validate_npz_keys
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


# FETCHING DISPATCHER
def ensure_dataset_npz(
    metadata: DatasetMetadata,
    retries: int = 5,
    delay: float = 5.0,
) -> Path:
    """
    Dispatcher that routes each dataset to its dedicated fetch pipeline.

    Automatically detects dataset type from ``metadata.name`` and delegates
    to the appropriate download/conversion module. Adding a new domain
    (e.g. a new resolution or source) only requires a new branch here and
    a corresponding fetch module.

    Args:
        metadata (DatasetMetadata): Metadata containing URL, MD5, name and target path.
        retries (int): Max number of download attempts (MedMNIST only).
        delay (float): Delay (seconds) between retries (MedMNIST only).

    Returns:
        Path: Path to the successfully validated .npz file.
    """
    # Galaxy10 requires HDF5 download and conversion to NPZ
    if metadata.name == "galaxy10":
        from .fetchers import ensure_galaxy10_npz

        return ensure_galaxy10_npz(metadata)

    # Default: standard MedMNIST NPZ download with retries and MD5 check
    from .fetchers import ensure_medmnist_npz

    return ensure_medmnist_npz(metadata, retries=retries, delay=delay)


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
