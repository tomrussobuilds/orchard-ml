"""
Dataset Fetching Dispatcher and Loading Interface.

Central entry point for dataset retrieval. Routes each dataset to its
dedicated fetch module inside the ``fetchers/`` sub-package and exposes the
loading functions that return ``DatasetData`` containers.

Adding a new domain only requires a new branch in ``ensure_dataset_npz``
and a corresponding module in ``fetchers/``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core import DatasetMetadata, validate_npz_keys
from ..core.paths import LOGGER_NAME


# DATA CONTAINERS
@dataclass(frozen=True)
class DatasetData:
    """
    Metadata container for a loaded dataset.

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
        retries (int): Max number of download attempts (NPZ fetcher only).
        delay (float): Delay (seconds) between retries (NPZ fetcher only).

    Returns:
        Path: Path to the successfully validated .npz file.
    """
    # Galaxy10 requires HDF5 download and conversion to NPZ
    if metadata.name == "galaxy10":
        from .fetchers import ensure_galaxy10_npz

        return ensure_galaxy10_npz(metadata)

    # CIFAR-10/100 via torchvision download and NPZ conversion
    if metadata.name in ("cifar10", "cifar100"):
        from .fetchers import ensure_cifar_npz

        return ensure_cifar_npz(metadata)

    # Default: standard NPZ download with retries and MD5 check
    from .fetchers import ensure_medmnist_npz

    return ensure_medmnist_npz(metadata, retries=retries, delay=delay)


# LOADING INTERFACE
def _load_and_inspect(
    metadata: DatasetMetadata,
    chunk_size: int | None = None,
) -> DatasetData:
    """
    Shared loader: fetch NPZ, validate keys, infer format.

    Args:
        metadata: Dataset metadata (URL, MD5, name, path).
        chunk_size: If given, only inspect the first *chunk_size* samples
                    (cheaper for health-checks). ``None`` inspects all.

    Returns:
        DatasetData with path, name, is_rgb, num_classes.
    """
    path = ensure_dataset_npz(metadata)

    with np.load(path) as data:
        validate_npz_keys(data)

        images = data["train_images"][:chunk_size]  # None → full array
        labels = data["train_labels"][:chunk_size]

        is_rgb = images.ndim == 4 and images.shape[-1] == 3
        num_classes = len(np.unique(labels))

    return DatasetData(path=path, name=metadata.name, is_rgb=is_rgb, num_classes=num_classes)


def load_dataset(metadata: DatasetMetadata) -> DatasetData:
    """
    Ensures the dataset is present and returns its metadata container.
    """
    return _load_and_inspect(metadata)


def load_dataset_health_check(metadata: DatasetMetadata, chunk_size: int = 100) -> DatasetData:
    """
    Quick health-check: inspects only the first *chunk_size* samples.

    Args:
        metadata: Dataset metadata (URL, MD5, name, path).
        chunk_size: Number of samples to inspect (default 100).

    Returns:
        DatasetData with format info derived from the chunk.
    """
    return _load_and_inspect(metadata, chunk_size=chunk_size)
