"""
Temporary DataLoader for Health Checks.

Provides a lightweight DataLoader builder for quick dataset validation
without requiring the full DataLoaderFactory configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from ..dataset import VisionDataset

_DEFAULT_HEALTHCHECK_BATCH_SIZE = 16  # Batch size for create_temp_loader


def create_temp_loader(
    dataset_path: Path, batch_size: int = _DEFAULT_HEALTHCHECK_BATCH_SIZE
) -> DataLoader[Any]:
    """
    Load a NPZ dataset lazily and return a DataLoader for health checks.

    This avoids loading the entire dataset into RAM at once, which is critical
    for large datasets (e.g., 224x224 images).
    """
    dataset = VisionDataset.lazy(dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader
