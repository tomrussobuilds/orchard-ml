"""
Synthetic Data Handler for Testing.

This module provides tiny synthetic NPZ datasets for unit tests without
requiring any external downloads or network access. It generates random image
data and labels that match the expected NPZ format specifications.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from .fetcher import DatasetData

_SYNTHETIC_SEED = 42  # Fixed seed for deterministic synthetic data generation
_SYNTHETIC_PIXEL_RANGE = 255  # Exclusive upper bound for rng.integers (uint8)
_MIN_SPLIT_SAMPLES = 10  # Floor for val/test split sizes


# FACTORY FUNCTIONS
def create_synthetic_dataset(
    num_classes: int = 8,
    samples: int = 100,
    resolution: int = 28,
    channels: int = 3,
    name: str = "syntheticmnist",
) -> DatasetData:
    """
    Create a synthetic NPZ-compatible dataset for testing.

    This function generates random image data and labels, saves them to a
    temporary .npz file, and returns a DatasetData object that can be used
    with the existing data pipeline.

    Args:
        num_classes: Number of classification categories (default: 8)
        samples: Number of training samples (default: 100)
        resolution: Image resolution (HxW) (default: 28)
        channels: Number of color channels (default: 3 for RGB)
        name: Dataset name for identification (default: "syntheticmnist")

    Returns:
        DatasetData: A data object compatible with the existing pipeline

    Example:
        >>> data = create_synthetic_dataset(num_classes=8, samples=100)
        >>> train_loader, val_loader, test_loader = get_dataloaders(
        ...     data, cfg.dataset, cfg.training, cfg.augmentation, cfg.num_workers
        ... )
    """
    rng = np.random.default_rng(_SYNTHETIC_SEED)

    # Generate synthetic image data
    train_images = rng.integers(
        0, _SYNTHETIC_PIXEL_RANGE, (samples, resolution, resolution, channels), dtype=np.uint8
    )
    train_labels = rng.integers(0, num_classes, (samples, 1), dtype=np.uint8)

    # Validation and test sets are smaller (10% of training size each)
    val_samples = max(_MIN_SPLIT_SAMPLES, samples // 10)
    test_samples = max(_MIN_SPLIT_SAMPLES, samples // 10)

    val_images = rng.integers(
        0, _SYNTHETIC_PIXEL_RANGE, (val_samples, resolution, resolution, channels), dtype=np.uint8
    )
    val_labels = rng.integers(0, num_classes, (val_samples, 1), dtype=np.uint8)

    test_images = rng.integers(
        0, _SYNTHETIC_PIXEL_RANGE, (test_samples, resolution, resolution, channels), dtype=np.uint8
    )
    test_labels = rng.integers(0, num_classes, (test_samples, 1), dtype=np.uint8)

    # Create a temporary .npz file with standard format
    temp_file = tempfile.NamedTemporaryFile(
        suffix=".npz", delete=False, prefix="synthetic_dataset_"
    )
    temp_path = Path(temp_file.name)

    # Save in NPZ format with correct key names
    np.savez(
        temp_path,
        train_images=train_images,
        train_labels=train_labels,
        val_images=val_images,
        val_labels=val_labels,
        test_images=test_images,
        test_labels=test_labels,
    )

    # Return a DatasetData object with all required parameters
    is_rgb = channels == 3

    return DatasetData(
        path=temp_path,
        name=name,
        is_rgb=is_rgb,
        num_classes=num_classes,
    )


# GRAYSCALE VARIANT
def create_synthetic_grayscale_dataset(
    num_classes: int = 8,
    samples: int = 100,
    resolution: int = 28,
) -> DatasetData:
    """
    Create a synthetic grayscale NPZ dataset for testing.

    Convenience function for creating single-channel (grayscale) synthetic data.

    Args:
        num_classes: Number of classification categories (default: 8)
        samples: Number of training samples (default: 100)
        resolution: Image resolution (HxW) (default: 28)

    Returns:
        DatasetData: A grayscale data object compatible with the pipeline
    """
    return create_synthetic_dataset(
        num_classes=num_classes,
        samples=samples,
        resolution=resolution,
        channels=1,
        name="syntheticmnist_gray",
    )
