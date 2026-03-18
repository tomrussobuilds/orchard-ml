"""
Synthetic Data Handler for Testing.

This module provides tiny synthetic NPZ datasets for unit tests without
requiring any external downloads or network access. It generates random image
data and labels that match the expected NPZ format specifications.

Note:
    mutmut's trampoline resolves default parameters in the wrapper before
    dispatching to the mutant function, making default-value mutations
    unkillable.  Body mutations on internal pixel/label generation are also
    unobservable because tests verify the returned ``DatasetData`` metadata,
    not raw byte content.  Lines are marked ``pragma: no mutate`` accordingly.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ...core.paths import DEFAULT_SEED, MIN_SPLIT_SAMPLES
from ..fetcher import DatasetData

_SYNTHETIC_SEED = DEFAULT_SEED  # pragma: no mutate
_SYNTHETIC_PIXEL_RANGE = 255  # pragma: no mutate


# FACTORY FUNCTIONS
def create_synthetic_dataset(
    num_classes: int = 8,  # pragma: no mutate
    samples: int = 100,  # pragma: no mutate
    resolution: int = 28,  # pragma: no mutate
    channels: int = 3,  # pragma: no mutate
    name: str = "syntheticmnist",  # pragma: no mutate
) -> DatasetData:
    """
    Create a synthetic NPZ-compatible dataset for testing.

    This function generates random image data and labels, saves them to a
    temporary .npz file, and returns a DatasetData object that can be used
    with the existing data pipeline.

    Args:
        num_classes: Number of target categories (default: 8)
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
    rng = np.random.default_rng(_SYNTHETIC_SEED)  # pragma: no mutate

    # Generate synthetic image data  # pragma: no mutate
    train_images = rng.integers(  # pragma: no mutate
        0,  # pragma: no mutate
        _SYNTHETIC_PIXEL_RANGE,  # pragma: no mutate
        (samples, resolution, resolution, channels),  # pragma: no mutate
        dtype=np.uint8,  # pragma: no mutate
    )
    train_labels = rng.integers(0, num_classes, (samples, 1), dtype=np.uint8)  # pragma: no mutate

    # Validation and test sets are smaller (10% of training size each)
    val_samples = max(MIN_SPLIT_SAMPLES, samples // 10)  # pragma: no mutate
    test_samples = max(MIN_SPLIT_SAMPLES, samples // 10)  # pragma: no mutate

    val_images = rng.integers(  # pragma: no mutate
        0,  # pragma: no mutate
        _SYNTHETIC_PIXEL_RANGE,  # pragma: no mutate
        (val_samples, resolution, resolution, channels),  # pragma: no mutate
        dtype=np.uint8,  # pragma: no mutate
    )
    val_labels = rng.integers(0, num_classes, (val_samples, 1), dtype=np.uint8)  # pragma: no mutate

    test_images = rng.integers(  # pragma: no mutate
        0,  # pragma: no mutate
        _SYNTHETIC_PIXEL_RANGE,  # pragma: no mutate
        (test_samples, resolution, resolution, channels),  # pragma: no mutate
        dtype=np.uint8,  # pragma: no mutate
    )
    test_labels = rng.integers(  # pragma: no mutate
        0, num_classes, (test_samples, 1), dtype=np.uint8  # pragma: no mutate
    )

    # Create a temporary .npz file with standard format
    temp_file = tempfile.NamedTemporaryFile(  # pragma: no mutate
        suffix=".npz", delete=False, prefix="synthetic_dataset_"  # pragma: no mutate
    )
    temp_path = Path(temp_file.name)
    temp_file.close()

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
    num_classes: int = 8,  # pragma: no mutate
    samples: int = 100,  # pragma: no mutate
    resolution: int = 28,  # pragma: no mutate
) -> DatasetData:
    """
    Create a synthetic grayscale NPZ dataset for testing.

    Convenience function for creating single-channel (grayscale) synthetic data.

    Args:
        num_classes: Number of target categories (default: 8)
        samples: Number of training samples (default: 100)
        resolution: Image resolution (HxW) (default: 28)

    Returns:
        DatasetData: A grayscale data object compatible with the pipeline
    """
    return create_synthetic_dataset(  # pragma: no mutate
        num_classes=num_classes,  # pragma: no mutate
        samples=samples,  # pragma: no mutate
        resolution=resolution,  # pragma: no mutate
        channels=1,
        name="syntheticmnist_gray",
    )
