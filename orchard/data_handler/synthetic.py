"""Synthetic Data Handler for Testing.

This module provides tiny synthetic MedMNIST datasets for unit tests without
requiring any external downloads or network access. It generates random image
data and labels that match the MedMNIST format specifications.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import tempfile
from pathlib import Path

# =========================================================================== #
#                               Third-Party Imports                           #
# =========================================================================== #
import numpy as np

# =========================================================================== #
#                              Internal Imports                               #
# =========================================================================== #
from orchard.data_handler.fetcher import MedMNISTData

# =========================================================================== #
#                                FACTORY FUNCTIONS                            #
# =========================================================================== #


def create_synthetic_dataset(
    num_classes: int = 8,
    samples: int = 100,
    resolution: int = 28,
    channels: int = 3,
    name: str = "syntheticmnist",
) -> MedMNISTData:
    """Create a synthetic MedMNIST-compatible dataset for testing.

    This function generates random image data and labels, saves them to a
    temporary .npz file, and returns a MedMNISTData object that can be used
    with the existing data pipeline.

    Args:
        num_classes: Number of classification categories (default: 8)
        samples: Number of training samples (default: 100)
        resolution: Image resolution (HxW) (default: 28)
        channels: Number of color channels (default: 3 for RGB)
        name: Dataset name for identification (default: "syntheticmnist")

    Returns:
        MedMNISTData: A data object compatible with the existing pipeline

    Example:
        >>> data = create_synthetic_dataset(num_classes=8, samples=100)
        >>> train_loader, val_loader, test_loader = get_dataloaders(data, cfg)
    """
    # Generate synthetic image data
    train_images = np.random.randint(
        0, 255, (samples, resolution, resolution, channels), dtype=np.uint8
    )
    train_labels = np.random.randint(0, num_classes, (samples, 1), dtype=np.uint8)

    # Validation and test sets are smaller (10% of training size each)
    val_samples = max(10, samples // 10)
    test_samples = max(10, samples // 10)

    val_images = np.random.randint(
        0, 255, (val_samples, resolution, resolution, channels), dtype=np.uint8
    )
    val_labels = np.random.randint(0, num_classes, (val_samples, 1), dtype=np.uint8)

    test_images = np.random.randint(
        0, 255, (test_samples, resolution, resolution, channels), dtype=np.uint8
    )
    test_labels = np.random.randint(0, num_classes, (test_samples, 1), dtype=np.uint8)

    # Create a temporary .npz file with MedMNIST format
    temp_file = tempfile.NamedTemporaryFile(
        suffix=".npz", delete=False, prefix="synthetic_medmnist_"
    )
    temp_path = Path(temp_file.name)

    # Save in MedMNIST .npz format with correct key names
    np.savez(
        temp_path,
        train_images=train_images,
        train_labels=train_labels,
        val_images=val_images,
        val_labels=val_labels,
        test_images=test_images,
        test_labels=test_labels,
    )

    # Return a MedMNISTData object with all required parameters
    is_rgb = channels == 3

    return MedMNISTData(
        path=temp_path,
        name=name,
        is_rgb=is_rgb,
        num_classes=num_classes,
    )


# =========================================================================== #
#                            GRAYSCALE VARIANT                                #
# =========================================================================== #


def create_synthetic_grayscale_dataset(
    num_classes: int = 8,
    samples: int = 100,
    resolution: int = 28,
) -> MedMNISTData:
    """Create a synthetic grayscale MedMNIST dataset for testing.

    Convenience function for creating single-channel (grayscale) synthetic data.

    Args:
        num_classes: Number of classification categories (default: 8)
        samples: Number of training samples (default: 100)
        resolution: Image resolution (HxW) (default: 28)

    Returns:
        MedMNISTData: A grayscale data object compatible with the pipeline
    """
    return create_synthetic_dataset(
        num_classes=num_classes,
        samples=samples,
        resolution=resolution,
        channels=1,
        name="syntheticmnist_gray",
    )
