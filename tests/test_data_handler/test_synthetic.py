"""
Pytest test suite for synthetic MedMNIST dataset generation.

Covers creation of RGB and grayscale synthetic datasets,
NPZ file structure, metadata correctness, and data integrity.
"""

from pathlib import Path

import numpy as np

from orchard.data_handler import (
    DatasetData,
    create_synthetic_dataset,
    create_synthetic_grayscale_dataset,
)


# TESTS
def test_create_synthetic_dataset_returns_metadata():
    """Factory should return a valid DatasetData object."""
    data = create_synthetic_dataset()

    assert isinstance(data, DatasetData)
    assert isinstance(data.path, Path)
    assert data.path.exists()
    assert data.name == "syntheticmnist"
    assert data.is_rgb is True
    assert data.num_classes == 8


def test_synthetic_npz_contains_all_required_keys():
    """Synthetic NPZ file should match MedMNIST format."""
    data = create_synthetic_dataset(samples=50)

    with np.load(data.path) as npz:
        keys = set(npz.keys())

    assert keys == {
        "train_images",
        "train_labels",
        "val_images",
        "val_labels",
        "test_images",
        "test_labels",
    }


def test_synthetic_dataset_shapes_rgb():
    """RGB synthetic dataset should have correct shapes."""
    samples = 40
    resolution = 32
    channels = 3

    data = create_synthetic_dataset(
        samples=samples,
        resolution=resolution,
        channels=channels,
    )

    with np.load(data.path) as npz:
        assert npz["train_images"].shape == (
            samples,
            resolution,
            resolution,
            channels,
        )
        assert npz["train_labels"].shape == (samples, 1)

        expected_aux = max(10, samples // 10)
        assert npz["val_images"].shape == (
            expected_aux,
            resolution,
            resolution,
            channels,
        )
        assert npz["test_images"].shape == (
            expected_aux,
            resolution,
            resolution,
            channels,
        )


def test_synthetic_dataset_label_range():
    """Labels should be within [0, num_classes)."""
    num_classes = 5
    data = create_synthetic_dataset(num_classes=num_classes)

    with np.load(data.path) as npz:
        for split in ["train_labels", "val_labels", "test_labels"]:
            labels = npz[split]
            assert labels.min() >= 0
            assert labels.max() < num_classes


def test_create_synthetic_grayscale_dataset():
    """Grayscale variant should generate single-channel data."""
    data = create_synthetic_grayscale_dataset(
        num_classes=3,
        samples=30,
        resolution=28,
    )

    assert isinstance(data, DatasetData)
    assert data.is_rgb is False
    assert data.name == "syntheticmnist_gray"

    with np.load(data.path) as npz:
        assert npz["train_images"].ndim == 4
        assert npz["train_images"].shape[-1] == 1


def test_multiple_calls_create_distinct_files():
    """Each synthetic dataset call should create a unique file."""
    d1 = create_synthetic_dataset()
    d2 = create_synthetic_dataset()

    assert d1.path != d2.path
    assert d1.path.exists()
    assert d2.path.exists()
