"""
Pytest test suite for the VisionDataset class.

Covers eager loading (from_npz), lazy loading, direct constructor,
deterministic subsampling, RGB vs grayscale handling, and __getitem__ behavior.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torchvision import transforms

from orchard.data_handler.dataset import VisionDataset

_rng = np.random.default_rng(0)


# FIXTURES
@pytest.fixture
def rgb_npz(tmp_path: Path):
    """Creates a valid RGB MedMNIST-like NPZ."""
    path = tmp_path / "rgb.npz"
    np.savez(
        path,
        train_images=_rng.integers(0, 255, (20, 28, 28, 3), dtype=np.uint8),
        train_labels=np.arange(20),
        val_images=_rng.integers(0, 255, (10, 28, 28, 3), dtype=np.uint8),
        val_labels=np.arange(10),
        test_images=_rng.integers(0, 255, (10, 28, 28, 3), dtype=np.uint8),
        test_labels=np.arange(10),
    )
    return path


@pytest.fixture
def grayscale_npz(tmp_path: Path):
    """Creates a valid Grayscale MedMNIST-like NPZ."""
    path = tmp_path / "gray.npz"
    np.savez(
        path,
        train_images=_rng.integers(0, 255, (20, 28, 28), dtype=np.uint8),
        train_labels=np.arange(20),
        val_images=_rng.integers(0, 255, (10, 28, 28), dtype=np.uint8),
        val_labels=np.arange(10),
        test_images=_rng.integers(0, 255, (10, 28, 28), dtype=np.uint8),
        test_labels=np.arange(10),
    )
    return path


# TEST: Initialization Errors
@pytest.mark.unit
def test_from_npz_requires_existing_file(tmp_path):
    """from_npz should fail if NPZ does not exist."""
    with pytest.raises(Exception, match="Dataset file not found"):
        VisionDataset.from_npz(path=tmp_path / "missing.npz")


# TEST: Basic Loading (from_npz)
@pytest.mark.unit
def test_len_matches_number_of_samples(rgb_npz):
    """__len__ should match number of loaded labels."""
    ds = VisionDataset.from_npz(path=rgb_npz, split="train")
    assert len(ds) == 20


@pytest.mark.unit
def test_getitem_returns_tensor_pair(rgb_npz):
    """__getitem__ should return (image, label) tensors."""
    ds = VisionDataset.from_npz(path=rgb_npz, split="train")

    img, label = ds[0]

    assert isinstance(img, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert label.dtype == torch.long
    assert img.ndim == 3
    assert img.shape[0] == 3


# TEST: Grayscale Handling
@pytest.mark.unit
def test_grayscale_images_are_expanded(grayscale_npz):
    """Grayscale datasets should be expanded to (H, W, 1)."""
    ds = VisionDataset.from_npz(path=grayscale_npz, split="train")

    assert ds.images.ndim == 4
    assert ds.images.shape[-1] == 1

    img, _ = ds[0]
    assert img.shape[0] == 1


# TEST: Deterministic Subsampling
@pytest.mark.unit
def test_max_samples_is_deterministic(rgb_npz):
    """Subsampling should be reproducible given the same seed."""
    ds1 = VisionDataset.from_npz(path=rgb_npz, split="train", max_samples=5, seed=42)
    ds2 = VisionDataset.from_npz(path=rgb_npz, split="train", max_samples=5, seed=42)

    assert len(ds1) == 5
    assert len(ds2) == 5
    assert np.array_equal(ds1.labels, ds2.labels)


@pytest.mark.unit
def test_max_samples_smaller_than_dataset(rgb_npz):
    """max_samples should reduce dataset size."""
    ds = VisionDataset.from_npz(path=rgb_npz, split="train", max_samples=7)
    assert len(ds) == 7


# TEST: Transform Application
@pytest.mark.unit
def test_custom_transform_is_applied(rgb_npz):
    """Custom transforms should be applied to images."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    ds = VisionDataset.from_npz(
        path=rgb_npz,
        split="train",
        transform=transform,
    )

    img, _ = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape[0] == 3
    assert img.dtype == torch.float32


# TEST: Different Splits
@pytest.mark.unit
@pytest.mark.parametrize("split,expected_len", [("train", 20), ("val", 10), ("test", 10)])
def test_dataset_splits(rgb_npz, split, expected_len):
    """Dataset should correctly load all supported splits."""
    ds = VisionDataset.from_npz(path=rgb_npz, split=split)
    assert len(ds) == expected_len


# TEST: Direct Constructor
@pytest.mark.unit
def test_direct_constructor_rgb():
    """Direct constructor should accept raw arrays."""
    images = np.random.default_rng(0).integers(0, 255, (5, 28, 28, 3), dtype=np.uint8)
    labels = np.arange(5)
    ds = VisionDataset(images, labels)

    assert len(ds) == 5
    img, label = ds[0]
    assert img.shape == (3, 28, 28)
    assert label.dtype == torch.long


@pytest.mark.unit
def test_direct_constructor_grayscale_expands():
    """Direct constructor should expand (N, H, W) to (N, H, W, 1)."""
    images = np.random.default_rng(0).integers(0, 255, (5, 28, 28), dtype=np.uint8)
    labels = np.arange(5)
    ds = VisionDataset(images, labels)

    assert ds.images.ndim == 4
    assert ds.images.shape[-1] == 1
    img, _ = ds[0]
    assert img.shape[0] == 1


# TEST: Lazy Loading
@pytest.mark.unit
def test_lazy_rgb(rgb_npz):
    """Lazy loading should return valid tensors for RGB images."""
    ds = VisionDataset.lazy(rgb_npz)

    assert len(ds) == 20
    img, label = ds[0]
    assert isinstance(img, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert img.shape[0] == 3
    assert label.dtype == torch.long


@pytest.mark.unit
def test_lazy_grayscale(grayscale_npz):
    """Lazy loading should handle grayscale images."""
    ds = VisionDataset.lazy(grayscale_npz)

    assert len(ds) == 20
    img, _ = ds[0]
    assert img.shape[0] == 1
    assert img.shape[1] == 28


@pytest.mark.unit
def test_lazy_keeps_npz_handle(rgb_npz):
    """Lazy loading should keep _npz_handle alive for mmap validity."""
    ds = VisionDataset.lazy(rgb_npz)
    assert ds._npz_handle is not None


@pytest.mark.unit
def test_lazy_val_split(rgb_npz):
    """Lazy loading should support non-default splits."""
    ds = VisionDataset.lazy(rgb_npz, split="val")
    assert len(ds) == 10


@pytest.mark.unit
def test_lazy_with_transform(rgb_npz):
    """Lazy loading should apply transforms."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    ds = VisionDataset.lazy(rgb_npz, transform=transform)

    img, _ = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape[0] == 3
    # Normalized values can be negative
    assert img.min() < 0


@pytest.mark.unit
def test_lazy_with_max_samples(rgb_npz):
    """Lazy loading should support deterministic subsampling."""
    ds = VisionDataset.lazy(rgb_npz, max_samples=5, seed=42)

    assert len(ds) == 5
    assert ds._indices is not None
    assert len(ds._indices) == 5

    # Deterministic: same seed produces same indices
    ds2 = VisionDataset.lazy(rgb_npz, max_samples=5, seed=42)
    assert np.array_equal(ds._indices, ds2._indices)
    assert np.array_equal(ds.labels, ds2.labels)


@pytest.mark.unit
def test_lazy_max_samples_larger_than_dataset(rgb_npz):
    """Lazy loading should ignore max_samples when larger than dataset."""
    ds = VisionDataset.lazy(rgb_npz, max_samples=100)

    assert len(ds) == 20
    assert ds._indices is None
