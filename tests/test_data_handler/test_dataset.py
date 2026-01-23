"""
Pytest test suite for the MedMNISTDataset class.

Covers dataset initialization, deterministic subsampling,
RGB vs grayscale handling, and __getitem__ behavior.
"""

# Standard Imports
from pathlib import Path
from types import SimpleNamespace

# Third-Party Imports
import numpy as np
import pytest
import torch
from torchvision import transforms

#                                Module Under Test                            #
from orchard.data_handler.dataset import MedMNISTDataset


# FIXTURES
@pytest.fixture
def cfg():
    """Minimal Config stub with deterministic seed."""
    return SimpleNamespace(
        training=SimpleNamespace(seed=42),
    )


@pytest.fixture
def rgb_npz(tmp_path: Path):
    """Creates a valid RGB MedMNIST-like NPZ."""
    path = tmp_path / "rgb.npz"
    np.savez(
        path,
        train_images=np.random.randint(0, 255, (20, 28, 28, 3), dtype=np.uint8),
        train_labels=np.arange(20),
        val_images=np.random.randint(0, 255, (10, 28, 28, 3), dtype=np.uint8),
        val_labels=np.arange(10),
        test_images=np.random.randint(0, 255, (10, 28, 28, 3), dtype=np.uint8),
        test_labels=np.arange(10),
    )
    return path


@pytest.fixture
def grayscale_npz(tmp_path: Path):
    """Creates a valid Grayscale MedMNIST-like NPZ."""
    path = tmp_path / "gray.npz"
    np.savez(
        path,
        train_images=np.random.randint(0, 255, (20, 28, 28), dtype=np.uint8),
        train_labels=np.arange(20),
        val_images=np.random.randint(0, 255, (10, 28, 28), dtype=np.uint8),
        val_labels=np.arange(10),
        test_images=np.random.randint(0, 255, (10, 28, 28), dtype=np.uint8),
        test_labels=np.arange(10),
    )
    return path


#                          TEST: Initialization Errors                        #
def test_init_requires_cfg(rgb_npz):
    """Dataset initialization without Config should fail."""
    with pytest.raises(ValueError):
        MedMNISTDataset(path=rgb_npz, cfg=None)


def test_init_requires_existing_file(cfg, tmp_path):
    """Dataset initialization should fail if NPZ does not exist."""
    with pytest.raises(FileNotFoundError):
        MedMNISTDataset(path=tmp_path / "missing.npz", cfg=cfg)


#                          TEST: Basic Loading                                #
def test_len_matches_number_of_samples(cfg, rgb_npz):
    """__len__ should match number of loaded labels."""
    ds = MedMNISTDataset(path=rgb_npz, split="train", cfg=cfg)
    assert len(ds) == 20


def test_getitem_returns_tensor_pair(cfg, rgb_npz):
    """__getitem__ should return (image, label) tensors."""
    ds = MedMNISTDataset(path=rgb_npz, split="train", cfg=cfg)

    img, label = ds[0]

    assert isinstance(img, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert label.dtype == torch.long
    assert img.ndim == 3
    assert img.shape[0] == 3


#                   TEST: Grayscale Handling                                  #
def test_grayscale_images_are_expanded(cfg, grayscale_npz):
    """Grayscale datasets should be expanded to (H, W, 1)."""
    ds = MedMNISTDataset(path=grayscale_npz, split="train", cfg=cfg)

    assert ds.images.ndim == 4
    assert ds.images.shape[-1] == 1

    img, _ = ds[0]
    assert img.shape[0] == 1


#                   TEST: Deterministic Subsampling                           #
def test_max_samples_is_deterministic(cfg, rgb_npz):
    """Subsampling should be reproducible given the same seed."""
    ds1 = MedMNISTDataset(path=rgb_npz, split="train", max_samples=5, cfg=cfg)
    ds2 = MedMNISTDataset(path=rgb_npz, split="train", max_samples=5, cfg=cfg)

    assert len(ds1) == 5
    assert len(ds2) == 5
    assert np.array_equal(ds1.labels, ds2.labels)


def test_max_samples_smaller_than_dataset(cfg, rgb_npz):
    """max_samples should reduce dataset size."""
    ds = MedMNISTDataset(path=rgb_npz, split="train", max_samples=7, cfg=cfg)
    assert len(ds) == 7


#                       TEST: Transform Application                           #
def test_custom_transform_is_applied(cfg, rgb_npz):
    """Custom transforms should be applied to images."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    ds = MedMNISTDataset(
        path=rgb_npz,
        split="train",
        transform=transform,
        cfg=cfg,
    )

    img, _ = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape[0] == 3
    assert img.dtype == torch.float32


#                     TEST: Different Splits                                  #
@pytest.mark.parametrize("split,expected_len", [("train", 20), ("val", 10), ("test", 10)])
def test_dataset_splits(cfg, rgb_npz, split, expected_len):
    """Dataset should correctly load all supported splits."""
    ds = MedMNISTDataset(path=rgb_npz, split=split, cfg=cfg)
    assert len(ds) == expected_len
