"""
Unit tests for DataLoaderFactory and related utilities.

Focus:
- DataLoaderFactory.build()
- WeightedRandomSampler
- _get_infrastructure_kwargs (Optuna, CUDA/MPS)
- VisionDataset.lazy() and create_temp_loader
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from orchard.core import DatasetRegistryWrapper
from orchard.data_handler import DataLoaderFactory, create_temp_loader
from orchard.data_handler.dataset import VisionDataset


# MOCK CONFIG AND METADATA
@pytest.fixture
def mock_cfg():
    cfg = MagicMock()
    cfg.dataset.dataset_name = "mock_dataset"
    cfg.dataset.use_weighted_sampler = True
    cfg.dataset.max_samples = 10
    cfg.dataset.num_classes = 2
    cfg.dataset.resolution = 28
    cfg.dataset.lazy_loading = True
    cfg.training.batch_size = 2
    cfg.num_workers = 0
    return cfg


@pytest.fixture
def mock_cfg_no_sampler():
    """Config with weighted sampler disabled."""
    cfg = MagicMock()
    cfg.dataset.dataset_name = "mock_dataset"
    cfg.dataset.use_weighted_sampler = False
    cfg.dataset.max_samples = None
    cfg.dataset.resolution = 28
    cfg.dataset.lazy_loading = True
    cfg.training.batch_size = 2
    cfg.num_workers = 0
    return cfg


@pytest.fixture
def mock_cfg_high_res():
    """Config with high resolution for Optuna test."""
    cfg = MagicMock()
    cfg.dataset.dataset_name = "mock_dataset"
    cfg.dataset.use_weighted_sampler = False
    cfg.dataset.max_samples = None
    cfg.dataset.resolution = 224
    cfg.dataset.lazy_loading = True
    cfg.training.batch_size = 2
    cfg.num_workers = 8
    return cfg


@pytest.fixture
def mock_metadata():
    metadata = MagicMock()
    metadata.path = "/fake/path"
    return metadata


# DATA LOADER FACTORY TESTS
@pytest.mark.unit
def test_build_loaders_with_weighted_sampler(mock_cfg, mock_metadata):
    """Test DataLoaderFactory.build() with sampler and transforms."""
    mock_ds_meta = MagicMock(in_channels=1)

    with patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta):
        with patch(
            "orchard.data_handler.loader.get_pipeline_transforms",
            lambda aug_cfg, img_size, meta: (lambda x: x, lambda x: x),
        ):

            class FakeDataset:
                def __init__(self, **kwargs):
                    self.labels = np.array([0, 1, 0, 1])

                @classmethod
                def from_npz(cls, **kwargs):
                    return cls()

                @classmethod
                def lazy(cls, **kwargs):
                    return cls()

                def __len__(self):
                    return 4

            with patch("orchard.data_handler.loader.VisionDataset", FakeDataset):
                factory = DataLoaderFactory(
                    mock_cfg.dataset,
                    mock_cfg.training,
                    mock_cfg.augmentation,
                    mock_cfg.num_workers,
                    mock_metadata,
                )
                train, val, test = factory.build()

                # Check number of samples
                assert len(train.dataset) == 4
                assert len(val.dataset) == 4
                assert len(test.dataset) == 4

                # Check number of batches
                assert len(train) == 2
                assert len(val) == 2
                assert len(test) == 2

                # Check sampler is WeightedRandomSampler
                assert train.sampler is not None
                assert train.batch_size == mock_cfg.training.batch_size


@pytest.mark.unit
def test_build_loaders_without_weighted_sampler(mock_cfg_no_sampler, mock_metadata):
    """Test DataLoaderFactory.build() WITHOUT weighted sampler."""
    mock_ds_meta = MagicMock(in_channels=1)

    with patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta):
        with patch(
            "orchard.data_handler.loader.get_pipeline_transforms",
            lambda aug_cfg, img_size, meta: (lambda x: x, lambda x: x),
        ):

            class FakeDataset:
                def __init__(self, **kwargs):
                    self.labels = np.array([0, 1, 0, 1])

                @classmethod
                def from_npz(cls, **kwargs):
                    return cls()

                @classmethod
                def lazy(cls, **kwargs):
                    return cls()

                def __len__(self):
                    return 4

            with patch("orchard.data_handler.loader.VisionDataset", FakeDataset):
                factory = DataLoaderFactory(
                    mock_cfg_no_sampler.dataset,
                    mock_cfg_no_sampler.training,
                    mock_cfg_no_sampler.augmentation,
                    mock_cfg_no_sampler.num_workers,
                    mock_metadata,
                )
                (
                    train,
                    _,
                    _,
                ) = factory.build()

                from torch.utils.data import WeightedRandomSampler

                assert not isinstance(train.sampler, WeightedRandomSampler)
                assert train.dataset is not None


@pytest.mark.unit
def test_infra_kwargs_optuna(mock_cfg, mock_metadata):
    """Test _get_infrastructure_kwargs behavior in Optuna mode."""
    mock_ds_meta = MagicMock(in_channels=1)

    with patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta):
        factory = DataLoaderFactory(
            mock_cfg.dataset,
            mock_cfg.training,
            mock_cfg.augmentation,
            mock_cfg.num_workers,
            mock_metadata,
        )
        infra = factory._get_infrastructure_kwargs(is_optuna=True)
        assert infra["num_workers"] <= 6
        assert infra["persistent_workers"] is False


@pytest.mark.unit
def test_infra_kwargs_optuna_high_res(mock_cfg_high_res, mock_metadata):
    """Test _get_infrastructure_kwargs with high resolution in Optuna mode."""
    mock_ds_meta = MagicMock(in_channels=1)

    with patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta):
        factory = DataLoaderFactory(
            mock_cfg_high_res.dataset,
            mock_cfg_high_res.training,
            mock_cfg_high_res.augmentation,
            mock_cfg_high_res.num_workers,
            mock_metadata,
        )
        infra = factory._get_infrastructure_kwargs(is_optuna=True)

        assert infra["num_workers"] <= 4
        assert infra["persistent_workers"] is False


@pytest.mark.unit
def test_infra_kwargs_pin_memory(monkeypatch, mock_cfg, mock_metadata):
    """Test that pin_memory is True if CUDA or MPS available."""
    mock_ds_meta = MagicMock(in_channels=1)

    with patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta):
        factory = DataLoaderFactory(
            mock_cfg.dataset,
            mock_cfg.training,
            mock_cfg.augmentation,
            mock_cfg.num_workers,
            mock_metadata,
        )
        monkeypatch.setattr(torch, "cuda", MagicMock(is_available=lambda: True))
        monkeypatch.setattr(torch.backends, "mps", MagicMock(is_available=lambda: False))

        infra = factory._get_infrastructure_kwargs()
        assert infra["pin_memory"] is True


@pytest.mark.unit
def test_infra_kwargs_no_pin_memory(monkeypatch, mock_cfg, mock_metadata):
    """Test that pin_memory is False when neither CUDA nor MPS available."""
    mock_ds_meta = MagicMock(in_channels=1)

    with patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta):
        factory = DataLoaderFactory(
            mock_cfg.dataset,
            mock_cfg.training,
            mock_cfg.augmentation,
            mock_cfg.num_workers,
            mock_metadata,
        )
        monkeypatch.setattr(torch, "cuda", MagicMock(is_available=lambda: False))
        monkeypatch.setattr(torch.backends, "mps", MagicMock(is_available=lambda: False))

        infra = factory._get_infrastructure_kwargs()
        assert infra["pin_memory"] is False


# LAZY VISION DATASET TESTS
@pytest.mark.unit
def test_vision_dataset_lazy():
    """Test VisionDataset.lazy() loads and returns tensors correctly."""
    rng = np.random.default_rng(seed=42)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "dummy.npz"
        data = {
            "train_images": rng.integers(0, 255, (5, 28, 28), dtype=np.uint8),
            "train_labels": rng.integers(0, 2, (5, 1), dtype=np.int64),
        }
        np.savez(tmp_path, **data)

        dataset = VisionDataset.lazy(tmp_path)
        assert len(dataset) == 5

        img, label = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape[0] == 1
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.long


@pytest.mark.unit
def test_vision_dataset_lazy_rgb():
    """Test VisionDataset.lazy() with RGB images."""
    rng = np.random.default_rng(seed=42)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "dummy_rgb.npz"
        data = {
            "train_images": rng.integers(0, 255, (5, 28, 28, 3), dtype=np.uint8),
            "train_labels": rng.integers(0, 2, (5, 1), dtype=np.int64),
        }
        np.savez(tmp_path, **data)

        dataset = VisionDataset.lazy(tmp_path)
        assert len(dataset) == 5

        img, label = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape[0] == 3
        assert img.shape[1] == 28
        assert img.shape[2] == 28
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.long


@pytest.mark.unit
def test_vision_dataset_lazy_grayscale_2d():
    """Test VisionDataset.lazy() with 2D grayscale images."""
    rng = np.random.default_rng(seed=42)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "dummy_gray.npz"
        data = {
            "train_images": rng.integers(0, 255, (5, 28, 28), dtype=np.uint8),
            "train_labels": rng.integers(0, 2, (5, 1), dtype=np.int64),
        }
        np.savez(tmp_path, **data)

        dataset = VisionDataset.lazy(tmp_path)
        img, _ = dataset[0]

        assert img.shape[0] == 1
        assert img.shape[1] == 28
        assert img.shape[2] == 28


@pytest.mark.unit
def test_create_temp_loader():
    """Test create_temp_loader returns a working DataLoader."""
    rng = np.random.default_rng(seed=42)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "dummy.npz"
        data = {
            "train_images": rng.integers(0, 255, (5, 28, 28), dtype=np.uint8),
            "train_labels": rng.integers(0, 2, (5, 1), dtype=np.int64),
        }
        np.savez(tmp_path, **data)

        loader = create_temp_loader(tmp_path, batch_size=2)
        batch_imgs, _ = next(iter(loader))
        assert batch_imgs.shape[0] <= 2
        assert batch_imgs.shape[1] == 1


@pytest.mark.unit
def test_create_temp_loader_rgb():
    """Test create_temp_loader with RGB images."""
    rng = np.random.default_rng(seed=42)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "dummy_rgb.npz"
        data = {
            "train_images": rng.integers(0, 255, (8, 32, 32, 3), dtype=np.uint8),
            "train_labels": rng.integers(0, 3, (8, 1), dtype=np.int64),
        }
        np.savez(tmp_path, **data)

        loader = create_temp_loader(tmp_path, batch_size=4)
        batch_imgs, _ = next(iter(loader))

        assert batch_imgs.shape[0] <= 4
        assert batch_imgs.shape[1] == 3
        assert batch_imgs.shape[2] == 32
        assert batch_imgs.shape[3] == 32


# TESTS: get_dataloader
@pytest.mark.unit
def test_get_dataloaders_convenience_function(mock_cfg, mock_metadata):
    """Test get_dataloaders convenience function."""
    from orchard.data_handler.loader import get_dataloaders

    with patch("orchard.data_handler.loader.DataLoaderFactory") as mock_factory_class:
        mock_factory = MagicMock()
        mock_train = MagicMock()
        mock_val = MagicMock()
        mock_test = MagicMock()
        mock_factory.build.return_value = (mock_train, mock_val, mock_test)
        mock_factory_class.return_value = mock_factory

        train, val, test = get_dataloaders(
            mock_metadata,
            mock_cfg.dataset,
            mock_cfg.training,
            mock_cfg.augmentation,
            mock_cfg.num_workers,
            is_optuna=False,
        )
        mock_factory_class.assert_called_once_with(
            mock_cfg.dataset,
            mock_cfg.training,
            mock_cfg.augmentation,
            mock_cfg.num_workers,
            mock_metadata,
        )
        mock_factory.build.assert_called_once_with(is_optuna=False)

        assert train == mock_train
        assert val == mock_val
        assert test == mock_test


@pytest.mark.unit
def test_get_dataloaders_with_optuna_mode(mock_cfg, mock_metadata):
    """Test get_dataloaders with is_optuna=True."""
    from orchard.data_handler.loader import get_dataloaders

    with patch("orchard.data_handler.loader.DataLoaderFactory") as mock_factory_class:
        mock_factory = MagicMock()
        mock_loaders = (MagicMock(), MagicMock(), MagicMock())
        mock_factory.build.return_value = mock_loaders
        mock_factory_class.return_value = mock_factory

        result = get_dataloaders(
            mock_metadata,
            mock_cfg.dataset,
            mock_cfg.training,
            mock_cfg.augmentation,
            mock_cfg.num_workers,
            is_optuna=True,
        )

        mock_factory.build.assert_called_once_with(is_optuna=True)
        assert result == mock_loaders


@pytest.mark.unit
def test_balancing_sampler_missing_class_raises(mock_cfg, mock_metadata):
    """Test _get_balancing_sampler raises ValueError when classes are missing."""
    mock_ds_meta = MagicMock(in_channels=1)

    with patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta):
        factory = DataLoaderFactory(
            mock_cfg.dataset,
            mock_cfg.training,
            mock_cfg.augmentation,
            mock_cfg.num_workers,
            mock_metadata,
        )

        dataset = MagicMock()
        dataset.labels = np.array([0, 1, 0, 1])  # only 2 of 3 classes
        mock_cfg.dataset.num_classes = 3

        with pytest.raises(ValueError, match="missing 1 of 3 classes"):
            factory._get_balancing_sampler(dataset)


@pytest.mark.unit
def test_balancing_sampler_all_classes_present(mock_cfg, mock_metadata):
    """Test _get_balancing_sampler succeeds when all classes are represented."""
    mock_ds_meta = MagicMock(in_channels=1)

    with patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta):
        factory = DataLoaderFactory(
            mock_cfg.dataset,
            mock_cfg.training,
            mock_cfg.augmentation,
            mock_cfg.num_workers,
            mock_metadata,
        )

        dataset = MagicMock()
        dataset.labels = np.array([0, 1, 0, 1])
        mock_cfg.dataset.num_classes = 2

        sampler = factory._get_balancing_sampler(dataset)
        assert sampler is not None


# MAIN TEST RUNNER
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
