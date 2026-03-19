"""
Unit tests for DataLoaderFactory and related utilities.

Focus:
- DataLoaderFactory.build()
- WeightedRandomSampler
- _get_infrastructure_kwargs (Optuna, CUDA/MPS)
- VisionDataset.lazy()
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from orchard.core import DatasetRegistryWrapper
from orchard.data_handler import DataLoaderFactory
from orchard.data_handler.dataset import VisionDataset
from orchard.data_handler.diagnostic import create_temp_loader
from orchard.exceptions import OrchardDatasetError


# MOCK CONFIG AND METADATA
@pytest.fixture
def mock_cfg() -> None:
    cfg = MagicMock()
    cfg.dataset.dataset_name = "mock_dataset"
    cfg.dataset.use_weighted_sampler = True
    cfg.dataset.max_samples = 10
    cfg.dataset.num_classes = 2
    cfg.dataset.resolution = 28
    cfg.dataset.lazy_loading = True
    cfg.training.batch_size = 2
    cfg.num_workers = 0
    return cfg  # type: ignore


@pytest.fixture
def mock_cfg_no_sampler() -> None:
    """Config with weighted sampler disabled."""
    cfg = MagicMock()
    cfg.dataset.dataset_name = "mock_dataset"
    cfg.dataset.use_weighted_sampler = False
    cfg.dataset.max_samples = None
    cfg.dataset.resolution = 28
    cfg.dataset.lazy_loading = True
    cfg.training.batch_size = 2
    cfg.num_workers = 0
    return cfg  # type: ignore


@pytest.fixture
def mock_cfg_high_res() -> None:
    """Config with high resolution for Optuna test."""
    cfg = MagicMock()
    cfg.dataset.dataset_name = "mock_dataset"
    cfg.dataset.use_weighted_sampler = False
    cfg.dataset.max_samples = None
    cfg.dataset.resolution = 224
    cfg.dataset.lazy_loading = True
    cfg.training.batch_size = 2
    cfg.num_workers = 8
    return cfg  # type: ignore


@pytest.fixture
def mock_metadata() -> None:
    metadata = MagicMock()
    metadata.path = "/fake/path"
    return metadata  # type: ignore


# DATA LOADER FACTORY TESTS
@pytest.mark.unit
def test_build_loaders_with_weighted_sampler(mock_cfg: MagicMock, mock_metadata: MagicMock) -> None:
    """Test DataLoaderFactory.build() with sampler and transforms."""
    mock_ds_meta = MagicMock(in_channels=1)

    with patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta):
        with patch(
            "orchard.data_handler.loader.get_pipeline_transforms",
            lambda aug_cfg, img_size, meta, **kw: (lambda x: x, lambda x: x),
        ):

            class FakeDataset:
                def __init__(self, **kwargs: object) -> None:
                    self.labels = np.array([0, 1, 0, 1])

                @classmethod
                def from_npz(cls: Any, **kwargs: object) -> None:
                    return cls()  # type: ignore

                @classmethod
                def lazy(cls: Any, **kwargs: object) -> None:
                    return cls()  # type: ignore

                def __len__(self) -> None:
                    return 4  # type: ignore

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
                assert len(train.dataset) == 4  # type: ignore
                assert len(val.dataset) == 4  # type: ignore
                assert len(test.dataset) == 4  # type: ignore

                # Check number of batches
                assert len(train) == 2
                assert len(val) == 2
                assert len(test) == 2

                # Check sampler is WeightedRandomSampler
                assert train.sampler is not None
                assert train.batch_size == mock_cfg.training.batch_size


@pytest.mark.unit
def test_build_loaders_without_weighted_sampler(
    mock_cfg_no_sampler: MagicMock, mock_metadata: MagicMock
) -> None:
    """Test DataLoaderFactory.build() WITHOUT weighted sampler."""
    mock_ds_meta = MagicMock(in_channels=1)

    with patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta):
        with patch(
            "orchard.data_handler.loader.get_pipeline_transforms",
            lambda aug_cfg, img_size, meta, **kw: (lambda x: x, lambda x: x),
        ):

            class FakeDataset:
                def __init__(self, **kwargs: object) -> None:
                    self.labels = np.array([0, 1, 0, 1])

                @classmethod
                def from_npz(cls: Any, **kwargs: object) -> None:
                    return cls()  # type: ignore

                @classmethod
                def lazy(cls: Any, **kwargs: object) -> None:
                    return cls()  # type: ignore

                def __len__(self) -> None:
                    return 4  # type: ignore

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
def test_infra_kwargs_optuna(mock_cfg: MagicMock, mock_metadata: MagicMock) -> None:
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
def test_infra_kwargs_optuna_high_res(
    mock_cfg_high_res: MagicMock, mock_metadata: MagicMock
) -> None:
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
def test_infra_kwargs_pin_memory(
    monkeypatch: pytest.MonkeyPatch, mock_cfg: MagicMock, mock_metadata: MagicMock
) -> None:
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
def test_infra_kwargs_no_pin_memory(
    monkeypatch: pytest.MonkeyPatch, mock_cfg: MagicMock, mock_metadata: MagicMock
) -> None:
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
def test_vision_dataset_lazy() -> None:
    """Test VisionDataset.lazy() loads and returns tensors correctly."""
    rng = np.random.default_rng(seed=42)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "dummy.npz"
        data = {
            "train_images": rng.integers(0, 255, (5, 28, 28), dtype=np.uint8),
            "train_labels": rng.integers(0, 2, (5, 1), dtype=np.int64),
        }
        np.savez(tmp_path, **data)  # type: ignore

        dataset = VisionDataset.lazy(tmp_path)
        assert len(dataset) == 5

        img, label = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape[0] == 1
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.long


@pytest.mark.unit
def test_vision_dataset_lazy_rgb() -> None:
    """Test VisionDataset.lazy() with RGB images."""
    rng = np.random.default_rng(seed=42)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "dummy_rgb.npz"
        data = {
            "train_images": rng.integers(0, 255, (5, 28, 28, 3), dtype=np.uint8),
            "train_labels": rng.integers(0, 2, (5, 1), dtype=np.int64),
        }
        np.savez(tmp_path, **data)  # type: ignore

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
def test_vision_dataset_lazy_grayscale_2d() -> None:
    """Test VisionDataset.lazy() with 2D grayscale images."""
    rng = np.random.default_rng(seed=42)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "dummy_gray.npz"
        data = {
            "train_images": rng.integers(0, 255, (5, 28, 28), dtype=np.uint8),
            "train_labels": rng.integers(0, 2, (5, 1), dtype=np.int64),
        }
        np.savez(tmp_path, **data)  # type: ignore

        dataset = VisionDataset.lazy(tmp_path)
        img, _ = dataset[0]

        assert img.shape[0] == 1
        assert img.shape[1] == 28
        assert img.shape[2] == 28


@pytest.mark.unit
def test_create_temp_loader() -> None:
    """Test create_temp_loader returns a working DataLoader."""
    rng = np.random.default_rng(seed=42)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "dummy.npz"
        data = {
            "train_images": rng.integers(0, 255, (5, 28, 28), dtype=np.uint8),
            "train_labels": rng.integers(0, 2, (5, 1), dtype=np.int64),
        }
        np.savez(tmp_path, **data)  # type: ignore

        loader = create_temp_loader(tmp_path, batch_size=2)
        batch_imgs, _ = next(iter(loader))
        assert batch_imgs.shape[0] <= 2
        assert batch_imgs.shape[1] == 1


@pytest.mark.unit
def test_create_temp_loader_rgb() -> None:
    """Test create_temp_loader with RGB images."""
    rng = np.random.default_rng(seed=42)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "dummy_rgb.npz"
        data = {
            "train_images": rng.integers(0, 255, (8, 32, 32, 3), dtype=np.uint8),
            "train_labels": rng.integers(0, 3, (8, 1), dtype=np.int64),
        }
        np.savez(tmp_path, **data)  # type: ignore

        loader = create_temp_loader(tmp_path, batch_size=4)
        batch_imgs, _ = next(iter(loader))

        assert batch_imgs.shape[0] <= 4
        assert batch_imgs.shape[1] == 3
        assert batch_imgs.shape[2] == 32
        assert batch_imgs.shape[3] == 32


# TESTS: get_dataloader
@pytest.mark.unit
def test_get_dataloaders_convenience_function(
    mock_cfg: MagicMock, mock_metadata: MagicMock
) -> None:
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
            task_type="classification",
        )
        mock_factory.build.assert_called_once_with(is_optuna=False)

        assert train == mock_train
        assert val == mock_val
        assert test == mock_test


@pytest.mark.unit
def test_get_dataloaders_with_optuna_mode(mock_cfg: MagicMock, mock_metadata: MagicMock) -> None:
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
def test_balancing_sampler_missing_class_raises(
    mock_cfg: MagicMock, mock_metadata: MagicMock
) -> None:
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

        with pytest.raises(OrchardDatasetError, match="missing 1 of 3 classes"):
            factory._get_balancing_sampler(dataset)


@pytest.mark.unit
def test_balancing_sampler_all_classes_present(
    mock_cfg: MagicMock, mock_metadata: MagicMock
) -> None:
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


# MUTATION-RESILIENT: EXACT CONSTANTS AND LOGIC
@pytest.mark.unit
def test_optuna_worker_constants() -> None:
    """Verify Optuna worker cap constants have exact expected values."""
    from orchard.core.paths.constants import HIGHRES_THRESHOLD
    from orchard.data_handler.loader import (
        _OPTUNA_WORKERS_HIGHRES,
        _OPTUNA_WORKERS_LOWRES,
    )

    assert _OPTUNA_WORKERS_HIGHRES == 4
    assert _OPTUNA_WORKERS_LOWRES == 6
    assert HIGHRES_THRESHOLD == 224


@pytest.mark.unit
def test_min_split_samples_constant() -> None:
    """Verify MIN_SPLIT_SAMPLES floor constant (centralized in constants.py)."""
    from orchard.core.paths import MIN_SPLIT_SAMPLES

    assert MIN_SPLIT_SAMPLES == 10


@pytest.mark.unit
def test_default_healthcheck_batch_size_constant() -> None:
    """Verify _DEFAULT_HEALTHCHECK_BATCH_SIZE constant."""
    from orchard.data_handler.diagnostic.temp_loader import _DEFAULT_HEALTHCHECK_BATCH_SIZE

    assert _DEFAULT_HEALTHCHECK_BATCH_SIZE == 16


@pytest.mark.unit
def test_infra_kwargs_persistent_workers_true_when_not_optuna(
    mock_cfg: MagicMock, mock_metadata: MagicMock
) -> None:
    """Verify persistent_workers=True when workers > 0 and NOT Optuna."""
    mock_ds_meta = MagicMock(in_channels=1)

    with patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta):
        factory = DataLoaderFactory(
            mock_cfg.dataset,
            mock_cfg.training,
            mock_cfg.augmentation,
            4,  # num_workers > 0
            mock_metadata,
        )
        infra = factory._get_infrastructure_kwargs(is_optuna=False)
        assert infra["persistent_workers"] is True


@pytest.mark.unit
def test_infra_kwargs_persistent_workers_false_when_optuna(
    mock_cfg: MagicMock, mock_metadata: MagicMock
) -> None:
    """Verify persistent_workers=False in Optuna mode even with workers > 0."""
    mock_ds_meta = MagicMock(in_channels=1)

    with patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta):
        factory = DataLoaderFactory(
            mock_cfg.dataset,
            mock_cfg.training,
            mock_cfg.augmentation,
            4,  # num_workers > 0
            mock_metadata,
        )
        infra = factory._get_infrastructure_kwargs(is_optuna=True)
        assert infra["persistent_workers"] is False


@pytest.mark.unit
def test_infra_kwargs_persistent_workers_false_when_zero_workers(
    mock_cfg: MagicMock, mock_metadata: MagicMock
) -> None:
    """Verify persistent_workers=False when num_workers=0 even without Optuna."""
    mock_ds_meta = MagicMock(in_channels=1)

    with patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta):
        factory = DataLoaderFactory(
            mock_cfg.dataset,
            mock_cfg.training,
            mock_cfg.augmentation,
            0,
            mock_metadata,
        )
        infra = factory._get_infrastructure_kwargs(is_optuna=False)
        assert infra["persistent_workers"] is False


@pytest.mark.unit
def test_infra_kwargs_worker_init_fn_none_when_zero_workers(
    mock_cfg: MagicMock, mock_metadata: MagicMock
) -> None:
    """Verify worker_init_fn is None when num_workers=0."""
    mock_ds_meta = MagicMock(in_channels=1)

    with patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta):
        factory = DataLoaderFactory(
            mock_cfg.dataset,
            mock_cfg.training,
            mock_cfg.augmentation,
            0,
            mock_metadata,
        )
        infra = factory._get_infrastructure_kwargs(is_optuna=False)
        assert infra["worker_init_fn"] is None


@pytest.mark.unit
def test_infra_kwargs_worker_init_fn_set_when_workers_positive(
    mock_cfg: MagicMock, mock_metadata: MagicMock
) -> None:
    """Verify worker_init_fn is not None when num_workers > 0."""
    mock_ds_meta = MagicMock(in_channels=1)

    with patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta):
        factory = DataLoaderFactory(
            mock_cfg.dataset,
            mock_cfg.training,
            mock_cfg.augmentation,
            2,
            mock_metadata,
        )
        infra = factory._get_infrastructure_kwargs(is_optuna=False)
        assert infra["worker_init_fn"] is not None


@pytest.mark.unit
def test_optuna_workers_capped_lowres(mock_cfg: MagicMock, mock_metadata: MagicMock) -> None:
    """Verify Optuna caps workers to 6 for low resolution."""
    mock_ds_meta = MagicMock(in_channels=1)
    mock_cfg.dataset.resolution = 28

    with patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta):
        factory = DataLoaderFactory(
            mock_cfg.dataset,
            mock_cfg.training,
            mock_cfg.augmentation,
            10,  # higher than cap
            mock_metadata,
        )
        infra = factory._get_infrastructure_kwargs(is_optuna=True)
        assert infra["num_workers"] == 6


@pytest.mark.unit
def test_optuna_workers_capped_highres(
    mock_cfg_high_res: MagicMock, mock_metadata: MagicMock
) -> None:
    """Verify Optuna caps workers to 4 for high resolution (>= 224)."""
    mock_ds_meta = MagicMock(in_channels=1)

    with patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta):
        factory = DataLoaderFactory(
            mock_cfg_high_res.dataset,
            mock_cfg_high_res.training,
            mock_cfg_high_res.augmentation,
            10,
            mock_metadata,
        )
        infra = factory._get_infrastructure_kwargs(is_optuna=True)
        assert infra["num_workers"] == 4


@pytest.mark.unit
def test_build_train_loader_shuffle_and_drop_last(
    mock_cfg_no_sampler: MagicMock, mock_metadata: MagicMock
) -> None:
    """Verify train_loader has shuffle=True (no sampler) and drop_last=True."""
    mock_ds_meta = MagicMock(in_channels=1)

    with patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta):
        with patch(
            "orchard.data_handler.loader.get_pipeline_transforms",
            lambda aug_cfg, img_size, meta, **kw: (lambda x: x, lambda x: x),
        ):

            class FakeDS:
                def __init__(self, **kwargs: object) -> None:
                    self.labels = np.array([0, 1, 0, 1])

                @classmethod
                def lazy(cls: Any, **kwargs: object) -> None:
                    return cls()  # type: ignore

                def __len__(self) -> None:
                    return 4  # type: ignore

            with patch("orchard.data_handler.loader.VisionDataset", FakeDS):
                factory = DataLoaderFactory(
                    mock_cfg_no_sampler.dataset,
                    mock_cfg_no_sampler.training,
                    mock_cfg_no_sampler.augmentation,
                    mock_cfg_no_sampler.num_workers,
                    mock_metadata,
                )
                train, val, test = factory.build()

                # No sampler → shuffle must be True
                # drop_last must be True for train
                # DataLoader doesn't expose shuffle directly, but we can verify via sampler type
                from torch.utils.data import SequentialSampler

                assert not isinstance(train.sampler, SequentialSampler)  # shuffle=True ≠ sequential
                assert val.sampler is not None  # shuffle=False → SequentialSampler
                assert isinstance(val.sampler, SequentialSampler)
                assert isinstance(test.sampler, SequentialSampler)


@pytest.mark.unit
def test_build_train_loader_no_shuffle_with_sampler(
    mock_cfg: MagicMock, mock_metadata: MagicMock
) -> None:
    """Verify train_loader has shuffle=False when WeightedRandomSampler is active."""
    mock_ds_meta = MagicMock(in_channels=1)

    with patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta):
        with patch(
            "orchard.data_handler.loader.get_pipeline_transforms",
            lambda aug_cfg, img_size, meta, **kw: (lambda x: x, lambda x: x),
        ):

            class FakeDS:
                def __init__(self, **kwargs: object) -> None:
                    self.labels = np.array([0, 1, 0, 1])

                @classmethod
                def lazy(cls: Any, **kwargs: object) -> None:
                    return cls()  # type: ignore

                def __len__(self) -> None:
                    return 4  # type: ignore

            with patch("orchard.data_handler.loader.VisionDataset", FakeDS):
                factory = DataLoaderFactory(
                    mock_cfg.dataset,
                    mock_cfg.training,
                    mock_cfg.augmentation,
                    mock_cfg.num_workers,
                    mock_metadata,
                )
                train, _, _ = factory.build()

                from torch.utils.data import WeightedRandomSampler

                assert isinstance(train.sampler, WeightedRandomSampler)


@pytest.mark.unit
def test_balancing_sampler_replacement_true(mock_cfg: MagicMock, mock_metadata: MagicMock) -> None:
    """Verify WeightedRandomSampler uses replacement=True."""
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
        assert sampler.replacement is True


@pytest.mark.unit
def test_balancing_sampler_num_samples(mock_cfg: MagicMock, mock_metadata: MagicMock) -> None:
    """Verify WeightedRandomSampler num_samples equals dataset size."""
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
        dataset.labels = np.array([0, 1, 0, 1, 0, 1])
        mock_cfg.dataset.num_classes = 2

        sampler = factory._get_balancing_sampler(dataset)
        assert sampler.num_samples == 6  # type: ignore


@pytest.mark.unit
def test_balancing_sampler_inverse_frequency_weights(
    mock_cfg: MagicMock, mock_metadata: MagicMock
) -> None:
    """Verify class weights are 1/count (inverse frequency)."""
    mock_ds_meta = MagicMock(in_channels=1)

    with patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta):
        factory = DataLoaderFactory(
            mock_cfg.dataset,
            mock_cfg.training,
            mock_cfg.augmentation,
            mock_cfg.num_workers,
            mock_metadata,
        )

        # 3 samples of class 0, 1 sample of class 1
        dataset = MagicMock()
        dataset.labels = np.array([0, 0, 0, 1])
        mock_cfg.dataset.num_classes = 2

        sampler = factory._get_balancing_sampler(dataset)
        weights = list(sampler.weights)  # type: ignore

        # class 0: count=3, weight=1/3; class 1: count=1, weight=1.0
        assert weights[0] == pytest.approx(1.0 / 3)
        assert weights[1] == pytest.approx(1.0 / 3)
        assert weights[2] == pytest.approx(1.0 / 3)
        assert weights[3] == pytest.approx(1.0)


@pytest.mark.unit
def test_build_sub_samples_calculation(mock_metadata: MagicMock) -> None:
    """Verify val/test sub_samples = max(MIN_SPLIT_SAMPLES, max_samples * val_ratio)."""
    mock_ds_meta = MagicMock(in_channels=1)
    cfg = MagicMock()
    cfg.dataset.dataset_name = "mock_dataset"
    cfg.dataset.use_weighted_sampler = False
    cfg.dataset.max_samples = 100
    cfg.dataset.val_ratio = 0.15
    cfg.dataset.resolution = 28
    cfg.dataset.lazy_loading = True
    cfg.training.batch_size = 2
    cfg.num_workers = 0

    with patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta):
        with patch(
            "orchard.data_handler.loader.get_pipeline_transforms",
            lambda aug_cfg, img_size, meta, **kw: (lambda x: x, lambda x: x),
        ):
            build_calls = []

            class SpyDS:
                def __init__(self, **kwargs: object) -> None:
                    self.labels = np.array([0, 1])
                    self.max_samples = kwargs.get("max_samples")
                    build_calls.append(kwargs)

                @classmethod
                def lazy(cls: Any, **kwargs: object) -> None:
                    return cls(**kwargs)  # type: ignore

                def __len__(self) -> None:
                    return 4  # type: ignore

            with patch("orchard.data_handler.loader.VisionDataset", SpyDS):
                factory = DataLoaderFactory(
                    cfg.dataset,
                    cfg.training,
                    cfg.augmentation,
                    cfg.num_workers,
                    mock_metadata,
                )
                factory.build()

                # Call 0=train, 1=val, 2=test
                assert build_calls[0]["max_samples"] == 100
                expected_sub = max(10, int(100 * 0.15))  # max(10, 15) = 15
                assert build_calls[1]["max_samples"] == expected_sub
                assert build_calls[2]["max_samples"] == expected_sub


@pytest.mark.unit
def test_build_sub_samples_floor(mock_metadata: MagicMock) -> None:
    """Verify sub_samples floors at MIN_SPLIT_SAMPLES when max_samples * val_ratio < 10."""
    mock_ds_meta = MagicMock(in_channels=1)
    cfg = MagicMock()
    cfg.dataset.dataset_name = "mock_dataset"
    cfg.dataset.use_weighted_sampler = False
    cfg.dataset.max_samples = 20
    cfg.dataset.val_ratio = 0.1  # 20 * 0.1 = 2, should be floored to 10
    cfg.dataset.resolution = 28
    cfg.dataset.lazy_loading = True
    cfg.training.batch_size = 2
    cfg.num_workers = 0

    with patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta):
        with patch(
            "orchard.data_handler.loader.get_pipeline_transforms",
            lambda aug_cfg, img_size, meta, **kw: (lambda x: x, lambda x: x),
        ):
            build_calls = []

            class SpyDS:
                def __init__(self, **kwargs: object) -> None:
                    self.labels = np.array([0, 1])
                    build_calls.append(kwargs)

                @classmethod
                def lazy(cls: Any, **kwargs: object) -> None:
                    return cls(**kwargs)  # type: ignore

                def __len__(self) -> None:
                    return 4  # type: ignore

            with patch("orchard.data_handler.loader.VisionDataset", SpyDS):
                factory = DataLoaderFactory(
                    cfg.dataset,
                    cfg.training,
                    cfg.augmentation,
                    cfg.num_workers,
                    mock_metadata,
                )
                factory.build()

                assert build_calls[1]["max_samples"] == 10  # floor
                assert build_calls[2]["max_samples"] == 10


@pytest.mark.unit
def test_build_sub_samples_none_when_no_max_samples(
    mock_cfg_no_sampler: MagicMock, mock_metadata: MagicMock
) -> None:
    """Verify val/test max_samples is None when dataset max_samples is None."""
    mock_ds_meta = MagicMock(in_channels=1)

    with patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta):
        with patch(
            "orchard.data_handler.loader.get_pipeline_transforms",
            lambda aug_cfg, img_size, meta, **kw: (lambda x: x, lambda x: x),
        ):
            build_calls = []

            class SpyDS:
                def __init__(self, **kwargs: object) -> None:
                    self.labels = np.array([0, 1])
                    build_calls.append(kwargs)

                @classmethod
                def lazy(cls: Any, **kwargs: object) -> None:
                    return cls(**kwargs)  # type: ignore

                def __len__(self) -> None:
                    return 4  # type: ignore

            with patch("orchard.data_handler.loader.VisionDataset", SpyDS):
                factory = DataLoaderFactory(
                    mock_cfg_no_sampler.dataset,
                    mock_cfg_no_sampler.training,
                    mock_cfg_no_sampler.augmentation,
                    mock_cfg_no_sampler.num_workers,
                    mock_metadata,
                )
                factory.build()

                assert build_calls[1]["max_samples"] is None
                assert build_calls[2]["max_samples"] is None


@pytest.mark.unit
def test_create_temp_loader_uses_default_batch_size() -> None:
    """Verify create_temp_loader uses _DEFAULT_HEALTHCHECK_BATCH_SIZE when not specified."""
    rng = np.random.default_rng(seed=42)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "dummy.npz"
        data = {
            "train_images": rng.integers(0, 255, (20, 28, 28), dtype=np.uint8),
            "train_labels": rng.integers(0, 2, (20, 1), dtype=np.int64),
        }
        np.savez(tmp_path, **data)  # type: ignore

        loader = create_temp_loader(tmp_path)
        assert loader.batch_size == 16


@pytest.mark.unit
def test_build_uses_lazy_when_lazy_loading_true(mock_metadata: MagicMock) -> None:
    """Verify build() calls VisionDataset.lazy when lazy_loading=True."""
    mock_ds_meta = MagicMock(in_channels=1)
    cfg = MagicMock()
    cfg.dataset.dataset_name = "mock_dataset"
    cfg.dataset.use_weighted_sampler = False
    cfg.dataset.max_samples = None
    cfg.dataset.resolution = 28
    cfg.dataset.lazy_loading = True
    cfg.training.batch_size = 2
    cfg.num_workers = 0

    with patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta):
        with patch(
            "orchard.data_handler.loader.get_pipeline_transforms",
            lambda aug_cfg, img_size, meta, **kw: (lambda x: x, lambda x: x),
        ):
            lazy_called = []
            from_npz_called = []

            class SpyDS:
                def __init__(self, **kwargs: object) -> None:
                    self.labels = np.array([0, 1])

                @classmethod
                def lazy(cls: Any, **kwargs: object) -> None:
                    lazy_called.append(True)
                    return cls(**kwargs)  # type: ignore

                @classmethod
                def from_npz(cls: Any, **kwargs: object) -> None:
                    from_npz_called.append(True)
                    return cls(**kwargs)  # type: ignore

                def __len__(self) -> None:
                    return 4  # type: ignore

            with patch("orchard.data_handler.loader.VisionDataset", SpyDS):
                factory = DataLoaderFactory(
                    cfg.dataset, cfg.training, cfg.augmentation, cfg.num_workers, mock_metadata
                )
                factory.build()

                assert len(lazy_called) == 3
                assert len(from_npz_called) == 0


@pytest.mark.unit
def test_build_uses_from_npz_when_lazy_loading_false(mock_metadata: MagicMock) -> None:
    """Verify build() calls VisionDataset.from_npz when lazy_loading=False."""
    mock_ds_meta = MagicMock(in_channels=1)
    cfg = MagicMock()
    cfg.dataset.dataset_name = "mock_dataset"
    cfg.dataset.use_weighted_sampler = False
    cfg.dataset.max_samples = None
    cfg.dataset.resolution = 28
    cfg.dataset.lazy_loading = False
    cfg.training.batch_size = 2
    cfg.num_workers = 0

    with patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta):
        with patch(
            "orchard.data_handler.loader.get_pipeline_transforms",
            lambda aug_cfg, img_size, meta, **kw: (lambda x: x, lambda x: x),
        ):
            lazy_called = []
            from_npz_called = []

            class SpyDS:
                def __init__(self, **kwargs: object) -> None:
                    self.labels = np.array([0, 1])

                @classmethod
                def lazy(cls: Any, **kwargs: object) -> None:
                    lazy_called.append(True)
                    return cls(**kwargs)  # type: ignore

                @classmethod
                def from_npz(cls: Any, **kwargs: object) -> None:
                    from_npz_called.append(True)
                    return cls(**kwargs)  # type: ignore

                def __len__(self) -> None:
                    return 4  # type: ignore

            with patch("orchard.data_handler.loader.VisionDataset", SpyDS):
                factory = DataLoaderFactory(
                    cfg.dataset, cfg.training, cfg.augmentation, cfg.num_workers, mock_metadata
                )
                factory.build()

                assert len(from_npz_called) == 3
                assert len(lazy_called) == 0


# MUTATION TESTS: argument wiring and DataLoader construction
@pytest.fixture
def _spy_ds_factory() -> None:
    """Factory that yields a SpyDS class capturing all build calls."""

    def _make() -> None:
        calls = []

        class SpyDS:
            def __init__(self, **kwargs: object) -> None:
                self.labels = np.array([0, 1, 0, 1])
                calls.append(kwargs)

            @classmethod
            def lazy(cls: Any, **kwargs: object) -> None:
                return cls(**kwargs)  # type: ignore

            @classmethod
            def from_npz(cls: Any, **kwargs: object) -> None:
                return cls(**kwargs)  # type: ignore

            def __len__(self) -> None:
                return 4  # type: ignore

        return SpyDS, calls  # type: ignore

    return _make  # type: ignore


def _build_factory(cfg: Any, mock_metadata: MagicMock, num_workers: Any = 0) -> None:
    """Helper to build a DataLoaderFactory with standard mocking."""
    mock_ds_meta = MagicMock(in_channels=1)

    with patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta):
        return DataLoaderFactory(  # type: ignore
            cfg.dataset, cfg.training, cfg.augmentation, num_workers, mock_metadata
        )


@pytest.mark.unit
def test_build_passes_correct_split_names(  # type: ignore
    mock_cfg_no_sampler: MagicMock, mock_metadata: MagicMock, _spy_ds_factory
) -> None:
    """Verify build() passes exact split names ('train', 'val', 'test')."""
    SpyDS, calls = _spy_ds_factory()
    mock_ds_meta = MagicMock(in_channels=1)

    with (
        patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta),
        patch(
            "orchard.data_handler.loader.get_pipeline_transforms",
            lambda aug_cfg, img_size, meta, **kw: (lambda x: x, lambda x: x),
        ),
        patch("orchard.data_handler.loader.VisionDataset", SpyDS),
    ):
        factory = DataLoaderFactory(
            mock_cfg_no_sampler.dataset,
            mock_cfg_no_sampler.training,
            mock_cfg_no_sampler.augmentation,
            mock_cfg_no_sampler.num_workers,
            mock_metadata,
        )
        factory.build()

        assert calls[0]["split"] == "train"
        assert calls[1]["split"] == "val"
        assert calls[2]["split"] == "test"


@pytest.mark.unit
def test_build_passes_path_and_seed(  # type: ignore
    mock_cfg_no_sampler: MagicMock, mock_metadata: MagicMock, _spy_ds_factory
) -> None:
    """Verify build() passes metadata path and training seed to each split."""
    SpyDS, calls = _spy_ds_factory()
    mock_ds_meta = MagicMock(in_channels=1)

    with (
        patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta),
        patch(
            "orchard.data_handler.loader.get_pipeline_transforms",
            lambda aug_cfg, img_size, meta, **kw: (lambda x: x, lambda x: x),
        ),
        patch("orchard.data_handler.loader.VisionDataset", SpyDS),
    ):
        factory = DataLoaderFactory(
            mock_cfg_no_sampler.dataset,
            mock_cfg_no_sampler.training,
            mock_cfg_no_sampler.augmentation,
            mock_cfg_no_sampler.num_workers,
            mock_metadata,
        )
        factory.build()

        for call in calls:
            assert "path" in call, "ds_params must include 'path'"
            assert call["path"] == mock_metadata.path
            assert "seed" in call, "ds_params must include 'seed'"
            assert call["seed"] == mock_cfg_no_sampler.training.seed


@pytest.mark.unit
def test_build_passes_transforms(  # type: ignore
    mock_cfg_no_sampler: MagicMock, mock_metadata: MagicMock, _spy_ds_factory
) -> None:
    """Verify build() passes train transform to train split and val transform to val/test."""
    SpyDS, calls = _spy_ds_factory()
    mock_ds_meta = MagicMock(in_channels=1)
    sentinel_train = object()
    sentinel_val = object()

    with (
        patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta),
        patch(
            "orchard.data_handler.loader.get_pipeline_transforms",
            lambda aug_cfg, img_size, meta, **kw: (sentinel_train, sentinel_val),
        ),
        patch("orchard.data_handler.loader.VisionDataset", SpyDS),
    ):
        factory = DataLoaderFactory(
            mock_cfg_no_sampler.dataset,
            mock_cfg_no_sampler.training,
            mock_cfg_no_sampler.augmentation,
            mock_cfg_no_sampler.num_workers,
            mock_metadata,
        )
        factory.build()

        assert calls[0]["transform"] is sentinel_train
        assert calls[1]["transform"] is sentinel_val
        assert calls[2]["transform"] is sentinel_val


@pytest.mark.unit
def test_build_train_loader_drop_last_true(  # type: ignore
    mock_cfg_no_sampler: MagicMock, mock_metadata: MagicMock, _spy_ds_factory
) -> None:
    """Verify train DataLoader has drop_last=True."""
    SpyDS, _ = _spy_ds_factory()
    mock_ds_meta = MagicMock(in_channels=1)

    with (
        patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta),
        patch(
            "orchard.data_handler.loader.get_pipeline_transforms",
            lambda aug_cfg, img_size, meta, **kw: (lambda x: x, lambda x: x),
        ),
        patch("orchard.data_handler.loader.VisionDataset", SpyDS),
    ):
        factory = DataLoaderFactory(
            mock_cfg_no_sampler.dataset,
            mock_cfg_no_sampler.training,
            mock_cfg_no_sampler.augmentation,
            mock_cfg_no_sampler.num_workers,
            mock_metadata,
        )
        train, _, _ = factory.build()

        assert train.drop_last is True


@pytest.mark.unit
def test_build_val_test_loaders_have_infra_kwargs(  # type: ignore
    mock_cfg_no_sampler: MagicMock, mock_metadata: MagicMock, _spy_ds_factory
) -> None:
    """Verify val/test DataLoaders receive infrastructure kwargs (num_workers, pin_memory)."""
    SpyDS, _ = _spy_ds_factory()
    mock_ds_meta = MagicMock(in_channels=1)
    num_workers = 2  # non-default to detect missing **infra_kwargs

    with (
        patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta),
        patch(
            "orchard.data_handler.loader.get_pipeline_transforms",
            lambda aug_cfg, img_size, meta, **kw: (lambda x: x, lambda x: x),
        ),
        patch("orchard.data_handler.loader.VisionDataset", SpyDS),
    ):
        factory = DataLoaderFactory(
            mock_cfg_no_sampler.dataset,
            mock_cfg_no_sampler.training,
            mock_cfg_no_sampler.augmentation,
            num_workers,
            mock_metadata,
        )
        _, val, test = factory.build()

        assert val.num_workers == num_workers
        assert test.num_workers == num_workers


@pytest.mark.unit
def test_infra_kwargs_boundary_num_workers_one(
    mock_cfg: MagicMock, mock_metadata: MagicMock
) -> None:
    """Verify worker_init_fn and persistent_workers with num_workers=1 (boundary)."""
    mock_ds_meta = MagicMock(in_channels=1)

    with patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta):
        factory = DataLoaderFactory(
            mock_cfg.dataset,
            mock_cfg.training,
            mock_cfg.augmentation,
            1,
            mock_metadata,
        )
        infra = factory._get_infrastructure_kwargs(is_optuna=False)
        assert infra["worker_init_fn"] is not None
        assert infra["persistent_workers"] is True


@pytest.mark.unit
def test_get_transformation_pipelines_passes_correct_args(
    mock_cfg_no_sampler: MagicMock, mock_metadata: MagicMock
) -> None:
    """Verify _get_transformation_pipelines passes all config args to get_pipeline_transforms."""
    mock_ds_meta = MagicMock(in_channels=1)
    captured = {}

    def spy_transforms(aug_cfg: Any, img_size: Any, meta: Any, **kw: object) -> None:
        captured["aug_cfg"] = aug_cfg
        captured["img_size"] = img_size
        captured["meta"] = meta
        captured.update(kw)
        return (lambda x: x, lambda x: x)  # type: ignore

    with (
        patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta),
        patch("orchard.data_handler.loader.get_pipeline_transforms", spy_transforms),
    ):
        factory = DataLoaderFactory(
            mock_cfg_no_sampler.dataset,
            mock_cfg_no_sampler.training,
            mock_cfg_no_sampler.augmentation,
            mock_cfg_no_sampler.num_workers,
            mock_metadata,
        )
        factory._get_transformation_pipelines()

        assert captured["aug_cfg"] is factory.aug_cfg
        assert captured["img_size"] == mock_cfg_no_sampler.dataset.img_size
        assert captured["meta"] is factory.ds_meta
        assert captured["force_rgb"] == mock_cfg_no_sampler.dataset.force_rgb
        assert captured["norm_mean"] == mock_cfg_no_sampler.dataset.mean
        assert captured["norm_std"] == mock_cfg_no_sampler.dataset.std


@pytest.mark.unit
def test_init_stores_ds_meta_from_registry(mock_cfg: MagicMock, mock_metadata: MagicMock) -> None:
    """Verify __init__ stores ds_meta from DatasetRegistryWrapper.get_dataset."""
    mock_ds_meta = MagicMock(in_channels=1)

    with patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta) as mock_get:
        factory = DataLoaderFactory(
            mock_cfg.dataset,
            mock_cfg.training,
            mock_cfg.augmentation,
            mock_cfg.num_workers,
            mock_metadata,
        )

        assert factory.ds_meta is mock_ds_meta
        mock_get.assert_called_once_with(mock_cfg.dataset.dataset_name)


@pytest.mark.unit
def test_init_stores_aug_cfg(mock_cfg: MagicMock, mock_metadata: MagicMock) -> None:
    """Verify __init__ stores aug_cfg correctly."""
    mock_ds_meta = MagicMock(in_channels=1)

    with patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta):
        factory = DataLoaderFactory(
            mock_cfg.dataset,
            mock_cfg.training,
            mock_cfg.augmentation,
            mock_cfg.num_workers,
            mock_metadata,
        )

        assert factory.aug_cfg is mock_cfg.augmentation


@pytest.mark.unit
def test_build_passes_is_optuna_to_infra_kwargs(  # type: ignore
    mock_cfg_no_sampler: MagicMock, mock_metadata: MagicMock, _spy_ds_factory
) -> None:
    """Verify build() passes is_optuna to _get_infrastructure_kwargs."""
    SpyDS, _ = _spy_ds_factory()
    mock_ds_meta = MagicMock(in_channels=1)

    with (
        patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta),
        patch(
            "orchard.data_handler.loader.get_pipeline_transforms",
            lambda aug_cfg, img_size, meta, **kw: (lambda x: x, lambda x: x),
        ),
        patch("orchard.data_handler.loader.VisionDataset", SpyDS),
    ):
        factory = DataLoaderFactory(
            mock_cfg_no_sampler.dataset,
            mock_cfg_no_sampler.training,
            mock_cfg_no_sampler.augmentation,
            4,
            mock_metadata,
        )

        with patch.object(
            factory, "_get_infrastructure_kwargs", wraps=factory._get_infrastructure_kwargs
        ) as spy:
            factory.build(is_optuna=True)
            spy.assert_called_once_with(is_optuna=True)


@pytest.mark.unit
def test_build_train_loader_receives_infra_kwargs(  # type: ignore
    mock_cfg_no_sampler: MagicMock, mock_metadata: MagicMock, _spy_ds_factory
) -> None:
    """Verify train DataLoader receives infra kwargs (num_workers, pin_memory)."""
    SpyDS, _ = _spy_ds_factory()
    mock_ds_meta = MagicMock(in_channels=1)
    num_workers = 2  # non-default to detect missing **infra_kwargs

    with (
        patch.object(DatasetRegistryWrapper, "get_dataset", return_value=mock_ds_meta),
        patch(
            "orchard.data_handler.loader.get_pipeline_transforms",
            lambda aug_cfg, img_size, meta, **kw: (lambda x: x, lambda x: x),
        ),
        patch("orchard.data_handler.loader.VisionDataset", SpyDS),
    ):
        factory = DataLoaderFactory(
            mock_cfg_no_sampler.dataset,
            mock_cfg_no_sampler.training,
            mock_cfg_no_sampler.augmentation,
            num_workers,
            mock_metadata,
        )
        train, _, _ = factory.build()

        assert train.num_workers == num_workers


# DETECTION TASK TYPE TESTS


@pytest.mark.unit
@patch("orchard.data_handler.loader.DatasetRegistryWrapper")
@patch("orchard.data_handler.detection_dataset.DetectionDataset.from_npz")
@patch("orchard.data_handler.loader.get_pipeline_transforms")
def test_detection_task_type_uses_collate_fn(
    mock_transforms: MagicMock,
    mock_det_from_npz: MagicMock,
    mock_wrapper: MagicMock,
    mock_cfg_no_sampler: Any,
    mock_metadata: Any,
) -> None:
    """Detection task_type passes detection_collate_fn to DataLoaders."""
    mock_transforms.return_value = (MagicMock(), MagicMock())
    mock_det_from_npz.return_value = MagicMock()
    mock_metadata.annotation_path = "/fake/annotations.npz"

    factory = DataLoaderFactory(
        mock_cfg_no_sampler.dataset,
        mock_cfg_no_sampler.training,
        mock_cfg_no_sampler.augmentation,
        0,
        mock_metadata,
        task_type="detection",
    )

    with patch("orchard.data_handler.loader.DataLoader") as mock_dl:
        mock_dl.return_value = MagicMock()
        factory.build()

        from orchard.data_handler.collate import detection_collate_fn

        for call in mock_dl.call_args_list:
            assert (
                call.kwargs.get("collate_fn") is detection_collate_fn
                or call[1].get("collate_fn") is detection_collate_fn
            )


@pytest.mark.unit
@patch("orchard.data_handler.loader.DatasetRegistryWrapper")
@patch("orchard.data_handler.detection_dataset.DetectionDataset.from_npz")
@patch("orchard.data_handler.loader.get_pipeline_transforms")
def test_detection_task_type_skips_sampler(
    mock_transforms: MagicMock,
    mock_det_from_npz: MagicMock,
    mock_wrapper: MagicMock,
    mock_cfg: Any,
    mock_metadata: Any,
) -> None:
    """Detection task_type skips WeightedRandomSampler even when enabled."""
    mock_transforms.return_value = (MagicMock(), MagicMock())
    mock_det_from_npz.return_value = MagicMock()
    mock_metadata.annotation_path = "/fake/annotations.npz"

    factory = DataLoaderFactory(
        mock_cfg.dataset,
        mock_cfg.training,
        mock_cfg.augmentation,
        0,
        mock_metadata,
        task_type="detection",
    )

    with patch("orchard.data_handler.loader.DataLoader") as mock_dl:
        mock_dl.return_value = MagicMock()
        factory.build()

        train_call = mock_dl.call_args_list[0]
        assert train_call.kwargs.get("sampler") is None or train_call[1].get("sampler") is None


@pytest.mark.unit
@patch("orchard.data_handler.loader.DatasetRegistryWrapper")
@patch("orchard.data_handler.loader.VisionDataset")
@patch("orchard.data_handler.loader.get_pipeline_transforms")
def test_classification_task_type_no_collate_fn(
    mock_transforms: MagicMock,
    mock_ds_cls: MagicMock,
    mock_wrapper: MagicMock,
    mock_cfg_no_sampler: Any,
    mock_metadata: Any,
) -> None:
    """Classification task_type passes collate_fn=None (default stacking)."""
    mock_transforms.return_value = (MagicMock(), MagicMock())
    mock_ds = MagicMock(spec=VisionDataset)
    mock_ds.labels = np.array([0, 1, 0, 1])
    mock_ds_cls.lazy.return_value = mock_ds

    factory = DataLoaderFactory(
        mock_cfg_no_sampler.dataset,
        mock_cfg_no_sampler.training,
        mock_cfg_no_sampler.augmentation,
        0,
        mock_metadata,
        task_type="classification",
    )

    with patch("orchard.data_handler.loader.DataLoader") as mock_dl:
        mock_dl.return_value = MagicMock()
        factory.build()

        for call in mock_dl.call_args_list:
            assert call.kwargs.get("collate_fn") is None or call[1].get("collate_fn") is None


@pytest.mark.unit
@patch("orchard.data_handler.loader.DatasetRegistryWrapper")
@patch("orchard.data_handler.loader.VisionDataset")
@patch("orchard.data_handler.loader.get_pipeline_transforms")
def test_task_type_stored_on_factory(
    mock_transforms: MagicMock,
    mock_ds_cls: MagicMock,
    mock_wrapper: MagicMock,
    mock_cfg_no_sampler: Any,
    mock_metadata: Any,
) -> None:
    """task_type is stored as _task_type attribute."""
    mock_transforms.return_value = (MagicMock(), MagicMock())

    factory = DataLoaderFactory(
        mock_cfg_no_sampler.dataset,
        mock_cfg_no_sampler.training,
        mock_cfg_no_sampler.augmentation,
        0,
        mock_metadata,
        task_type="detection",
    )

    assert factory._task_type == "detection"


# MAIN TEST RUNNER
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
