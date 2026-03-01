"""
Data Loader Orchestration Module.

Provides the DataLoaderFactory for building PyTorch DataLoaders with advanced
features: class balancing via WeightedRandomSampler, hardware-aware configuration
(workers, pinned memory), and Optuna-compatible resource management.

Architecture:

- Factory Pattern: Centralizes DataLoader construction logic
- Hardware Optimization: Adaptive workers and memory pinning (CUDA/MPS)
- Class Balancing: WeightedRandomSampler for imbalanced datasets
- Optuna Integration: Resource-conservative settings for hyperparameter tuning

Key Components:

- ``DataLoaderFactory``: Main orchestrator for train/val/test loader creation
- ``get_dataloaders``: Convenience function for direct loader retrieval
- ``create_temp_loader``: Quick DataLoader builder for health checks

Example:
    >>> from orchard.data_handler import get_dataloaders, load_dataset
    >>> data = load_dataset(ds_meta)
    >>> train_loader, val_loader, test_loader = get_dataloaders(
    ...     data, cfg.dataset, cfg.training, cfg.augmentation, cfg.num_workers
    ... )
    >>> print(f"Batches: {len(train_loader)}")
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..core import (
    LOGGER_NAME,
    AugmentationConfig,
    DatasetConfig,
    DatasetRegistryWrapper,
    LogStyle,
    TrainingConfig,
    worker_init_fn,
)
from .dataset import VisionDataset
from .fetcher import DatasetData
from .transforms import get_pipeline_transforms

# Optuna mode: cap workers to prevent file descriptor exhaustion during trials
_OPTUNA_WORKERS_HIGHRES = 4  # Max workers for resolution >= _HIGHRES_THRESHOLD
_OPTUNA_WORKERS_LOWRES = 6  # Max workers for resolution < _HIGHRES_THRESHOLD
_HIGHRES_THRESHOLD = 224  # Resolution boundary for worker tuning

_MIN_SUBSAMPLED_SPLIT = 10  # Floor for val/test splits under max_samples
_DEFAULT_HEALTHCHECK_BATCH_SIZE = 16  # Batch size for create_temp_loader


# DATALOADER FACTORY
class DataLoaderFactory:
    """
    Orchestrates the creation of optimized PyTorch DataLoaders.

    This factory centralizes the configuration of training, validation, and
    testing pipelines. It ensures that data transformations, class balancing,
    and hardware settings are synchronized across all splits.

    Attributes:
        dataset_cfg (DatasetConfig): Dataset sub-config.
        training_cfg (TrainingConfig): Training sub-config.
        aug_cfg (AugmentationConfig): Augmentation sub-config.
        num_workers (int): Resolved worker count from hardware config.
        metadata (DatasetData): Data path and raw format information.
        ds_meta (DatasetMetadata): Official dataset registry specifications.
        logger (logging.Logger): Module-specific logger.
    """

    def __init__(
        self,
        dataset_cfg: DatasetConfig,
        training_cfg: TrainingConfig,
        aug_cfg: AugmentationConfig,
        num_workers: int,
        metadata: DatasetData,
    ) -> None:
        """
        Initializes the factory with environment and dataset metadata.

        Args:
            dataset_cfg: Dataset sub-config (splits, classes, resolution).
            training_cfg: Training sub-config (batch size, seed).
            aug_cfg: Augmentation sub-config (transforms pipeline).
            num_workers: Resolved worker count from hardware config.
            metadata: Metadata from the data fetcher/downloader.
        """
        self.dataset_cfg = dataset_cfg
        self.training_cfg = training_cfg
        self.aug_cfg = aug_cfg
        self._num_workers = num_workers
        self.metadata = metadata

        wrapper = DatasetRegistryWrapper(resolution=dataset_cfg.resolution)
        self.ds_meta = wrapper.get_dataset(dataset_cfg.dataset_name)
        self.logger = logging.getLogger(LOGGER_NAME)

    def _get_transformation_pipelines(self) -> tuple[torch.nn.Module, torch.nn.Module]:
        """
        Retrieves specialized vision pipelines.

        Returns:
            A tuple containing (train_transform, val_transform).
        """
        return get_pipeline_transforms(self.aug_cfg, self.dataset_cfg.img_size, self.ds_meta)

    def _get_balancing_sampler(self, dataset: VisionDataset) -> WeightedRandomSampler | None:
        """
        Calculates class weights and builds a WeightedRandomSampler.

        This method addresses class imbalance by assigning higher sampling
        probabilities to under-represented classes.

        Args:
            dataset: The training dataset instance.

        Returns:
            A WeightedRandomSampler if enabled in config, otherwise None.
        """
        if not self.dataset_cfg.use_weighted_sampler:
            return None

        labels = dataset.labels.flatten()
        classes, counts = np.unique(labels, return_counts=True)

        # Guard: aggressive max_samples can eliminate entire classes
        expected_classes = self.dataset_cfg.num_classes
        if len(classes) < expected_classes:
            missing = sorted(set(range(expected_classes)) - set(classes))
            raise ValueError(
                f"Training set is missing {len(missing)} of {expected_classes} classes "
                f"{missing} after subsampling (max_samples={self.dataset_cfg.max_samples}). "
                f"Increase max_samples or disable use_weighted_sampler."
            )

        # Inverse frequency balancing: weight = 1 / frequency
        class_weights = 1.0 / counts
        weight_map: dict[int, float] = dict(zip(classes, class_weights))

        sample_weights = torch.tensor(
            [weight_map[int(label)] for label in labels], dtype=torch.float
        )

        self.logger.info(  # pragma: no mutant
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Balancing':<18}: WeightedRandomSampler generated"
        )
        return WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=len(sample_weights),
            replacement=True,
        )

    def _get_infrastructure_kwargs(self, is_optuna: bool = False) -> dict:
        """
        Determines hardware and process-level parameters for DataLoaders.

        Adjust workers and persistence for Optuna compatibility.

        Args:
            is_optuna: If True, optimize for trial stability over performance

        Returns:
            A dictionary of DataLoader arguments (num_workers, pin_memory, etc.).
        """
        num_workers = self._num_workers

        # OPTUNA MODE: Reduce workers to prevent file descriptor exhaustion
        if is_optuna:
            cap = (
                _OPTUNA_WORKERS_HIGHRES
                if self.dataset_cfg.resolution >= _HIGHRES_THRESHOLD
                else _OPTUNA_WORKERS_LOWRES
            )
            num_workers = min(num_workers, cap)

            self.logger.info(  # pragma: no mutant
                f"{LogStyle.INDENT}{LogStyle.ARROW} {'Optuna Workers':<18}: "
                f"{num_workers} (Resolution={self.dataset_cfg.resolution})"
            )

        # Hardware acceleration: Pin memory for CUDA or MPS
        has_cuda = torch.cuda.is_available()
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        return {
            "num_workers": num_workers,
            "pin_memory": has_cuda or has_mps,
            "worker_init_fn": worker_init_fn if num_workers > 0 else None,
            "persistent_workers": (num_workers > 0) and (not is_optuna),
        }

    def build(self, is_optuna: bool = False) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        Constructs and returns the full suite of DataLoaders.

        Assembles train/val/test splits with transforms, optional class
        balancing, and hardware-aware infrastructure settings.

        Args:
            is_optuna: If True, use memory-conservative settings for
                hyperparameter tuning (fewer workers, no persistent workers).

        Returns:
            A tuple of (train_loader, val_loader, test_loader).
        """
        # 1. Setup transforms
        train_trans, val_trans = self._get_transformation_pipelines()

        # 2. Instantiate Dataset splits (lazy=mmap, eager=full RAM copy)
        _build = VisionDataset.lazy if self.dataset_cfg.lazy_loading else VisionDataset.from_npz
        ds_params = {"path": self.metadata.path, "seed": self.training_cfg.seed}

        train_ds = _build(
            **ds_params,
            split="train",
            transform=train_trans,
            max_samples=self.dataset_cfg.max_samples,
        )

        # Proportional downsizing for validation/testing if max_samples is set
        sub_samples = None
        if self.dataset_cfg.max_samples:
            sub_samples = max(
                _MIN_SUBSAMPLED_SPLIT,
                int(self.dataset_cfg.max_samples * self.dataset_cfg.val_ratio),
            )

        val_ds = _build(**ds_params, split="val", transform=val_trans, max_samples=sub_samples)
        test_ds = _build(**ds_params, split="test", transform=val_trans, max_samples=sub_samples)

        # 3. Resolve Sampler and Infrastructure
        sampler = self._get_balancing_sampler(train_ds)
        infra_kwargs = self._get_infrastructure_kwargs(is_optuna=is_optuna)

        # 4. Construct DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size=self.training_cfg.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            drop_last=True,
            **infra_kwargs,
        )

        val_loader = DataLoader(
            val_ds, batch_size=self.training_cfg.batch_size, shuffle=False, **infra_kwargs
        )

        test_loader = DataLoader(
            test_ds, batch_size=self.training_cfg.batch_size, shuffle=False, **infra_kwargs
        )

        mode_str = "RGB" if self.ds_meta.in_channels == 3 else "Grayscale"
        optuna_str = " (Optuna)" if is_optuna else ""
        self.logger.info(  # pragma: no mutant
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'DataLoaders':<18}: "
            f"({mode_str}){optuna_str} → "
            f"Train:[{len(train_ds)}] Val:[{len(val_ds)}] Test:[{len(test_ds)}]"
        )

        return train_loader, val_loader, test_loader


def get_dataloaders(
    metadata: DatasetData,
    dataset_cfg: DatasetConfig,
    training_cfg: TrainingConfig,
    aug_cfg: AugmentationConfig,
    num_workers: int,
    is_optuna: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience function for creating train/val/test DataLoaders.

    Wraps DataLoaderFactory for streamlined loader construction with
    automatic class balancing, hardware optimization, and Optuna support.

    Args:
        metadata: Dataset metadata from load_dataset (paths, splits).
        dataset_cfg: Dataset sub-config (splits, classes, resolution).
        training_cfg: Training sub-config (batch size, seed).
        aug_cfg: Augmentation sub-config (transforms pipeline).
        num_workers: Resolved worker count from hardware config.
        is_optuna: If True, use memory-conservative settings for
            hyperparameter tuning.

    Returns:
        A 3-tuple of (train_loader, val_loader, test_loader).

    Example:
        >>> data = load_dataset(ds_meta)
        >>> loaders = get_dataloaders(
        ...     data, cfg.dataset, cfg.training, cfg.augmentation, cfg.num_workers
        ... )
    """
    factory = DataLoaderFactory(dataset_cfg, training_cfg, aug_cfg, num_workers, metadata)
    return factory.build(is_optuna=is_optuna)


# HEALTH UTILITIES
def create_temp_loader(
    dataset_path: Path, batch_size: int = _DEFAULT_HEALTHCHECK_BATCH_SIZE
) -> DataLoader:
    """
    Load a NPZ dataset lazily and return a DataLoader for health checks.

    This avoids loading the entire dataset into RAM at once, which is critical
    for large datasets (e.g., 224x224 images).
    """
    dataset = VisionDataset.lazy(dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader
