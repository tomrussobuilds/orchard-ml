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
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..core import (
    HIGHRES_THRESHOLD,
    LOGGER_NAME,
    AugmentationConfig,
    DatasetConfig,
    LogStyle,
    TrainingConfig,
    has_mps_backend,
    worker_init_fn,
)
from ..core.metadata.wrapper import get_registry
from ..core.paths import MIN_SPLIT_SAMPLES
from ..exceptions import OrchardDatasetError
from .collate import detection_collate_fn
from .dataset import VisionDataset
from .fetcher import DatasetData
from .transforms import get_pipeline_transforms

# Optuna mode: cap workers to prevent file descriptor exhaustion during trials
_OPTUNA_WORKERS_HIGHRES = 4  # Max workers for resolution >= HIGHRES_THRESHOLD
_OPTUNA_WORKERS_LOWRES = 6  # Max workers for resolution < HIGHRES_THRESHOLD


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
        task_type: str = "classification",  # pragma: no mutate
    ) -> None:
        """
        Initializes the factory with environment and dataset metadata.

        Args:
            dataset_cfg: Dataset sub-config (splits, classes, resolution).
            training_cfg: Training sub-config (batch size, seed).
            aug_cfg: Augmentation sub-config (transforms pipeline).
            num_workers: Resolved worker count from hardware config.
            metadata: Metadata from the data fetcher/downloader.
            task_type: Task type (``"classification"`` or ``"detection"``).
                Controls collate function and sampler selection.
        """
        self.dataset_cfg = dataset_cfg
        self.training_cfg = training_cfg
        self.aug_cfg = aug_cfg
        self._num_workers = num_workers
        self.metadata = metadata
        self._task_type = task_type

        # task_type→None is unkillable (falls back to classification, same as default)
        wrapper = get_registry(dataset_cfg.resolution, task_type)  # pragma: no mutate
        self.ds_meta = wrapper.get_dataset(dataset_cfg.dataset_name)
        self.logger = logging.getLogger(LOGGER_NAME)

    def _get_transformation_pipelines(self) -> tuple[torch.nn.Module, torch.nn.Module]:
        """
        Retrieves specialized transform pipelines.

        Returns:
            A tuple containing (train_transform, val_transform).
        """
        return get_pipeline_transforms(
            self.aug_cfg,
            self.dataset_cfg.img_size,
            self.ds_meta,
            force_rgb=self.dataset_cfg.force_rgb,
            norm_mean=self.dataset_cfg.mean,
            norm_std=self.dataset_cfg.std,
        )

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
            raise OrchardDatasetError(
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

        self.logger.info(
            "%s%s %-18s: WeightedRandomSampler generated",
            LogStyle.INDENT,
            LogStyle.ARROW,
            "Balancing",
        )
        return WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=len(sample_weights),
            replacement=True,
        )

    def _get_infrastructure_kwargs(self, is_optuna: bool = False) -> Mapping[str, Any]:
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
                if self.dataset_cfg.resolution >= HIGHRES_THRESHOLD
                else _OPTUNA_WORKERS_LOWRES
            )
            num_workers = min(num_workers, cap)

            self.logger.info(
                "%s%s %-18s: %d (Resolution=%d)",
                LogStyle.INDENT,
                LogStyle.ARROW,
                "Optuna Workers",
                num_workers,
                self.dataset_cfg.resolution,
            )

        # Hardware acceleration: Pin memory for CUDA or MPS
        has_cuda = torch.cuda.is_available()
        has_mps = has_mps_backend()

        return MappingProxyType(
            {
                "num_workers": num_workers,
                "pin_memory": has_cuda or has_mps,
                "worker_init_fn": worker_init_fn if num_workers > 0 else None,
                "persistent_workers": (num_workers > 0) and (not is_optuna),
            }
        )

    def _build_classification_splits(
        self,
        train_trans: torch.nn.Module,
        val_trans: torch.nn.Module,
        sub_samples: int | None,
    ) -> tuple[VisionDataset, VisionDataset, VisionDataset]:
        """Build train/val/test VisionDataset splits for classification."""
        _build = VisionDataset.lazy if self.dataset_cfg.lazy_loading else VisionDataset.from_npz
        train_ds = _build(
            path=self.metadata.path,
            split="train",
            transform=train_trans,
            max_samples=self.dataset_cfg.max_samples,
            seed=self.training_cfg.seed,
        )
        val_ds = _build(
            path=self.metadata.path,
            split="val",
            transform=val_trans,
            max_samples=sub_samples,
            seed=self.training_cfg.seed,
        )
        test_ds = _build(
            path=self.metadata.path,
            split="test",
            transform=val_trans,
            max_samples=sub_samples,
            seed=self.training_cfg.seed,
        )
        return train_ds, val_ds, test_ds

    def _build_detection_splits(
        self,
        train_trans: torch.nn.Module,
        val_trans: torch.nn.Module,
        sub_samples: int | None,
    ) -> tuple[Any, Any, Any]:
        """Build train/val/test DetectionDataset splits for detection."""
        from .detection_dataset import DetectionDataset

        assert self.metadata.annotation_path is not None  # nosec B101
        train_ds = DetectionDataset.from_npz(
            image_path=self.metadata.path,
            annotation_path=self.metadata.annotation_path,
            split="train",
            transform=train_trans,
            max_samples=self.dataset_cfg.max_samples,
            seed=self.training_cfg.seed,
        )
        val_ds = DetectionDataset.from_npz(
            image_path=self.metadata.path,
            annotation_path=self.metadata.annotation_path,
            split="val",
            transform=val_trans,
            max_samples=sub_samples,
            seed=self.training_cfg.seed,
        )
        test_ds = DetectionDataset.from_npz(
            image_path=self.metadata.path,
            annotation_path=self.metadata.annotation_path,
            split="test",
            transform=val_trans,
            max_samples=sub_samples,
            seed=self.training_cfg.seed,
        )
        return train_ds, val_ds, test_ds

    def build(
        self, is_optuna: bool = False
    ) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
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

        # 2. Instantiate Dataset splits
        is_detection = self._task_type == "detection"  # pragma: no mutate

        sub_samples = None
        if self.dataset_cfg.max_samples:
            sub_samples = max(
                MIN_SPLIT_SAMPLES,
                int(self.dataset_cfg.max_samples * self.dataset_cfg.val_ratio),
            )

        if is_detection and self.metadata.annotation_path is not None:
            train_ds, val_ds, test_ds = self._build_detection_splits(
                train_trans, val_trans, sub_samples
            )
        else:
            train_ds, val_ds, test_ds = self._build_classification_splits(
                train_trans, val_trans, sub_samples
            )

        # 3. Resolve Sampler, Collate, and Infrastructure
        sampler = None if is_detection else self._get_balancing_sampler(train_ds)
        collate_fn = detection_collate_fn if is_detection else None
        infra_kwargs = self._get_infrastructure_kwargs(is_optuna=is_optuna)

        # 4. Construct DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size=self.training_cfg.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            drop_last=True,
            collate_fn=collate_fn,
            **infra_kwargs,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=self.training_cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            **infra_kwargs,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=self.training_cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            **infra_kwargs,
        )

        optuna_str = " (Optuna)" if is_optuna else ""
        self.logger.info(
            "%s%s %-18s: (%s)%s → Train:[%d] Val:[%d] Test:[%d]",
            LogStyle.INDENT,
            LogStyle.ARROW,
            "DataLoaders",
            self.dataset_cfg.processing_mode,
            optuna_str,
            len(train_ds),
            len(val_ds),
            len(test_ds),
        )

        return train_loader, val_loader, test_loader


def get_dataloaders(
    metadata: DatasetData,
    dataset_cfg: DatasetConfig,
    training_cfg: TrainingConfig,
    aug_cfg: AugmentationConfig,
    num_workers: int,
    is_optuna: bool = False,
    task_type: str = "classification",  # pragma: no mutate
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
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
        task_type: Task type (``"classification"`` or ``"detection"``).

    Returns:
        A 3-tuple of (train_loader, val_loader, test_loader).

    Example:
        >>> data = load_dataset(ds_meta)
        >>> loaders = get_dataloaders(
        ...     data, cfg.dataset, cfg.training, cfg.augmentation, cfg.num_workers
        ... )
    """
    factory = DataLoaderFactory(
        dataset_cfg, training_cfg, aug_cfg, num_workers, metadata, task_type=task_type
    )
    return factory.build(is_optuna=is_optuna)
