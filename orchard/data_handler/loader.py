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
    DataLoaderFactory: Main orchestrator for train/val/test loader creation
    get_dataloaders: Convenience function for direct loader retrieval
    create_temp_loader: Quick DataLoader builder for health checks

Example:
    >>> from orchard.data_handler import get_dataloaders, load_dataset
    >>> data = load_dataset(ds_meta)
    >>> train_loader, val_loader, test_loader = get_dataloaders(data, cfg)
    >>> print(f"Batches: {len(train_loader)}")
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..core import LOGGER_NAME, Config, DatasetRegistryWrapper, worker_init_fn
from .dataset import LazyNPZDataset, VisionDataset
from .fetcher import DatasetData
from .transforms import get_pipeline_transforms


# DATALOADER FACTORY
class DataLoaderFactory:
    """Orchestrates the creation of optimized PyTorch DataLoaders.

    This factory centralizes the configuration of training, validation, and
    testing pipelines. It ensures that data transformations, class balancing,
    and hardware settings are synchronized across all splits.

    Attributes:
        cfg (Config): Validated global configuration.
        metadata (DatasetData): Data path and raw format information.
        ds_meta (DatasetMetadata): Official dataset registry specifications.
        logger (logging.Logger): Module-specific logger.
    """

    def __init__(self, cfg: Config, metadata: DatasetData) -> None:
        """Initializes the factory with environment and dataset metadata.

        Args:
            cfg: The global configuration object (Pydantic).
            metadata: Metadata from the data fetcher/downloader.
        """
        self.cfg = cfg
        self.metadata = metadata

        wrapper = DatasetRegistryWrapper(resolution=cfg.dataset.resolution)
        self.ds_meta = wrapper.get_dataset(cfg.dataset.dataset_name)
        self.logger = logging.getLogger(LOGGER_NAME)

    def _get_transformation_pipelines(self) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Retrieves specialized vision pipelines.

        Returns:
            A tuple containing (train_transform, val_transform).
        """
        return get_pipeline_transforms(self.cfg, self.ds_meta)

    def _get_balancing_sampler(self, dataset: VisionDataset) -> WeightedRandomSampler | None:
        """Calculates class weights and builds a WeightedRandomSampler.

        This method addresses class imbalance by assigning higher sampling
        probabilities to under-represented classes.

        Args:
            dataset: The training dataset instance.

        Returns:
            A WeightedRandomSampler if enabled in config, otherwise None.
        """
        if not self.cfg.dataset.use_weighted_sampler:
            return None

        labels = dataset.labels.flatten()
        classes, counts = np.unique(labels, return_counts=True)

        # Inverse frequency balancing: weight = 1 / frequency
        # Guard against zero-count classes (possible with aggressive max_samples)
        safe_counts = np.maximum(counts, 1)
        if np.any(counts == 0):
            self.logger.warning(
                "Some classes have 0 samples in the training set. "
                "Their sampling weight has been set to 1.0."
            )
        class_weights = 1.0 / safe_counts
        weight_map: dict[int, float] = dict(zip(classes, class_weights))

        sample_weights = torch.tensor(
            [weight_map[int(label)] for label in labels], dtype=torch.float
        )

        self.logger.info("Class balancing: WeightedRandomSampler generated.")
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
        num_workers = self.cfg.num_workers

        # OPTUNA MODE: Reduce workers to prevent file descriptor exhaustion
        _OPTUNA_WORKERS_HIGHRES = 4  # Cap for resolution >= _HIGHRES_THRESHOLD
        _OPTUNA_WORKERS_LOWRES = 6  # Cap for resolution < _HIGHRES_THRESHOLD
        _HIGHRES_THRESHOLD = 224  # Resolution boundary for worker tuning
        if is_optuna:
            cap = (
                _OPTUNA_WORKERS_HIGHRES
                if self.cfg.dataset.resolution >= _HIGHRES_THRESHOLD
                else _OPTUNA_WORKERS_LOWRES
            )
            num_workers = min(num_workers, cap)

            self.logger.info(
                f"Optuna mode: Reducing workers to {num_workers} "
                f"(resolution={self.cfg.dataset.resolution})"
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
        """Constructs and returns the full suite of DataLoaders.

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
        ds_params = {"path": self.metadata.path, "seed": self.cfg.training.seed}

        train_ds = VisionDataset(
            **ds_params,
            split="train",
            transform=train_trans,
            max_samples=self.cfg.dataset.max_samples,
        )

        # Proportional downsizing for validation/testing if max_samples is set
        sub_samples = None
        if self.cfg.dataset.max_samples:
            sub_samples = max(10, int(self.cfg.dataset.max_samples * self.cfg.dataset.val_ratio))

        val_ds = VisionDataset(
            **ds_params, split="val", transform=val_trans, max_samples=sub_samples
        )
        test_ds = VisionDataset(
            **ds_params, split="test", transform=val_trans, max_samples=sub_samples
        )

        # 3. Resolve Sampler and Infrastructure
        sampler = self._get_balancing_sampler(train_ds)
        infra_kwargs = self._get_infrastructure_kwargs(is_optuna=is_optuna)

        # 4. Construct DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.training.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            drop_last=True,
            **infra_kwargs,
        )

        val_loader = DataLoader(
            val_ds, batch_size=self.cfg.training.batch_size, shuffle=False, **infra_kwargs
        )

        test_loader = DataLoader(
            test_ds, batch_size=self.cfg.training.batch_size, shuffle=False, **infra_kwargs
        )

        mode_str = "RGB" if self.ds_meta.in_channels == 3 else "Grayscale"
        optuna_str = " [OPTUNA-MODE]" if is_optuna else ""
        self.logger.info(
            f"DataLoaders synchronized ({mode_str}){optuna_str} → "
            f"Train:[{len(train_ds)}] Val:[{len(val_ds)}] Test:[{len(test_ds)}] "
            f"Workers:[{infra_kwargs['num_workers']}]"
        )

        return train_loader, val_loader, test_loader


def get_dataloaders(
    metadata: DatasetData, cfg: Config, is_optuna: bool = False
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience function for creating train/val/test DataLoaders.

    Wraps DataLoaderFactory for streamlined loader construction with
    automatic class balancing, hardware optimization, and Optuna support.

    Args:
        metadata (DatasetData): Dataset metadata from load_dataset (paths, splits)
        cfg (Config): Global configuration with batch size, workers, and augmentation settings
        is_optuna (bool): If True, use memory-conservative settings for hyperparameter tuning

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: A 3-tuple of
            (train_loader, val_loader, test_loader)

    Example:
        >>> data = load_dataset(ds_meta)
        >>> loaders = get_dataloaders(data, cfg, is_optuna=False)
        >>> train_loader, val_loader, test_loader = loaders
    """
    factory = DataLoaderFactory(cfg, metadata)
    return factory.build(is_optuna=is_optuna)


# HEALTH UTILITIES
def create_temp_loader(dataset_path: Path, batch_size: int = 16) -> DataLoader:
    """
    Load a NPZ dataset lazily and return a DataLoader for health checks.

    This avoids loading the entire dataset into RAM at once, which is critical
    for large datasets (e.g., 224x224 images).
    """
    dataset = LazyNPZDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader
