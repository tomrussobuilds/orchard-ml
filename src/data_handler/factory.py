"""
DataLoader Generation Module

This module orchestrates the creation of PyTorch DataLoaders by combining 
the fetched data, the dataset structure, and the transformation pipelines.
It dynamically adapts to RGB or Grayscale datasets.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import logging
from typing import Tuple

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from src.core import Config
from .fetcher import MedMNISTData
from .dataset import MedMNISTDataset
from .transforms import get_pipeline_transforms, worker_init_fn

# =========================================================================== #
#                             DATALOADER CREATION                             #
# =========================================================================== #
# Global logger instance
logger = logging.getLogger("medmnist_pipeline")


def get_dataloaders(
        metadata: MedMNISTData,
        cfg: Config,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders for train, validation, and test splits.
    
    Uses lazy loading via Path to keep RAM usage low and applies 
    WeightedRandomSampler for imbalanced datasets if configured.
    """
    # 1. Get Transformation Pipelines (Using metadata for RGB/Gray detection)
    train_transform, val_transform = get_pipeline_transforms(
        cfg, is_rgb=metadata.is_rgb
    )

    dataset_params = {
        "path": metadata.path,
        "cfg": cfg,
    }

    # 2. Create Datasets (Using the generic MedMNISTDataset class)
    train_ds = MedMNISTDataset(
        **dataset_params,
        split="train",
        transform=train_transform,
        max_samples=cfg.dataset.max_samples,
    )

    # Calculate proportional samples for validation and test
    # If max_samples is set, we take a fraction (e.g., 10%) for val/test
    # If None, val_samples remains None to use the full original splits
    if cfg.dataset.max_samples is not None:
        # Using a 10% ratio of the training samples
        # Example: 20,000 train -> 2,000 val and 2,000 test
        val_samples = max(1, int(cfg.dataset.max_samples * 0.10))
    else:
        val_samples = None

    val_ds = MedMNISTDataset(
        **dataset_params,
        split="val",
        transform=val_transform,
        max_samples=val_samples,
    )
    test_ds = MedMNISTDataset(
        **dataset_params,
        split="test",
        transform=val_transform,
        max_samples=val_samples,
    )

    # 3. Handle Class Balancing
    sampler = None
    shuffle = True

    if cfg.dataset.use_weighted_sampler:
        # Ensure labels are a flat 1D array of integers
        labels = train_ds.labels.flatten()
        classes, counts = np.unique(labels, return_counts=True)
        
        # Calculate weight for each class (inverse frequency)
        class_weights = 1.0 / counts
        
        # Create a mapping dictionary for safety
        weight_map = dict(zip(classes, class_weights))
        
        # Map each sample to its corresponding class weight
        sample_weights = torch.tensor(
            [weight_map[label] for label in labels], 
            dtype=torch.float
        )

        sampler = WeightedRandomSampler(
            weights=sample_weights, 
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False
        logger.info("WeightedRandomSampler enabled for training.")

    # 4. Setup DataLoader Parameters
    init_fn = worker_init_fn if cfg.num_workers > 0 else None
    pin_memory = torch.cuda.is_available()

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=init_fn,
        persistent_workers=False
    )

    common_params = {
        "batch_size": cfg.training.batch_size,
        "shuffle": False,
        "num_workers": cfg.num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": init_fn,
        "persistent_workers": (cfg.num_workers > 0)
    }

    val_loader = DataLoader(val_ds, **common_params)
    test_loader = DataLoader(test_ds, **common_params)
    
    mode_str = "RGB" if metadata.is_rgb else "Grayscale"
    logger.info(
        f"DataLoaders ready ({mode_str}) → "
        f"Train:{len(train_ds)} | Val:{len(val_ds)} | Test:{len(test_ds)}"
    )
    
    return train_loader, val_loader, test_loader