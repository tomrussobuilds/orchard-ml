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
import torch
from torch.utils.data import DataLoader

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from scripts.core import Config
from .fetcher import MedMNISTData
from .dataset import MedMNISTDataset
from .transforms import get_pipeline_transforms, worker_init_fn

# =========================================================================== #
#                             DATALOADER CREATION                             #
# =========================================================================== #
# Global logger instance
logger = logging.getLogger("medmnist_pipeline")


def get_dataloaders(
        data: MedMNISTData,
        cfg: Config,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders for train, validation, and test splits.
    
    This function detects the image format (RGB/Gray) from the data shape
    and applies the corresponding transformation pipelines.
    """
    # 1. Detect if dataset is RGB or Grayscale
    # RGB images have shape (N, 28, 28, 3), Gray have (N, 28, 28)
    is_rgb = (data.X_train.ndim == 4 and data.X_train.shape[-1] == 3)
    
    # 2. Get Transformation Pipelines (Passing the detected is_rgb flag)
    train_transform, val_transform = get_pipeline_transforms(cfg, is_rgb=is_rgb)

    # 3. Create Datasets (Using the generic MedMNISTDataset class)
    train_ds = MedMNISTDataset(
        data.X_train,
        data.y_train,
        path=data.path,
        transform=train_transform
    )
    val_ds   = MedMNISTDataset(
        data.X_val,
        data.y_val,
        path=data.path,
        transform=val_transform
    )
    test_ds  = MedMNISTDataset(
        data.X_test,
        data.y_test,
        path=data.path,
        transform=val_transform
    )
    
    # 4. Setup DataLoader Parameters
    init_fn = worker_init_fn if cfg.num_workers > 0 else None
    pin_memory = torch.cuda.is_available()

    # 5. Create DataLoaders (Explicit definitions restored)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=init_fn,
        persistent_workers=(cfg.num_workers > 0)
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=init_fn,
        persistent_workers=(cfg.num_workers > 0)
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=init_fn,
        persistent_workers=(cfg.num_workers > 0)
    )
    
    mode_str = "RGB" if is_rgb else "Grayscale"
    logger.info(
        f"DataLoaders ready ({mode_str}) → "
        f"Train:{len(train_ds)} | Val:{len(val_ds)} | Test:{len(test_ds)}"
    )
    
    return train_loader, val_loader, test_loader