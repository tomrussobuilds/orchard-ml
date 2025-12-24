"""
Data Transformations Module

This module defines the image augmentation pipelines for training and 
the standard normalization for validation/testing. It also includes 
utilities for deterministic worker initialization. It supports both RGB
and Grayscale datasets dynamically.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import random
from typing import Tuple, Final

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import numpy as np
import torch
from torchvision import transforms

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from src.core import Config


# =========================================================================== #
#                             TRANSFORMATION PIPELINES                        #
# =========================================================================== #
# Standard constants
IMG_SIZE: Final[int] = 28

# Normalization values for ImageNet (RGB)
RGB_MEAN: Final[Tuple[float, float, float]] = (0.485, 0.456, 0.406)
RGB_STD: Final[Tuple[float, float, float]] = (0.229, 0.224, 0.225)

# Grayscale normalization values
GRAY_MEAN: Final[Tuple[float]] = (0.5,)
GRAY_STD: Final[Tuple[float]] = (0.5,)


def get_augmentations_description(cfg: Config) -> str:
    """
    Generates a descriptive string of the augmentations using values from Config.
    Used for logging and run traceability.
    """ 
    augs = [
        f"HFlip(p={cfg.augmentation.hflip})",
        f"Rotation({cfg.augmentation.rotation_angle}°)",
        f"Jitter(v={cfg.augmentation.jitter_val})",
        f"ResizedCrop({IMG_SIZE}, scale=(0.9, 1.0))"
    ]
    if cfg.training.mixup_alpha > 0:
        augs.append(f"MixUp(α={cfg.training.mixup_alpha})")
    
    return ", ".join(augs)


def worker_init_fn(worker_id: int):
    """
    Initializes random number generators (PRNGs) for each DataLoader worker.
    Crucial for maintaining augmentation diversity and reproducibility 
    when using multiple workers for lazy-loading.
    """
    # Create a seed unique to the worker but based on the global initial seed
    base_seed = torch.initial_seed() % 2**32
    worker_seed = base_seed + worker_id
    
    np.random.seed(worker_seed)
    random.seed(worker_seed) 
    torch.manual_seed(worker_seed)


def get_pipeline_transforms(
        cfg: Config,
        is_rgb: bool = True
    ) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Defines the transformation pipelines for training and evaluation.
    
    Args:
        cfg: Configuration object with augmentation parameters.
        is_rgb: Boolean indicating if the dataset is RGB or Grayscale.
    
    Returns:
        Tuple[transforms.Compose, transforms.Compose]: (train_transform, val_transform)
    """
    mean = RGB_MEAN if is_rgb else GRAY_MEAN
    std = RGB_STD if is_rgb else GRAY_STD

    # Training pipeline: Focus on robust generalization
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=cfg.augmentation.hflip),
        transforms.RandomRotation(cfg.augmentation.rotation_angle),
        transforms.ColorJitter(
            brightness=cfg.augmentation.jitter_val,
            contrast=cfg.augmentation.jitter_val,
            saturation=cfg.augmentation.jitter_val if is_rgb else 0.0,
        ),
        # Using a subtle scale range to preserve medical feature proportions
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    # Validation/Inference pipeline: Strict consistency
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    return train_transform, val_transform