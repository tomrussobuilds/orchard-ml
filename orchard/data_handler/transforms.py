"""
Data Transformation Pipelines.

Defines image augmentation for training and normalization for validation/testing.
Supports both RGB and Grayscale datasets with automatic channel promotion.
Optimized for both CPU and GPU execution (torchvision v2).
"""

# Standard Imports
from typing import Tuple

# Third-Party Imports
import torch
from torchvision.transforms import v2

# Internal Imports
from orchard.core import Config, DatasetMetadata


# TRANSFORMATION UTILITIES
def get_augmentations_description(cfg: Config) -> str:
    """
    Generates descriptive string of augmentations for logging.

    Args:
        cfg: Global configuration

    Returns:
        Human-readable augmentation summary
    """
    params = {
        "HFlip": cfg.augmentation.hflip,
        "Rotation": f"{cfg.augmentation.rotation_angle}°",
        "Jitter": cfg.augmentation.jitter_val,
        "ResizedCrop": f"{cfg.dataset.img_size} ({cfg.augmentation.min_scale}, 1.0)",
    }

    descr = [f"{k}({v})" for k, v in params.items()]

    if cfg.training.mixup_alpha > 0:
        descr.append(f"MixUp(α={cfg.training.mixup_alpha})")

    return ", ".join(descr)


def get_pipeline_transforms(cfg: Config, ds_meta: DatasetMetadata) -> Tuple[v2.Compose, v2.Compose]:
    """
    Constructs training and validation transformation pipelines.

    Dynamically adapts to dataset characteristics (RGB vs Grayscale) and
    ensures 3-channel output for ResNet/EfficientNet compatibility.
    Uses torchvision v2 transforms for improved CPU/GPU performance.

    Pipeline Logic:
        1. Convert to tensor format (ToImage + ToDtype)
        2. Promote 1-channel to 3-channel if needed (Grayscale → RGB)
        3. Apply augmentations (training only)
        4. Normalize with dataset-specific statistics

    Args:
        cfg: Global configuration with augmentation parameters
        ds_meta: Dataset metadata (channels, normalization stats)

    Returns:
        Tuple[v2.Compose, v2.Compose]: (train_transform, val_transform)
    """

    # Determine if dataset is native RGB or requires grayscale promotion
    is_rgb = ds_meta.in_channels == 3

    # Extract normalization statistics from registry
    # Replicate single-channel stats for grayscale → RGB promotion
    if ds_meta.in_channels == 1:
        mean = [ds_meta.mean[0]] * 3
        std = [ds_meta.std[0]] * 3
    else:
        mean = ds_meta.mean
        std = ds_meta.std

    def get_base_ops():
        """
        Foundational operations common to all pipelines.

        Returns:
            List of base transforms (tensor conversion + channel promotion)
        """
        ops = [
            v2.ToImage(),  # Convert PIL/ndarray to tensor
            v2.ToDtype(torch.float32, scale=True),  # Scale to [0,1]
        ]

        # Promote 1-channel to 3-channel for architecture compatibility
        if not is_rgb:
            ops.append(v2.Grayscale(num_output_channels=3))

        return ops

    # --- TRAINING PIPELINE ---
    # Includes spatial and photometric augmentations
    train_transform = v2.Compose(
        [
            *get_base_ops(),
            v2.RandomHorizontalFlip(p=cfg.augmentation.hflip),
            v2.RandomRotation(cfg.augmentation.rotation_angle),
            v2.ColorJitter(
                brightness=cfg.augmentation.jitter_val,
                contrast=cfg.augmentation.jitter_val,
                saturation=cfg.augmentation.jitter_val if is_rgb else 0.0,
            ),
            v2.RandomResizedCrop(
                size=cfg.dataset.img_size,
                scale=(cfg.augmentation.min_scale, 1.0),
                antialias=True,
                interpolation=v2.InterpolationMode.BILINEAR,
            ),
            v2.Normalize(mean=mean, std=std),
        ]
    )

    # --- VALIDATION/INFERENCE PIPELINE ---
    # Deterministic transformations only (no augmentation)
    val_transform = v2.Compose(
        [
            *get_base_ops(),
            v2.Resize(size=cfg.dataset.img_size, antialias=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )

    return train_transform, val_transform
