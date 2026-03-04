"""
Data Transformation Pipelines.

Defines image augmentation for training and normalization for validation/testing.
Supports both RGB and Grayscale datasets with automatic channel promotion.
Optimized for both CPU and GPU execution (torchvision v2).
"""

from __future__ import annotations

import torch
from torchvision.transforms import v2

from ..core import DatasetMetadata
from ..core.config import AugmentationConfig


# TRANSFORMATION UTILITIES
def get_augmentations_description(
    aug_cfg: AugmentationConfig,
    img_size: int,
    mixup_alpha: float,
    ds_meta: DatasetMetadata | None = None,
) -> str:
    """
    Generates descriptive string of augmentations for logging.

    Args:
        aug_cfg: Augmentation sub-configuration
        img_size: Target image size for resized crop
        mixup_alpha: MixUp alpha (0.0 to disable)
        ds_meta: Dataset metadata (if provided, respects domain flags)

    Returns:
        Human-readable augmentation summary
    """
    is_anatomical = ds_meta.is_anatomical if ds_meta else True
    is_texture = ds_meta.is_texture_based if ds_meta else True

    params = {}
    if not is_anatomical:
        params["HFlip"] = aug_cfg.hflip
        params["Rotation"] = f"{aug_cfg.rotation_angle}°"
    if not is_texture:
        params["Jitter"] = aug_cfg.jitter_val
    params["ResizedCrop"] = f"{img_size} ({aug_cfg.min_scale}, 1.0)"

    descr = [f"{k}({v})" for k, v in params.items()]

    if mixup_alpha > 0:
        descr.append(f"MixUp(α={mixup_alpha})")

    return ", ".join(descr)


def get_pipeline_transforms(
    aug_cfg: AugmentationConfig,
    img_size: int,
    ds_meta: DatasetMetadata,
    *,
    force_rgb: bool = True,
    norm_mean: tuple[float, ...] | None = None,
    norm_std: tuple[float, ...] | None = None,
) -> tuple[v2.Compose, v2.Compose]:
    """
    Constructs training and validation transformation pipelines.

    Dynamically adapts to dataset characteristics (RGB vs Grayscale) and
    optionally promotes grayscale to 3-channel for pretrained-weight
    compatibility.  Uses torchvision v2 transforms for improved CPU/GPU
    performance.

    Pipeline Logic:
        1. Convert to tensor format (ToImage + ToDtype)
        2. Promote 1-channel to 3-channel when ``force_rgb`` is True
           and the dataset is native grayscale
        3. Apply domain-aware augmentations (training only):
           geometric transforms disabled for anatomical datasets,
           color jitter reduced for texture-based datasets
        4. Normalize with dataset-specific statistics

    Args:
        aug_cfg: Augmentation sub-configuration
        img_size: Target image size
        ds_meta: Dataset metadata (channels, domain flags)
        force_rgb: Promote grayscale datasets to 3-channel RGB
        norm_mean: Pre-computed normalization mean (from DatasetConfig).
            When None, computed from ds_meta + force_rgb.
        norm_std: Pre-computed normalization std (from DatasetConfig).
            When None, computed from ds_meta + force_rgb.

    Returns:
        tuple[v2.Compose, v2.Compose]: (train_transform, val_transform)
    """
    # Determine if dataset is native RGB or requires grayscale promotion
    is_rgb = ds_meta.in_channels == 3
    promote_to_rgb = not is_rgb and force_rgb

    # Normalization statistics: prefer caller-provided (single source of truth
    # via DatasetConfig.mean/std), fall back to local computation.
    if norm_mean is not None and norm_std is not None:
        mean = list(norm_mean)
        std = list(norm_std)
    elif promote_to_rgb:
        mean = [ds_meta.mean[0]] * 3
        std = [ds_meta.std[0]] * 3
    else:
        mean = list(ds_meta.mean)
        std = list(ds_meta.std)

    def get_base_ops() -> list[v2.Transform]:
        """
        Foundational operations common to all pipelines.

        Returns:
            list of base transforms (tensor conversion + channel promotion)
        """
        ops = [
            v2.ToImage(),  # Convert PIL/ndarray to tensor
            v2.ToDtype(torch.float32, scale=True),  # Scale to [0,1]
        ]

        # Promote 1-channel to 3-channel for architecture compatibility
        if promote_to_rgb:
            ops.append(v2.Grayscale(num_output_channels=3))

        return ops

    # --- TRAINING PIPELINE ---
    # Domain-aware augmentations: respects is_anatomical and is_texture_based flags
    train_ops = [*get_base_ops()]

    # Geometric: disabled for anatomical datasets (orientation is diagnostic)
    if not ds_meta.is_anatomical:
        train_ops.append(v2.RandomHorizontalFlip(p=aug_cfg.hflip))
        train_ops.append(v2.RandomRotation(aug_cfg.rotation_angle))

    # Photometric: reduced for texture-based datasets (fine patterns are fragile)
    if not ds_meta.is_texture_based:
        train_ops.append(
            v2.ColorJitter(
                brightness=aug_cfg.jitter_val,
                contrast=aug_cfg.jitter_val,
                saturation=aug_cfg.jitter_val if is_rgb else 0.0,
            )
        )

    train_ops.extend(
        [
            v2.RandomResizedCrop(
                size=img_size,
                scale=(aug_cfg.min_scale, 1.0),
                antialias=True,
                interpolation=v2.InterpolationMode.BILINEAR,
            ),
            v2.Normalize(mean=mean, std=std),
        ]
    )

    train_transform = v2.Compose(train_ops)

    # --- VALIDATION/INFERENCE PIPELINE ---
    # Deterministic transformations only (no augmentation)
    val_transform = v2.Compose(
        [
            *get_base_ops(),
            v2.Resize(size=img_size, antialias=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )

    return train_transform, val_transform
