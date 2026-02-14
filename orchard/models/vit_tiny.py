"""
Vision Transformer Tiny (ViT-Tiny) for 224×224 Medical Imaging.

Implements the Vision Transformer architecture via timm library with support for
multiple pretrained weight variants. Designed for efficient medical image
classification with transfer learning capabilities.

Key Features:
    - Patch-Based Attention: Processes 16×16 patches with transformer encoders
    - Multi-Weight Support: Compatible with ImageNet-1k/21k pretraining
    - Adaptive Input: Dynamic first-layer modification for grayscale datasets
    - Efficient Scale: Tiny variant balances performance and compute requirements

Pretrained Weight Options:
    - 'vit_tiny_patch16_224.augreg_in21k_ft_in1k': ImageNet-21k → 1k fine-tuned
    - 'vit_tiny_patch16_224.augreg_in21k': ImageNet-21k (requires custom head)
    - 'vit_tiny_patch16_224': ImageNet-1k baseline
"""

import logging
from typing import cast

import timm
import torch
import torch.nn as nn

from orchard.core import LOGGER_NAME, Config

# LOGGER CONFIGURATION
logger = logging.getLogger(LOGGER_NAME)


# MODEL BUILDER
def build_vit_tiny(
    device: torch.device, num_classes: int, in_channels: int, cfg: Config
) -> nn.Module:
    """
    Constructs Vision Transformer Tiny adapted for medical imaging datasets.

    Workflow:
        1. Resolve pretrained weight variant from config (if enabled)
        2. Load model via timm with automatic head replacement
        3. Modify patch embedding layer for custom input channels
        4. Apply weight morphing for channel compression (if grayscale)
        5. Deploy model to target device (CUDA/MPS/CPU)

    Args:
        device: Target hardware for model placement
        num_classes: Number of dataset classes for classification head
        in_channels: Input channels (1=Grayscale, 3=RGB)
        cfg: Global configuration with pretrained/dropout/weight_variant settings

    Returns:
        Adapted ViT-Tiny model deployed to device

    Raises:
        ValueError: If weight variant is invalid or incompatible with pretrained flag
    """
    # --- Step 1: Resolve Weight Variant ---
    weight_variant = cfg.architecture.weight_variant or "vit_tiny_patch16_224.augreg_in21k_ft_in1k"

    if cfg.architecture.pretrained:
        logger.info(f"Loading ViT-Tiny with pretrained weights: {weight_variant}")
        pretrained_flag = True
    else:
        logger.info("Initializing ViT-Tiny with random weights")
        pretrained_flag = False
        weight_variant = "vit_tiny_patch16_224"  # Use base architecture

    # --- Step 2: Load Model via timm ---
    try:
        model = timm.create_model(
            weight_variant,
            pretrained=pretrained_flag,
            num_classes=num_classes,
            in_chans=3,  # Initially load for 3 channels (will adapt below)
        )
    except (RuntimeError, ValueError) as e:
        logger.error(f"Failed to load ViT variant '{weight_variant}': {e}")
        raise ValueError(f"Invalid ViT weight variant: {weight_variant}") from e

    # --- Step 3: Adapt Patch Embedding Layer ---
    if in_channels != 3:
        logger.info(f"Adapting patch embedding from 3 to {in_channels} channels")

        # Type-narrow patch_embed.proj to Conv2d for mypy
        # Note: timm VisionTransformer.patch_embed has dynamic type, ignore for type checking
        old_proj = cast(nn.Conv2d, model.patch_embed.proj)  # type: ignore[union-attr]

        # Extract attributes (cast to specific types for mypy)
        kernel_size = cast("tuple[int, int]", old_proj.kernel_size)
        stride = cast("tuple[int, int]", old_proj.stride)
        padding = cast("tuple[int, int] | int", old_proj.padding)

        # Create new projection layer
        new_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_proj.out_channels,  # 192 for ViT-Tiny
            kernel_size=kernel_size,  # (16, 16)
            stride=stride,  # (16, 16)
            padding=padding,
            bias=old_proj.bias is not None,
        )

        # --- Step 4: Weight Morphing (Transfer Pretrained Knowledge) ---
        if cfg.architecture.pretrained:
            with torch.no_grad():
                w = old_proj.weight  # Shape: [192, 3, 16, 16]

                if in_channels == 1:
                    # Compress RGB channels by averaging (preserves learned patterns)
                    w = w.mean(dim=1, keepdim=True)  # [192, 1, 16, 16]

                new_proj.weight.copy_(w)

                if old_proj.bias is not None and new_proj.bias is not None:
                    new_proj.bias.copy_(old_proj.bias)

        # Replace patch embedding projection
        model.patch_embed.proj = new_proj  # type: ignore[union-attr]

    # --- Step 5: Device Placement ---
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"ViT-Tiny deployed | Parameters: {total_params:,}")

    return model
