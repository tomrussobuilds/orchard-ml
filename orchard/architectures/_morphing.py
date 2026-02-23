"""
Weight Morphing Utilities for Pretrained Model Adaptation.

Provides shared helpers for adapting pretrained ImageNet weights when
input channels differ (e.g., 1-channel medical images vs 3-channel RGB).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def morph_conv_weights(
    old_conv: nn.Conv2d,
    new_conv: nn.Conv2d,
    in_channels: int,
    target_kernel_size: tuple[int, int] | None = None,
) -> None:
    """
    Adapt pretrained conv weights for channel and optional spatial mismatch.

    Performs in-place weight copy from old_conv to new_conv with:
    - Optional bicubic spatial interpolation (e.g., 7x7 → 3x3)
    - Channel compression via mean averaging for grayscale (RGB → 1ch)
    - Conditional bias transfer

    Args:
        old_conv: Source conv layer with pretrained weights.
        new_conv: Target conv layer to receive morphed weights.
        in_channels: Number of input channels for the target model.
        target_kernel_size: If provided, spatially resize the kernel
            via bicubic interpolation before channel morphing.
    """
    with torch.no_grad():
        w = old_conv.weight.clone()

        if target_kernel_size is not None:
            w = F.interpolate(w, size=target_kernel_size, mode="bicubic", align_corners=True)

        if in_channels == 1:
            w = w.mean(dim=1, keepdim=True)

        new_conv.weight.copy_(w)

        if old_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
