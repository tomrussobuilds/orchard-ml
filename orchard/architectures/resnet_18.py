"""
ResNet-18 Multi-Resolution Architecture.

Adaptive ResNet-18 supporting 28x28, 32x32, 64x64, and 224x224 resolutions
with resolution-specific architectural modifications.

Resolution-Specific Adaptations:
    28x28 / 32x32:
        - 7x7 Conv1 → 3x3 Conv1 (stride 1 instead of 2)
        - MaxPool removed (prevents excessive spatial loss)
        - Weight morphing via bicubic interpolation
    64x64 / 224x224:
        - Standard ResNet-18 stem (7x7 Conv1, stride 2, MaxPool)
        - Only channel adaptation for grayscale inputs
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ..core import Config


# MODEL BUILDER
def build_resnet18(
    device: torch.device, num_classes: int, in_channels: int, cfg: Config
) -> nn.Module:
    """
    Constructs ResNet-18 with resolution-aware architectural adaptation.

    At 28x28, performs stem surgery to preserve spatial resolution.
    At 64x64 and 224x224, uses the standard ResNet-18 architecture.

    Workflow:
        1. Load ImageNet pretrained ResNet-18 (if enabled)
        2. Apply resolution-specific stem adaptation
        3. Replace classification head with dataset-specific linear layer
        4. Deploy model to target device

    Args:
        device: Target hardware for model placement
        num_classes: Number of dataset classes
        in_channels: Input channels (1=Grayscale, 3=RGB)
        cfg: Global configuration with pretrained settings and resolution

    Returns:
        Adapted ResNet-18 deployed to device
    """

    # --- Step 1: Initialize with Optional Pretraining ---
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if cfg.architecture.pretrained else None
    model = models.resnet18(weights=weights)

    # --- Step 2: Resolution-Aware Stem Adaptation ---
    resolution = cfg.dataset.resolution
    pretrained = cfg.architecture.pretrained

    if resolution <= 32:  # 28, 32: small stem (3x3 stride-1, no MaxPool)
        _adapt_stem_28(model, in_channels, pretrained)
    else:  # 64, 224 — standard stem
        _adapt_stem_224(model, in_channels, pretrained)

    # --- Step 3: Replace Classification Head ---
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # --- Step 4: Device Placement ---
    model = model.to(device)

    return model


# INTERNAL HELPERS
def _adapt_stem_28(model: nn.Module, in_channels: int, pretrained: bool) -> None:
    """
    Adapts ResNet-18 stem for 28x28 inputs.

    Replaces 7x7 conv1 with 3x3 stride-1, removes MaxPool, and applies
    bicubic weight interpolation from pretrained 7x7 kernels.
    """
    old_conv = cast(nn.Conv2d, model.conv1)

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    if pretrained:
        with torch.no_grad():
            w = old_conv.weight.clone()  # [64, 3, 7, 7]
            w = F.interpolate(w, size=(3, 3), mode="bicubic", align_corners=True)

            if in_channels == 1:
                w = w.mean(dim=1, keepdim=True)

            new_conv.weight.copy_(w)

    model.conv1 = new_conv
    model.maxpool = nn.Identity()


def _adapt_stem_224(model: nn.Module, in_channels: int, pretrained: bool) -> None:
    """
    Adapts ResNet-18 stem for 224x224 grayscale inputs.

    Keeps the standard 7x7 conv1 and MaxPool, only modifying
    input channels via weight averaging when grayscale.
    """
    if in_channels == 3:
        return  # No modification needed for standard RGB

    old_conv = cast(nn.Conv2d, model.conv1)

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )

    if pretrained:
        with torch.no_grad():
            w = old_conv.weight.clone()  # [64, 3, 7, 7]

            if in_channels == 1:
                w = w.mean(dim=1, keepdim=True)

            new_conv.weight.copy_(w)

    model.conv1 = new_conv
