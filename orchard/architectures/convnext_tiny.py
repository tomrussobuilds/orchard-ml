"""
ConvNeXt-Tiny Architecture for 224x224 Image Classification.

Adapts ConvNeXt-Tiny (modernized ConvNet architecture) for image
classification with transfer learning support. Handles both RGB and grayscale
inputs through dynamic first-layer adaptation.

Key Features:

- Modern ConvNet Design: Incorporates design choices from transformers
- Transfer Learning: Leverages ImageNet pretrained weights
- Adaptive Input: Customizes first layer for grayscale datasets
- Channel Compression: Weight morphing for RGB→grayscale adaptation
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

from ._morphing import morph_conv_weights


# MODEL BUILDER
def build_convnext_tiny(
    num_classes: int,
    in_channels: int,
    *,
    pretrained: bool,
) -> nn.Module:
    """
    Constructs ConvNeXt-Tiny adapted for image classification datasets.

    Workflow:
        1. Load pretrained weights from ImageNet (if enabled)
        2. Modify first conv layer to accept custom input channels
        3. Apply weight morphing for channel compression (if grayscale)
        4. Replace classification head with dataset-specific linear layer

    Args:
        num_classes: Number of dataset classes for classification head
        in_channels: Input channels (1=Grayscale, 3=RGB)
        pretrained: Whether to load ImageNet pretrained weights

    Returns:
        Adapted ConvNeXt-Tiny model (device placement handled by factory).
    """
    # --- Step 1: Initialize with Optional Pretraining ---
    weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.convnext_tiny(weights=weights)

    # Snapshot original first conv layer (before replacement)
    old_conv = model.features[0][0]  # Conv2d(3, 96, kernel_size=4, stride=4)

    # --- Step 2: Adapt First Convolutional Layer ---
    # ConvNeXt expects 3-channel input; modify for grayscale if needed
    new_conv = nn.Conv2d(
        in_channels=in_channels,  # Custom: 1 or 3
        out_channels=96,  # ConvNeXt-Tiny standard
        kernel_size=(4, 4),
        stride=(4, 4),
        padding=(0, 0),
        bias=True,  # ConvNeXt uses bias in stem conv
    )

    # --- Step 3: Weight Morphing (Transfer Pretrained Knowledge) ---
    if pretrained:
        morph_conv_weights(old_conv, new_conv, in_channels)

    # Replace entry layer with adapted version
    model.features[0][0] = new_conv

    # --- Step 4: Modify Classification Head ---
    # Replace ImageNet 1000-class head with dataset-specific projection
    # model.classifier[2] is Linear(768, 1000)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    return model  # type: ignore[no-any-return]
