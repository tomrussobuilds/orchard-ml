"""
EfficientNet-B0 Architecture for 224x224 Medical Imaging.

Adapts EfficientNet-B0 (compound scaling architecture) for medical image
classification with transfer learning support. Handles both RGB and grayscale
inputs through dynamic first-layer adaptation.

Key Features:
    - Efficient Scaling: Balances depth, width, and resolution
    - Transfer Learning: Leverages ImageNet pretrained weights
    - Adaptive Input: Customizes first layer for grayscale datasets
    - Channel Compression: Weight morphing for 1→3 channel promotion
"""

# Third-Party Imports
import torch
import torch.nn as nn
from torchvision import models

# Internal Imports
from orchard.core import Config


# MODEL BUILDER
def build_efficientnet_b0(
    device: torch.device, num_classes: int, in_channels: int, cfg: Config
) -> nn.Module:
    """
    Constructs EfficientNet-B0 adapted for medical imaging datasets.

    Workflow:
        1. Load pretrained weights from ImageNet (if enabled)
        2. Modify first conv layer to accept custom input channels
        3. Apply weight morphing for channel compression (if grayscale)
        4. Replace classification head with dataset-specific linear layer
        5. Deploy model to target device (CUDA/MPS/CPU)

    Args:
        device: Target hardware for model placement
        num_classes: Number of dataset classes for classification head
        in_channels: Input channels (1=Grayscale, 3=RGB)
        cfg: Global configuration with pretrained/dropout settings

    Returns:
        Adapted EfficientNet-B0 model deployed to device
    """

    # --- Step 1: Initialize with Optional Pretraining ---
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if cfg.model.pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # Snapshot original first conv layer (before replacement)
    old_conv = model.features[0][0]

    # --- Step 2: Adapt First Convolutional Layer ---
    # EfficientNet expects 3-channel input; modify for grayscale if needed
    new_conv = nn.Conv2d(
        in_channels=in_channels,  # Custom: 1 or 3
        out_channels=32,  # EfficientNet standard
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        bias=False,
    )

    # --- Step 3: Weight Morphing (Transfer Pretrained Knowledge) ---
    if cfg.model.pretrained:
        with torch.no_grad():
            w = old_conv.weight  # Shape: [32, 3, 3, 3]

            # For grayscale: compress RGB channels by averaging
            # Preserves learned edge detectors while adapting to 1-channel input
            if in_channels == 1:
                w = w.mean(dim=1, keepdim=True)  # [32, 1, 3, 3]

            new_conv.weight.copy_(w)

    # Replace entry layer with adapted version
    model.features[0][0] = new_conv

    # --- Step 4: Modify Classification Head ---
    # Replace ImageNet 1000-class head with dataset-specific projection
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)  # 1280 features

    # --- Step 5: Device Placement ---
    model = model.to(device)

    return model
