"""
ResNet-18 Adapted for Low-Resolution Medical Imaging (28x28).

Performs "architectural surgery" on standard ResNet-18 to preserve spatial
resolution for small medical images. Uses bicubic weight interpolation to
transfer ImageNet knowledge while adapting to reduced input dimensions.

Key Modifications:
    - 7x7 Conv1 → 3x3 Conv1 (stride 1 instead of 2)
    - MaxPool removed (prevents 75% spatial loss)
    - Weight morphing via bicubic interpolation
    - Grayscale → RGB channel compression
"""

# Third-Party Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Internal Imports
from orchard.core import Config


# MODEL BUILDER
def build_resnet18_adapted(
    device: torch.device, num_classes: int, in_channels: int, cfg: Config
) -> nn.Module:
    """
    Constructs ResNet-18 optimized for 28x28 MedMNIST datasets.

    Standard ResNet aggressively downsamples (224→56→28→14), unsuitable for
    28x28 inputs. This adaptation preserves native resolution through stem
    modification and MaxPool bypass.

    Workflow:
        1. Load ImageNet pretrained ResNet-18 (if enabled)
        2. Replace 7x7 conv1 with 3x3 (stride 1 for spatial preservation)
        3. Interpolate 7x7 weights to 3x3 using bicubic smoothing
        4. Compress RGB→Grayscale if dataset is single-channel
        5. Remove MaxPool (prevents additional 2x downsampling)
        6. Replace classification head with dataset-specific linear layer

    Args:
        device: Target hardware for model placement
        num_classes: Number of dataset classes
        in_channels: Input channels (1=Grayscale, 3=RGB)
        cfg: Global configuration with pretrained settings

    Returns:
        Adapted ResNet-18 deployed to device
    """

    # --- Step 1: Initialize with Optional Pretraining ---
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if cfg.model.pretrained else None
    model = models.resnet18(weights=weights)

    # Snapshot original conv1 (before replacement)
    old_conv = model.conv1  # Shape: [64, 3, 7, 7]

    # --- Step 2: Re-engineer Input Layer ---
    # 3x3 kernel with stride 1 maintains 28x28 spatial dimensions
    new_conv = nn.Conv2d(
        in_channels=in_channels,  # Custom: 1 or 3
        out_channels=64,  # ResNet standard
        kernel_size=3,  # Reduced from 7
        stride=1,  # Reduced from 2 (key change!)
        padding=1,
        bias=False,
    )

    # --- Step 3: Weight Morphing (Knowledge Distillation) ---
    if cfg.model.pretrained:
        with torch.no_grad():
            w = old_conv.weight  # [64, 3, 7, 7]

            # Downsample 7x7 kernels to 3x3 using bicubic interpolation
            # Preserves learned edge detectors while adapting to smaller receptive field
            w = F.interpolate(w, size=(3, 3), mode="bicubic", align_corners=True)
            # Result: [64, 3, 3, 3]

            # For grayscale: compress RGB channels by averaging
            # Simulates brightness-preserving conversion
            if in_channels == 1:
                w = w.mean(dim=1, keepdim=True)  # [64, 1, 3, 3]

            new_conv.weight.copy_(w)

    # Replace entry layer with spatially-optimized version
    model.conv1 = new_conv

    # --- Step 4: Remove MaxPool (Prevent Additional Downsampling) ---
    # Standard Imports
    # Adapted: 28x28 → 28x28 via Identity (no downsampling)
    model.maxpool = nn.Identity()

    # --- Step 5: Replace Classification Head ---
    # ImageNet 1000 classes → Dataset-specific classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # --- Step 6: Device Placement ---
    model = model.to(device)

    return model
