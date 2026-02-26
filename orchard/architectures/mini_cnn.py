"""
Lightweight CNN for 28×28 Low-Resolution Medical Imaging.

Custom compact architecture designed specifically for small-scale datasets
like 28×28 low-resolution datasets. Provides a fast baseline alternative to adapted ResNet-18
with reduced parameter count and computational overhead.

Key Features:

- Minimal Depth: 3 convolutional blocks for rapid convergence
- Adaptive Pooling: Maintains spatial resolution through progressive downsampling
- Dropout Regularization: Configurable dropout for overfitting prevention
- Efficient Design: ~50K parameters vs ~11M for ResNet-18

Architecture:
    Input [28×28×C] → Conv1 [14×14×32] → Conv2 [7×7×64] → Conv3 [7×7×128]
                   → AdaptiveAvgPool [1×1×128] → Dropout → FC [num_classes]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..core import Config


# MODEL DEFINITION
class MiniCNN(nn.Module):
    """Compact CNN optimized for 28×28 resolution datasets."""

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.0) -> None:
        """
        Initialize MiniCNN architecture.

        Args:
            in_channels: Input channels (1=Grayscale, 3=RGB)
            num_classes: Number of output classes
            dropout: Dropout probability before final FC layer
        """
        super().__init__()

        # Block 1: 28×28 → 14×14 (spatial compression)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28 → 14
        )

        # Block 2: 14×14 → 7×7 (feature extraction)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 14 → 7
        )

        # Block 3: 7×7 → 7×7 (deep features)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )

        # Global pooling + classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def build_mini_cnn(
    device: torch.device, num_classes: int, in_channels: int, cfg: Config
) -> nn.Module:
    """
    Constructs MiniCNN for low-resolution medical imaging.

    Args:
        device: Target hardware for model placement
        num_classes: Number of dataset classes
        in_channels: Input channels (1=Grayscale, 3=RGB)
        cfg: Global configuration with dropout settings

    Returns:
        MiniCNN model deployed to device
    """
    model = MiniCNN(
        in_channels=in_channels, num_classes=num_classes, dropout=cfg.architecture.dropout
    )

    return model.to(device)
