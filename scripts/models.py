"""
Model Definition Module

This module defines the architecture of the deep learning model used for the
BloodMNIST classification task. It leverages a pre-trained ResNet-18 model from
torchvision and adapts it specifically to handle the 28x28 pixel input size of
the BloodMNIST dataset. The primary modifications include adjusting the initial
convolutional layer and removing the first pooling layer to preserve spatial
resolution, as well as replacing the final classification head.
"""
# =========================================================================== #
#                                Standard Imports
# =========================================================================== #
from typing import Final
import logging

# =========================================================================== #
#                                Third-Party Imports
# =========================================================================== #
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# =========================================================================== #
#                                Internal Imports
# =========================================================================== #
from .utils import Logger

# Global logger instance
logger: Final[logging.Logger] = Logger().get_logger()


# =========================================================================== #
#                               MODEL DEFINITION
# =========================================================================== #

def get_model(device: torch.device) -> nn.Module:
    """
    Loads a pre-trained ResNet-18 model (ImageNet weights) and adapts its
    structure for the BloodMNIST dataset (28x28 inputs and 8 classes).

    The adaptation steps are:
    1. Replace the original 7x7 `conv1` (stride 2) with a 3x3 `conv1` (stride 1)
       to avoid immediate downsampling.
    2. Remove the `maxpool` layer entirely to retain the 28x28 spatial resolution.
    3. Replace the final fully connected layer (`fc`) with one for 8 classes.
    4. Interpolate and transfer weights from the old `conv1` to the new `conv1`.

    Args:
        device (torch.device): The device (CPU or CUDA) to move the model to.

    Returns:
        nn.Module: The adapted ResNet-18 model ready for training.
    """
    # 1. Load pre-trained ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Store the original conv1 layer for weight transfer
    old_conv = model.conv1

    # 2. Define the new initial convolution layer (3x3, stride 1)
    # The original ResNet-18 uses conv1(7x7, stride 2) and maxpool, which reduces
    # 28x28 input to 6x6, losing too much information. We replace it.
    new_conv = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,  # Smaller kernel
        stride=1,       # No immediate downsampling
        padding=1,
        bias=False
    )

    # 3. Transfer weights from the old 7x7 layer to the new 3x3 layer
    # This keeps the benefit of ImageNet pre-training.
    with torch.no_grad():
        w = old_conv.weight
        # Interpolate the 7x7 weights to 3x3 using bicubic interpolation
        w = F.interpolate(w, size=(3,3), mode='bicubic', align_corners=True)
        new_conv.weight[:] = w
    
    # Apply the adaptations to the model structure
    model.conv1 = new_conv
    
    # Remove the MaxPool layer by replacing it with an Identity function
    model.maxpool = nn.Identity()
    
    # 4. Replace the final classification head
    # The input feature size remains the same (e.g., 512 for ResNet18)
    # The output is set to 8 classes (BloodMNIST)
    model.fc = nn.Linear(model.fc.in_features, 8)
    
    # Move the model to the specified device
    model = model.to(device)
    
    logger.info(
        "ResNet-18 successfully ADAPTED for 28×28 inputs "
        "(3×3 conv1 + maxpool removed + head changed to 8 classes)"
    )

    return model