"""
Model Orchestration and Architecture Definition Module

This module provides a factory for deep learning architectures adapted for 
the MedMNIST ecosystem. It specializes in fine-tuning standard Torchvision 
models to handle low-resolution biomedical images (28x28 pixels).

Key Architectural Adaptations:
1. Spatial Preservation: Replaces the standard ResNet 7x7 (stride 2) entry 
   convolution with a 3x3 (stride 1) layer and removes initial pooling. This 
   prevents excessive information loss in the early stages of feature extraction.
2. Cross-Modal Weight Transfer: Implements bicubic interpolation of pre-trained 
   ImageNet weights and handles channel-depth conversion (e.g., RGB to Grayscale).
3. Dynamic Head Reconfiguration: Automatically adjusts the final linear 
   layers based on the target dataset's class cardinality.
"""
# =========================================================================== #
#                                Standard Imports
# =========================================================================== #
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
from src.core import Config

# =========================================================================== #
#                               MODEL DEFINITION
# =========================================================================== #
# Global logger instance
logger = logging.getLogger("medmnist_pipeline")

def build_resnet18_adapted(
        device: torch.device,
        num_classes: int,
        in_channels: int = 3,
        cfg: Config | None = None
    ) -> nn.Module:
    """
    Loads a pre-trained ResNet-18 model (ImageNet weights) and adapts its
    structure for the BloodMNIST dataset (28x28 inputs).

    The adaptation steps are:
    1. Replace the original 7x7 `conv1` (stride 2) with a 3x3 `conv1` (stride 1)
       to avoid immediate downsampling.
    2. Remove the `maxpool` layer entirely to retain the 28x28 spatial resolution.
    3. Replace the final fully connected layer (`fc`) with one for the target classes.
    4. Bicubic interpolation and transfer of pre-trained weights from the old `7x7`
       kernel to the new `3x3` kernel.

    Args:
        device (torch.device): The device (CPU or CUDA) to move the model to.
        num_classes (int): Number of output classes.
        in_channels (int): Number of input channels.
        cfg (Config | None): Configuration object for logging.

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
        in_channels=in_channels,
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

        if in_channels == 1:
            # If input is grayscale, average the weights across the RGB channels
            w = w.mean(dim=1, keepdim=True)

        new_conv.weight[:] = w
    
    # Apply the adaptations to the model structure
    model.conv1 = new_conv
    
    # Remove the MaxPool layer by replacing it with an Identity function
    model.maxpool = nn.Identity()
    
    # 4. Replace the final classification head
    # The input feature size remains the same (e.g., 512 for ResNet18)
    # The output is set to the number of target classes (BloodMNIST)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Move the model to the specified device
    model = model.to(device)
    model_display_name = cfg.model_name if cfg else "ResNet-18"
    
    logger.info(
        f"{model_display_name} successfully ADAPTED: in_channels={in_channels}, "
        f"num_classes={num_classes} and moved to {device.type}."
    )

    return model