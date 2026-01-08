"""
Test-Time Augmentation (TTA) Module

This module implements adaptive TTA strategies for robust inference. 
It provides an ensemble-based prediction mechanism that respects 
anatomical constraints and texture preservation requirements of medical imaging.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
from typing import List

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from src.core import Config

# =========================================================================== #
#                                  TTA HELPERS                                #
# =========================================================================== #

def _get_tta_transforms(
    device: torch.device,
    is_anatomical: bool,
    is_texture_based: bool,
    cfg: Config
) -> List:
    """
    Internal factory to resolve the augmentation suite based on 
    dataset constraints and hardware capabilities.
    """
    # 1. BASE TRANSFORMS: Safe for all medical datasets
    t_list = [
        (lambda x: x),  # Original identity
        (lambda x: torch.flip(x, dims=[3])),  # Horizontal flip
    ]

    # 2. TEXTURE-AWARE TRANSFORMS: Geometric vs Pixel-level
    if is_texture_based:
        # Subtle shift: only 1px if image is small, to avoid losing detail
        t_list.append(lambda x: TF.affine(x, angle=0, translate=(1, 1), scale=1.0, shear=0))
    else:
        # Standard pixel-level augmentations for morphology-based data (Blood, Chest)
        t_list.extend([
            (lambda x: TF.affine(
                x, angle=0, 
                translate=(cfg.augmentation.tta_translate, cfg.augmentation.tta_translate), 
                scale=1.0, shear=0
            )),
            (lambda x: TF.affine(
                x, angle=0, translate=(0, 0), 
                scale=cfg.augmentation.tta_scale, shear=0
            )),
            # Gaussian Blur and Noise are the most destructive for texture
            (lambda x: TF.gaussian_blur(
                x, kernel_size=3, sigma=cfg.augmentation.tta_blur_sigma
            )),
            # Gaussian Noise addition with clamping to [0, 1]
            (lambda x: (x + 0.01 * torch.randn_like(x)).clamp(0, 1)),
        ])

    # 3. ADVANCED TRANSFORMS: Geometric augmentations
    # Only enabled for non-anatomical data and non-CPU devices
    if not is_anatomical and device.type != "cpu":
        t_list.extend([
            (lambda x: torch.rot90(x, k=1, dims=[2, 3])),  # 90 degree rotation
            (lambda x: torch.rot90(x, k=2, dims=[2, 3])),  # 180 degree rotation
            (lambda x: torch.rot90(x, k=3, dims=[2, 3])),  # 270 degree rotation
        ])
    elif not is_anatomical and device.type == "cpu":
        # Light CPU fallback: Additional flip only
        t_list.append((lambda x: torch.flip(x, dims=[2])))
        
    return t_list

# =========================================================================== #
#                               CORE TTA LOGIC                                #
# =========================================================================== #

def adaptive_tta_predict(
    model: nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
    is_anatomical: bool,
    is_texture_based: bool,
    cfg: Config
) -> torch.Tensor:
    """
    Performs Test-Time Augmentation (TTA) inference on a batch of inputs.

    Applies a set of standard augmentations in addition to the original input. 
    Predictions from all augmented versions are averaged in the probability space.
    If is_anatomical is True, it restricts augmentations to orientation-preserving
    transforms. If is_texture_based is True, it disables destructive pixel-level 
    noise/blur to preserve local patterns. Hardware-awareness is implemented 
    to toggle between Full and Light TTA modes.

    Args:
        model (nn.Module): The trained PyTorch model.
        inputs (torch.Tensor): The batch of test images.
        device (torch.device): The device to run the inference on.
        is_anatomical (bool): Whether the dataset has fixed anatomical orientation.
        is_texture_based (bool): Whether the dataset relies on high-frequency textures.
        cfg (Config): The global configuration object containing TTA parameters.

    Returns:
        torch.Tensor: The averaged softmax probability predictions (mean ensemble).
    """
    model.eval()
    inputs = inputs.to(device)
    
    # Generate the suite of transforms via module-level factory
    transforms = _get_tta_transforms(device, is_anatomical, is_texture_based, cfg)

    # ENSEMBLE EXECUTION: Iterative probability accumulation to save VRAM
    ensemble_probs = None
    
    with torch.no_grad():
        for t in transforms:
            aug_input = t(inputs)
            logits = model(aug_input)
            probs = F.softmax(logits, dim=1)
            
            if ensemble_probs is None:
                ensemble_probs = probs
            else:
                ensemble_probs += probs
    
    # Calculate the mean probability across all augmentation passes
    return ensemble_probs / len(transforms)