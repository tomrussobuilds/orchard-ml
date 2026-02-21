"""
Test-Time Augmentation (TTA) Module

This module implements adaptive TTA strategies for robust inference.
It provides an ensemble-based prediction mechanism that respects
anatomical constraints and texture preservation requirements of medical imaging.

Transform selection is deterministic and hardware-independent: the same
``tta_mode`` config always produces the same ensemble regardless of whether
inference runs on CPU, CUDA, or MPS, guaranteeing cross-platform reproducibility.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from ..core import Config


# TTA HELPERS
def _get_tta_transforms(is_anatomical: bool, is_texture_based: bool, cfg: Config) -> List:
    """
    Internal factory to resolve the augmentation suite based on
    dataset constraints and configuration policy.

    Transform selection is deterministic and hardware-independent to ensure
    reproducible predictions across CPU, CUDA, and MPS devices.

    Transform selection logic:
        - Anatomical datasets: NO flips or rotations (orientation is diagnostic)
        - Texture-based datasets: Minimal transforms (texture patterns are fragile)
        - Non-anatomical + Non-texture: Full or light suite based on tta_mode config

    Args:
        is_anatomical: If True, preserves spatial orientation (no flips/rotations)
        is_texture_based: If True, avoids destructive pixel operations
        cfg: Configuration with TTA parameters and tta_mode policy

    Returns:
        List of transform functions to apply during TTA inference
    """
    tta_mode = cfg.augmentation.tta_mode

    # Scale TTA intensity relative to resolution.
    # A 2px shift on 224x224 (~0.9%) should map to ~0.25px on 28x28 (~0.9%).
    _TTA_BASELINE_RESOLUTION = 224  # Reference resolution for TTA scaling
    resolution = cfg.dataset.resolution
    scale_factor = resolution / _TTA_BASELINE_RESOLUTION
    tta_translate = cfg.augmentation.tta_translate * scale_factor
    tta_scale = 1.0 + (cfg.augmentation.tta_scale - 1.0) * scale_factor
    tta_blur_sigma = cfg.augmentation.tta_blur_sigma * scale_factor

    # 1. BASE TRANSFORMS: Always include identity
    t_list = [
        (lambda x: x),  # Original (always first)
    ]

    # 2. FLIP: Only for non-anatomical datasets
    # Anatomical data (CT scans, X-rays) has fixed orientation - flipping is invalid
    if not is_anatomical:
        t_list.append(lambda x: torch.flip(x, dims=[3]))  # Horizontal flip

    # 3. TEXTURE-BASED: Minimal augmentation to preserve pattern integrity
    # Texture datasets (dermoscopy, histology) rely on fine-grained patterns
    if is_texture_based:
        # Skip aggressive transforms - texture patterns are sensitive
        # Only identity (+ flip if non-anatomical) is applied
        pass
    else:
        # Capture scaled values for closures
        _translate = tta_translate
        _scale = tta_scale
        _sigma = max(tta_blur_sigma, 0.01)  # blur sigma must be > 0

        # Non-texture datasets can tolerate geometric/photometric perturbations
        t_list.extend(
            [
                (
                    lambda x: TF.affine(
                        x, angle=0, translate=(_translate, _translate), scale=1.0, shear=0
                    )
                ),
                (lambda x: TF.affine(x, angle=0, translate=(0, 0), scale=_scale, shear=0)),
                (lambda x: TF.gaussian_blur(x, kernel_size=3, sigma=_sigma)),
            ]
        )

    # 4. ADVANCED TRANSFORMS: Config-driven, hardware-independent
    # Rotations are valid only when spatial orientation is not diagnostic
    if not is_anatomical and not is_texture_based:
        if tta_mode == "full":
            t_list.extend(
                [
                    (lambda x: torch.rot90(x, k=1, dims=[2, 3])),  # 90°
                    (lambda x: torch.rot90(x, k=2, dims=[2, 3])),  # 180°
                    (lambda x: torch.rot90(x, k=3, dims=[2, 3])),  # 270°
                ]
            )
        else:
            # Light mode: vertical flip instead of rotations (faster on CPU)
            t_list.append(lambda x: torch.flip(x, dims=[2]))

    return t_list


# CORE TTA LOGIC
def adaptive_tta_predict(
    model: nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
    is_anatomical: bool,
    is_texture_based: bool,
    cfg: Config,
) -> torch.Tensor:
    """
    Performs Test-Time Augmentation (TTA) inference on a batch of inputs.

    Applies a set of standard augmentations in addition to the original input.
    Predictions from all augmented versions are averaged in the probability space.
    If is_anatomical is True, it restricts augmentations to orientation-preserving
    transforms. If is_texture_based is True, it disables destructive pixel-level
    noise/blur to preserve local patterns. The ``tta_mode`` config field controls
    ensemble complexity (full vs light) independently of hardware.

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
    transforms = _get_tta_transforms(is_anatomical, is_texture_based, cfg)

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
    assert ensemble_probs is not None, "TTA transforms list cannot be empty"  # nosec B101
    return ensemble_probs / len(transforms)
