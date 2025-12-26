"""
Evaluation Engine Module

This module handles the core inference logic, including standard prediction 
and Test-Time Augmentation (TTA). It focuses on generating model outputs 
and calculating performance metrics without visualization overhead.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
from typing import Tuple, List
import logging

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from sklearn.metrics import f1_score

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from src.core import Config

# =========================================================================== #
#                               EVALUATION LOGIC                              #
# =========================================================================== #

# Global logger instance
logger = logging.getLogger("medmnist_pipeline")

def tta_predict_batch(
    model: nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
    is_anatomical: bool,
    cfg: Config
) -> torch.Tensor:
    """
    Performs Test-Time Augmentation (TTA) inference on a batch of inputs.

    Applies a set of standard augmentations in addition to the original input. 
    Predictions from all augmented versions are averaged in the probability space.
    If is_anatomical is True, it restricts augmentations to orientation-preserving
    transforms to avoid confusing the model with physically impossible organ positions.
    Hardware-awareness is implemented to toggle between Full and Light TTA modes.

    Args:
        model (nn.Module): The trained PyTorch model.
        inputs (torch.Tensor): The batch of test images.
        device (torch.device): The device to run the inference on.
        is_anatomical (bool): Whether the dataset has fixed anatomical orientation.
        cfg (Config): The global configuration object containing TTA parameters.

    Returns:
        torch.Tensor: The averaged softmax probability predictions (mean ensemble).
    """
    model.eval()
    inputs = inputs.to(device)
    
    # 1. BASE TRANSFORMS: Safe for all medical datasets (including anatomical ones)
    # These parameters are now sourced directly from cfg.augmentation
    transforms = [
        lambda x: x,  # Original identity
        lambda x: TF.affine(
            x, angle=0, 
            translate=(cfg.augmentation.tta_translate, cfg.augmentation.tta_translate), 
            scale=1.0, shear=0
        ),
        lambda x: TF.affine(
            x, angle=0, translate=(0, 0), 
            scale=cfg.augmentation.tta_scale, shear=0
        ),
        lambda x: TF.gaussian_blur(
            x, kernel_size=3, sigma=cfg.augmentation.tta_blur_sigma
        ),
        lambda x: (x + 0.01 * torch.randn_like(x)).clamp(0, 1)  # Gaussian Noise
    ]

    # 2. ADVANCED TRANSFORMS: Geometric augmentations
    # Only enabled for non-anatomical data and non-CPU devices to optimize performance
    if not is_anatomical and device.type != "cpu":
        transforms.extend([
            lambda x: torch.flip(x, dims=[3]),           # Horizontal flip
            lambda x: torch.rot90(x, k=1, dims=[2, 3]),  # 90 degree rotation
            lambda x: torch.rot90(x, k=2, dims=[2, 3]),  # 180 degree rotation
            lambda x: torch.rot90(x, k=3, dims=[2, 3]),  # 270 degree rotation
        ])
    elif not is_anatomical and device.type == "cpu":
        # Light CPU fallback: Horizontal flip only to reduce overhead
        transforms.append(lambda x: torch.flip(x, dims=[3]))

    # 3. ENSEMBLE EXECUTION: Iterative probability accumulation to save VRAM
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


def evaluate_model(
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        use_tta: bool = False,
        is_anatomical: bool = False,
        cfg: Config = None
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Evaluates the model on the test set, optionally using Test-Time Augmentation (TTA).

    Args:
        model (nn.Module): The trained PyTorch model.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): The device (CPU/CUDA/MPS) to run the evaluation on.
        use_tta (bool, optional): Whether to enable TTA prediction. Defaults to False.
        is_anatomical (bool): Whether the dataset has fixed anatomical orientation.
        cfg (Config, optional): The global configuration for TTA hyperparams.

    Returns:
        Tuple[np.ndarray, np.ndarray, float, float]:
            all_preds: Array of model predictions (indices).
            all_labels: Array of true labels (indices).
            accuracy: Test set accuracy score.
            macro_f1: Test set Macro F1-score.
    """
    model.eval()
    all_preds_list: List[np.ndarray] = []
    all_labels_list: List[np.ndarray] = []

    # Safeguard: Ensure TTA only runs if config is provided
    actual_tta = use_tta and (cfg is not None)

    with torch.no_grad():
        for inputs, targets in test_loader:
            targets_np = targets.cpu().numpy()

            if actual_tta:
                # Perform TTA inference
                outputs = tta_predict_batch(model, inputs, device, is_anatomical, cfg)
                batch_preds = outputs.argmax(dim=1).cpu().numpy()
            else:
                # Standard inference
                inputs = inputs.to(device)
                outputs = model(inputs)
                batch_preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds_list.append(batch_preds)
            all_labels_list.append(targets_np)

    # Consolidate results across all batches
    all_preds = np.concatenate(all_preds_list)
    all_labels = np.concatenate(all_labels_list)
    
    # Compute final metrics
    accuracy = np.mean(all_preds == all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    # Logging with hardware-aware context
    log_message = (
        f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%) | "
        f"Macro F1: {macro_f1:.4f}"
    )
    if actual_tta:
        tta_mode = 'Full' if device.type != 'cpu' else 'Light/CPU'
        log_message += f" | TTA ENABLED (Mode: {tta_mode})"
    
    logger.info(log_message)

    return all_preds, all_labels, accuracy, macro_f1