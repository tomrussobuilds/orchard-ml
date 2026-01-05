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
from sklearn.metrics import f1_score, roc_auc_score

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from src.core import Config

# =========================================================================== #
#                               EVALUATION LOGIC                              #
# =========================================================================== #

# Global logger instance
logger = logging.getLogger("medmnist_pipeline")

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

    def _get_tta_transforms() -> List:
        """
        Internal factory to resolve the augmentation suite based on 
        dataset constraints and hardware capabilities.
        """
        # 1. BASE TRANSFORMS: Safe for all medical datasets
        t_list = [
            lambda x: x,  # Original identity
            lambda x: torch.flip(x, dims=[3]),  # Horizontal flip
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

    # --- MAIN EXECUTION FLOW ---
    model.eval()
    inputs = inputs.to(device)
    
    # Generate the suite of transforms via internal factory
    transforms = _get_tta_transforms()

    # 4. ENSEMBLE EXECUTION: Iterative probability accumulation to save VRAM
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
    is_texture_based: bool = False,
    cfg: Config = None
) -> Tuple[np.ndarray, np.ndarray, dict, float]:
    """
    Evaluates the model on the test set, optionally using Test-Time Augmentation (TTA).

    This function executes full-set inference and calculates classification 
    accuracy, macro-averaged F1-score, and macro-averaged ROC-AUC. If TTA is 
    active, the metrics are computed based on the averaged probability 
    distribution across multiple augmented versions of the input.

    Args:
        model (nn.Module): The trained PyTorch model.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): Hardware target (CUDA, MPS, or CPU).
        use_tta (bool): Whether to enable Test-Time Augmentation prediction.
        is_anatomical (bool): True if dataset has fixed anatomical orientation.
        is_texture_based (bool): True if dataset requires fine texture preservation.
        cfg (Config, optional): Configuration object containing TTA hyperparameters.

    Returns:
        Tuple[np.ndarray, np.ndarray, dict, float]:
            - all_preds: Array of class predictions (indices).
            - all_labels: Array of ground truth labels.
            - test_metrics: Dictionary with {'accuracy': float, 'auc': float}.
            - macro_f1: Macro-averaged F1 score.
    """
    model.eval()
    all_preds_list: List[np.ndarray] = []
    all_labels_list: List[np.ndarray] = []
    all_probs_list: List[np.ndarray] = []

    actual_tta = use_tta and (cfg is not None)

    with torch.no_grad():
        for inputs, targets in test_loader:
            targets_np = targets.cpu().numpy()

            if actual_tta:
                # adaptive_tta_predict returns averaged softmax probabilities
                probs = adaptive_tta_predict(
                    model, inputs, device, 
                    is_anatomical, is_texture_based, cfg
                )
            else:
                inputs = inputs.to(device)
                logits = model(inputs)
                probs = torch.softmax(logits, dim=1)

            # Move results to CPU and store for global metric calculation
            probs_np = probs.cpu().numpy()
            batch_preds = probs_np.argmax(axis=1)

            all_preds_list.append(batch_preds)
            all_labels_list.append(targets_np)
            all_probs_list.append(probs_np)

    # --- Data Consolidation ---
    all_preds = np.concatenate(all_preds_list)
    all_labels = np.concatenate(all_labels_list)
    all_probs = np.concatenate(all_probs_list)
    
    # --- Final Metric Computation ---
    accuracy = np.mean(all_preds == all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    
    # Compute ROC-AUC (One-vs-Rest, Macro)
    try:
        # Handles binary and multiclass cases automatically via OvR
        auc = roc_auc_score(
            all_labels, 
            all_probs, 
            multi_class="ovr", 
            average="macro"
        )
    except Exception as e:
        logger.warning(f"ROC-AUC calculation failed: {e}. Defaulting to 0.0")
        auc = 0.0

    # Bundle metrics for the Reporting module
    test_metrics = {
        "accuracy": float(accuracy),
        "auc": float(auc)
    }

    # --- Logging ---
    log_message = (
        f"Test Metrics -> Acc: {accuracy:.4f} | "
        f"AUC: {auc:.4f} | F1: {macro_f1:.4f}"
    )
    if actual_tta:
        tta_mode = 'Full' if device.type != 'cpu' else 'Light/CPU'
        log_message += f" | TTA ENABLED (Mode: {tta_mode})"
    
    logger.info(log_message)

    return all_preds, all_labels, test_metrics, macro_f1