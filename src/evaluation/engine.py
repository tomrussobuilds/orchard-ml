"""
Evaluation Engine Module

This module handles the core inference logic, including standard prediction 
and Test-Time Augmentation (TTA). It focuses on generating model outputs 
and calculating performance metrics without visualization overhead.
"""

# =========================================================================== #
#                                Standard Imports
# =========================================================================== #
from typing import Tuple, List
import logging

# =========================================================================== #
#                                Third-Party Imports
# =========================================================================== #
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from sklearn.metrics import f1_score

# =========================================================================== #
#                               EVALUATION LOGIC
# =========================================================================== #
# Global logger instance
logger = logging.getLogger("medmnist_pipeline")

def tta_predict_batch(
    model: nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Performs Test-Time Augmentation (TTA) inference on a batch of inputs.

    Applies a set of 6 standard augmentations (flips, rotations, light blur,
    and Gaussian noise) in addition to the original input (7 total). Predictions
    from all augmented versions are averaged in the probability space.

    Args:
        model (nn.Module): The trained PyTorch model.
        inputs (torch.Tensor): The batch of test images.
        device (torch.device): The device to run the inference on.

    Returns:
        torch.Tensor: The averaged softmax probability predictions (mean ensemble).
    """
    model.eval()
    inputs = inputs.to(device)
    
    # Define a list of augmented versions of the input batch (7-way ensemble)
    augs: List[torch.Tensor] = [
        inputs,
        torch.flip(inputs, dims=[3]),           # Horizontal flip
        torch.rot90(inputs, k=1, dims=[2, 3]),  # 90 degree rotation
        torch.rot90(inputs, k=2, dims=[2, 3]),  # 180 degree rotation
        torch.rot90(inputs, k=3, dims=[2, 3]),  # 270 degree rotation
        TF.gaussian_blur(inputs, kernel_size=3, sigma=0.8), # Light Gaussian blur
        # Add small Gaussian noise and clamp
        (inputs + 0.01 * torch.randn_like(inputs)).clamp(0, 1),
    ]
    
    preds: List[torch.Tensor] = []
    with torch.no_grad():
        for aug in augs:
            logits = model(aug)
            # Use softmax output for averaging (ensemble in probability space)
            preds.append(F.softmax(logits, dim=1))
    
    # Stack all predictions and take the mean along the batch dimension
    return torch.stack(preds).mean(0)


def evaluate_model(
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        use_tta: bool = False
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Evaluates the model on the test set, optionally using Test-Time Augmentation (TTA).

    Args:
        model (nn.Module): The trained PyTorch model.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): The device (CPU/CUDA) to run the evaluation on.
        use_tta (bool, optional): Whether to enable TTA prediction. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray, float, float]:
            all_preds: Array of model predictions.
            all_labels: Array of true labels.
            accuracy: Test set accuracy.
            macro_f1: Test set Macro F1-score.
    """
    model.eval()
    all_preds_list: List[np.ndarray] = []
    all_labels_list: List[np.ndarray] = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            targets_np = targets.cpu().numpy()

            if use_tta:
                outputs = tta_predict_batch(model, inputs, device)
                batch_preds = outputs.argmax(dim=1).cpu().numpy()
            
            else:
                inputs = inputs.to(device)
                outputs = model(inputs)
                batch_preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds_list.append(batch_preds)
            all_labels_list.append(targets_np)

    # Efficient concatenation of batch results
    all_preds = np.concatenate(all_preds_list)
    all_labels = np.concatenate(all_labels_list)
    
    # Calculate performance metrics
    accuracy = np.mean(all_preds == all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    log_message = (
        f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%) | "
        f"Macro F1: {macro_f1:.4f}"
    )
    if use_tta:
        log_message += " | TEST-TIME AUGMENTATION ENABLED"
    
    logger.info(log_message)

    return all_preds, all_labels, accuracy, macro_f1