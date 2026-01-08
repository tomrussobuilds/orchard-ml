"""
Evaluation Engine Module

Orchestrates the model inference lifecycle on test datasets.
Handles batch processing, TTA integration, and results consolidation.
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
from torch.utils.data import DataLoader

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from src.core import Config, LOGGER_NAME
from .tta import adaptive_tta_predict
from .metrics import compute_classification_metrics

# =========================================================================== #
#                               EVALUATION ENGINE                             #
# =========================================================================== #

logger = logging.getLogger(LOGGER_NAME)

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
    Performs full-set inference and coordinates metric calculation.
    
    Args:
        model: The trained neural network.
        test_loader: DataLoader for the evaluation set.
        device: Hardware target (CPU/CUDA/MPS).
        use_tta: Flag to enable Test-Time Augmentation.
        is_anatomical: Dataset-specific orientation constraint.
        is_texture_based: Dataset-specific texture preservation flag.
        cfg: Global configuration manifest.
        
    Returns:
        Tuple containing predictions, labels, metrics dict, and macro-f1 scalar.
    """
    model.eval()
    all_probs_list: List[np.ndarray] = []
    all_labels_list: List[np.ndarray] = []

    actual_tta = use_tta and (cfg is not None)

    with torch.no_grad():
        for inputs, targets in test_loader:
            if actual_tta:
                # TTA logic handles its own device placement and softmax
                probs = adaptive_tta_predict(
                    model, inputs, device, 
                    is_anatomical, is_texture_based, cfg
                )
            else:
                # Standard single-pass inference
                inputs = inputs.to(device)
                logits = model(inputs)
                probs = torch.softmax(logits, dim=1)

            all_probs_list.append(probs.cpu().numpy())
            all_labels_list.append(targets.numpy())

    # Consolidate batch results into global arrays
    all_probs = np.concatenate(all_probs_list)
    all_labels = np.concatenate(all_labels_list)
    all_preds = all_probs.argmax(axis=1)
    
    # Delegate statistical analysis to the metrics module
    metrics = compute_classification_metrics(all_labels, all_preds, all_probs)

    # Performance logging
    log_msg = (
        f"Test Metrics -> Acc: {metrics['accuracy']:.4f} | "
        f"AUC: {metrics['auc']:.4f} | F1: {metrics['f1']:.4f}"
    )
    if actual_tta:
        mode = 'Full' if device.type != 'cpu' else 'Light'
        log_msg += f" | TTA ENABLED (Mode: {mode})"
    
    logger.info(log_msg)

    return all_preds, all_labels, metrics, metrics['f1']