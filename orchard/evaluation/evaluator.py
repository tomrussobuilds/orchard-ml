"""
Evaluation Engine Module.

Runs batch-level inference on a labelled test set and consolidates
predictions into global classification metrics (accuracy, macro F1,
macro AUC). Supports optional Test-Time Augmentation via the ``tta``
sub-module, applying domain-aware transforms (anatomical, texture)
and averaging softmax outputs across the ensemble.

Key Functions:

- ``evaluate_model``: Full-dataset evaluation with optional TTA,
  returning predictions, labels, metric dict, and macro F1.

Example:
    >>> preds, labels, metrics, f1 = evaluate_model(
    ...     model, test_loader, device, use_tta=True, cfg=cfg
    ... )
    >>> print(f"Test AUC: {metrics['auc']:.4f}")
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..core import LOGGER_NAME, LogStyle
from ..core.config import AugmentationConfig
from ..core.paths import METRIC_ACCURACY, METRIC_AUC, METRIC_F1
from .metrics import compute_classification_metrics
from .tta import adaptive_tta_predict

# EVALUATION ENGINE
logger = logging.getLogger(LOGGER_NAME)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    use_tta: bool = False,
    is_anatomical: bool = False,
    is_texture_based: bool = False,
    aug_cfg: AugmentationConfig | None = None,
    resolution: int = 28,
) -> tuple[np.ndarray, np.ndarray, dict, float]:
    """
    Performs full-set evaluation and coordinates metric calculation.

    Args:
        model: The trained neural network.
        test_loader: DataLoader for the evaluation set.
        device: Hardware target (CPU/CUDA/MPS).
        use_tta: Flag to enable Test-Time Augmentation.
        is_anatomical: Dataset-specific orientation constraint.
        is_texture_based: Dataset-specific texture preservation flag.
        aug_cfg: Augmentation sub-configuration (required for TTA).
        resolution: Dataset resolution for TTA intensity scaling.

    Returns:
        tuple[np.ndarray, np.ndarray, dict, float]: A 4-tuple of:

            - **all_preds** -- Predicted class indices, shape ``(N,)``
            - **all_labels** -- Ground truth labels, shape ``(N,)``
            - **metrics** -- Dict with keys ``accuracy``, ``auc``, ``f1``
            - **macro_f1** -- Macro-averaged F1 score (convenience shortcut)
    """
    model.eval()
    all_probs_list: list[np.ndarray] = []
    all_labels_list: list[np.ndarray] = []

    actual_tta = use_tta and (aug_cfg is not None)

    with torch.no_grad():
        for inputs, targets in test_loader:
            if actual_tta:
                # TTA logic handles its own device placement and softmax
                probs = adaptive_tta_predict(
                    model, inputs, device, is_anatomical, is_texture_based, aug_cfg, resolution
                )
            else:
                # Standard forward pass
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
        f"{LogStyle.INDENT}{LogStyle.ARROW} {'Test Metrics':<18}: "
        f"Acc: {metrics[METRIC_ACCURACY]:.4f} | "
        f"AUC: {metrics[METRIC_AUC]:.4f} | F1: {metrics[METRIC_F1]:.4f}"
    )
    if actual_tta and aug_cfg is not None:
        mode = aug_cfg.tta_mode.upper()
        log_msg += f" | TTA ENABLED (Mode: {mode})"

    logger.info(log_msg)

    return all_preds, all_labels, metrics, metrics[METRIC_F1]
