"""
Metrics Computation Module

Provides a standardized interface for calculating classification performance
metrics from model outputs. Isolates statistical logic from inference loops.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from ..core import LOGGER_NAME
from ..core.paths import METRIC_ACCURACY, METRIC_AUC, METRIC_F1

logger = logging.getLogger(LOGGER_NAME)


# METRIC LOGIC
def compute_classification_metrics(
    labels: np.ndarray, preds: np.ndarray, probs: np.ndarray
) -> dict:
    """
    Computes accuracy, macro-averaged F1, and macro-averaged ROC-AUC.

    Args:
        labels: Ground truth class indices.
        preds: Predicted class indices.
        probs: Softmax probability distributions.

    Returns:
        dict: A dictionary containing 'accuracy', 'auc', and 'f1'.
    """
    # Direct accuracy calculation via NumPy
    accuracy = np.mean(preds == labels)

    # Macro-averaged F1 for class-imbalance awareness
    macro_f1 = f1_score(labels, preds, average="macro")

    try:
        n_classes = probs.shape[1] if probs.ndim == 2 else 1
        if n_classes <= 2:
            # Binary: use probability of the positive class
            auc = roc_auc_score(labels, probs[:, 1] if probs.ndim == 2 else probs)
        else:
            # Multiclass: One-vs-Rest, macro-averaged
            auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except (ValueError, TypeError) as e:
        logger.warning(f"ROC-AUC calculation failed: {e}. Defaulting to 0.0")
        auc = 0.0

    # roc_auc_score returns nan for single-class labels (binary case)
    if np.isnan(auc):
        auc = 0.0

    return {METRIC_ACCURACY: float(accuracy), METRIC_AUC: float(auc), METRIC_F1: float(macro_f1)}
