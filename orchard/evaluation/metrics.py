"""
Metrics Computation Module.

Provides a standardized interface for calculating classification performance
metrics from model outputs. Isolates statistical logic from inference loops.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from ..core.paths import METRIC_ACCURACY, METRIC_AUC, METRIC_F1
from ..trainer.engine import (  # noqa: F401 (re-exported via evaluation/__init__.py)
    compute_auc,
)


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
        dict[str, float]: Metric dictionary with keys:

            - ``accuracy`` -- Overall classification accuracy
            - ``auc`` -- Macro-averaged ROC-AUC (0.0 if computation fails)
            - ``f1`` -- Macro-averaged F1 score
    """
    accuracy = np.mean(preds == labels)
    macro_f1 = f1_score(labels, preds, average="macro")
    auc = compute_auc(labels, probs)

    return {METRIC_ACCURACY: float(accuracy), METRIC_AUC: float(auc), METRIC_F1: float(macro_f1)}
