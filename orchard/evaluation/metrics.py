"""
Metrics Computation Module.

Provides a standardized interface for calculating classification performance
metrics from model outputs. Isolates statistical logic from inference loops.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import f1_score

from ..core.paths import METRIC_ACCURACY, METRIC_AUC, METRIC_F1
from ..trainer.engine import compute_auc as compute_auc  # re-exported via evaluation/__init__.py


# METRIC LOGIC
def compute_classification_metrics(
    labels: npt.NDArray[Any], preds: npt.NDArray[Any], probs: npt.NDArray[Any]
) -> dict[str, float]:
    """
    Computes accuracy, macro-averaged F1, and macro-averaged ROC-AUC.

    Args:
        labels: Ground truth class indices.
        preds: Predicted class indices.
        probs: Softmax probability distributions.

    Returns:
        dict[str, float]: Metric dictionary with keys:

            - ``accuracy`` -- Overall classification accuracy
            - ``auc`` -- Macro-averaged ROC-AUC (NaN if computation fails)
            - ``f1`` -- Macro-averaged F1 score
    """
    accuracy = np.mean(preds == labels)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0.0)
    auc = compute_auc(labels, probs)

    return {METRIC_ACCURACY: float(accuracy), METRIC_AUC: float(auc), METRIC_F1: float(macro_f1)}
