"""
Metrics Computation Module

Provides a standardized interface for calculating classification performance
metrics from model outputs. Isolates statistical logic from inference loops.
"""

# Standard Imports
import logging

# Third-Party Imports
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

# Internal Imports
from orchard.core import LOGGER_NAME

# METRIC LOGIC
logger = logging.getLogger(LOGGER_NAME)


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
        # One-vs-Rest ROC-AUC calculation
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except Exception as e:
        logger.warning(f"ROC-AUC calculation failed: {e}. Defaulting to 0.0")
        auc = 0.0

    return {"accuracy": float(accuracy), "auc": float(auc), "f1": float(macro_f1)}
