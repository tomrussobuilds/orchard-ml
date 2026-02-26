"""
Unit tests for evaluation metrics in the Orchard library.

This module provides a suite of tests for classification performance metrics,
ensuring robust calculation of accuracy, F1-score, and ROC-AUC, including
graceful handling of edge cases such as single-class labels.
"""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

from orchard.evaluation import compute_classification_metrics


@pytest.mark.unit
class TestClassificationMetrics:
    """
    Test suite for the compute_classification_metrics function.
    Covers accuracy, F1, and ROC-AUC calculations.
    """

    def test_compute_metrics_perfect_prediction(self):
        """Test behavior with 100% correct predictions."""
        labels = np.array([0, 1, 2])
        preds = np.array([0, 1, 2])
        probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        results = compute_classification_metrics(labels, preds, probs)

        assert results["accuracy"] == pytest.approx(1.0)
        assert results["f1"] == pytest.approx(1.0)
        assert results["auc"] == pytest.approx(1.0)

    def test_compute_metrics_half_wrong(self):
        """Test accuracy and F1 with partially incorrect predictions."""
        labels = np.array([0, 0, 1, 1])
        preds = np.array([0, 1, 1, 0])
        probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.1, 0.9], [0.7, 0.3]])

        results = compute_classification_metrics(labels, preds, probs)

        assert results["accuracy"] == pytest.approx(0.5)
        assert 0.0 < results["f1"] < 1.0
        assert isinstance(results["auc"], float)

    @pytest.mark.parametrize("input_size", [10, 50, 100])
    def test_data_types_and_shapes(self, input_size):
        """Parametrized test to ensure consistency across different input sizes."""
        rng = np.random.default_rng(seed=42)

        labels = rng.integers(0, 2, size=input_size)
        preds = rng.integers(0, 2, size=input_size)
        probs = rng.random((input_size, 2))
        probs /= probs.sum(axis=1)[:, np.newaxis]

        results = compute_classification_metrics(labels, preds, probs)

        assert isinstance(results["accuracy"], float)
        assert isinstance(results["f1"], float)
        assert "accuracy" in results
        assert "auc" in results
        assert "f1" in results

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
    def test_auc_fallback_single_class(self):
        """AUC returns NaN when all labels are the same class."""
        labels = np.array([0, 0, 0, 0])
        preds = np.array([0, 0, 0, 0])
        probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])

        results = compute_classification_metrics(labels, preds, probs)

        assert math.isnan(results["auc"])

    def test_auc_fallback_value_error(self):
        """AUC returns NaN when roc_auc_score raises ValueError."""
        labels = np.array([0, 1])
        preds = np.array([0, 1])
        probs = np.array([[0.8, 0.2], [0.2, 0.8]])

        with patch(
            "orchard.trainer.engine.roc_auc_score",
            side_effect=ValueError("mocked"),
        ):
            results = compute_classification_metrics(labels, preds, probs)

        assert math.isnan(results["auc"])

    def test_return_types(self):
        """Ensure the returned dictionary contains standard Python floats (not NumPy types)."""
        labels = np.array([0, 1])
        preds = np.array([0, 1])
        probs = np.array([[0.8, 0.2], [0.2, 0.8]])

        results = compute_classification_metrics(labels, preds, probs)

        for key, value in results.items():
            assert isinstance(value, float), f"Key {key} is not a standard float"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
