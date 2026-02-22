"""
Metric extraction utilities.

This module provides a helper class for extracting, validating,
and tracking metrics from validation results during training
or evaluation workflows.
"""

from __future__ import annotations


# METRIC EXTRACTOR
class MetricExtractor:
    """
    Extracts and tracks metrics from validation results.

    Handles metric extraction with validation and maintains
    the best metric value achieved during training.

    Attributes:
        metric_name: Name of metric to track (e.g., 'auc', 'accuracy')
        best_metric: Best metric value achieved so far

    Example:
        >>> extractor = MetricExtractor("auc")
        >>> val_metrics = {"loss": 0.5, "accuracy": 0.85, "auc": 0.92}
        >>> current = extractor.extract(val_metrics)  # 0.92
        >>> best = extractor.update_best(current)  # 0.92
    """

    def __init__(self, metric_name: str) -> None:
        """
        Initialize metric extractor.

        Args:
            metric_name: Name of metric to track
        """
        self.metric_name = metric_name
        self.best_metric = -float("inf")

    def extract(self, val_metrics: dict[str, float]) -> float:
        """
        Extract target metric from validation results.

        Args:
            val_metrics: Dictionary of validation metrics

        Returns:
            Value of target metric

        Raises:
            KeyError: If metric_name not found in val_metrics
        """
        if self.metric_name not in val_metrics:
            available = list(val_metrics.keys())
            raise KeyError(f"Metric '{self.metric_name}' not found. Available: {available}")
        return val_metrics[self.metric_name]

    def reset(self) -> None:
        """Reset best metric tracking for a new trial."""
        self.best_metric = -float("inf")

    def update_best(self, current_metric: float) -> float:
        """
        Update and return best metric achieved within current trial.

        Args:
            current_metric: Current metric value

        Returns:
            Best metric value (max of current and previous best)
        """
        self.best_metric = max(self.best_metric, current_metric)

        return self.best_metric
