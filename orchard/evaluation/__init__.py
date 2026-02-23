"""
Evaluation and Reporting Package

This package coordinates model inference, performance visualization,
and structured experiment reporting. It provides a unified interface to
assess model generalization through standard metrics and Test-Time
Augmentation (TTA), while automating the generation of artifacts
(plots, reports) for experimental tracking.
"""

from .evaluation_pipeline import run_final_evaluation
from .evaluator import evaluate_model
from .metrics import compute_auc, compute_classification_metrics
from .reporting import TrainingReport, create_structured_report
from .tta import _get_tta_transforms, adaptive_tta_predict
from .visualization import plot_confusion_matrix, plot_training_curves, show_predictions

__all__ = [
    # Inference & Evaluation
    "evaluate_model",
    "run_final_evaluation",
    # Visualizations
    "plot_confusion_matrix",
    "plot_training_curves",
    "show_predictions",
    # Reporting
    "TrainingReport",
    "create_structured_report",
    # TTA
    "adaptive_tta_predict",
    "_get_tta_transforms",
    # Metrics
    "compute_auc",
    "compute_classification_metrics",
]
