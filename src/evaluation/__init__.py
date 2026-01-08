"""
Evaluation and Reporting Package

This package coordinates model inference, performance visualization, 
and structured experiment reporting. It provides a unified interface to 
assess model generalization through standard metrics and Test-Time 
Augmentation (TTA), while automating the generation of artifacts 
(plots, reports) for experimental tracking.
"""

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from .engine import evaluate_model
from .visualization import (
    plot_confusion_matrix,
    plot_training_curves,
    show_predictions
)
from .reporting import (
    TrainingReport, 
    create_structured_report
)
from .pipeline import run_final_evaluation

# =========================================================================== #
#                                PACKAGE INTERFACE                            #
# =========================================================================== #

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
    "create_structured_report"
]