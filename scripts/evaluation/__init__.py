"""
Evaluation and Reporting Package exposing core evaluation utilities.
"""

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from .engine import evaluate_model

from .reporting import TrainingReport, create_structured_report

from .visualization import (
    plot_confusion_matrix,
    plot_training_curves,
    show_predictions
)

from .pipeline import run_final_evaluation