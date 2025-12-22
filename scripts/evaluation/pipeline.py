"""
Evaluation and Reporting Package

This package coordinates model inference, performance visualization, 
and structured experiment reporting.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
from typing import Tuple, List
import logging

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from scripts.core import Config, RunPaths
from .engine import evaluate_model
from .visualization import (
    plot_confusion_matrix, 
    plot_training_curves, 
    show_predictions
)
from .reporting import create_structured_report

# =========================================================================== #
#                               EVALUATION PIPELINE                           #
# =========================================================================== #
# Global logger instance
logger = logging.getLogger("medmnist_pipeline")


def run_final_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    class_names: List[str],
    train_losses: List[float],
    val_accuracies: List[float],
    device: torch.device,
    paths: RunPaths,
    cfg: Config,
    use_tta: bool = False,
    aug_info: str = "N/A"
) -> Tuple[float, float]:
    """
    Executes the complete evaluation pipeline, generating all figures, 
    metrics, and the final Excel report within the current Run directory.

    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader for test set inference.
        test_images: NumPy array of test images (for visualization).
        test_labels: NumPy array of true test labels.
        class_names: List of category labels from the dataset registry.
        train_losses: History of training losses.
        val_accuracies: History of validation accuracies.
        device: Torch device (cpu/cuda).
        paths: RunPaths instance for the current experiment.
        cfg: Configuration object.
        use_tta: Whether to use Test-Time Augmentation.
        aug_info: String description of applied augmentations.

    Returns:
        Tuple[float, float]: (Macro F1-score, Accuracy)
    """
    
    # --- 1) Inference & Metrics ---
    # Generates predictions and calculates core metrics using the engine
    all_preds, all_labels, test_acc, macro_f1 = evaluate_model(
        model, test_loader, device, use_tta=use_tta
    )

    # --- 2) Visualizations (Agnostic) ---
    # Confusion Matrix
    plot_confusion_matrix(
        all_labels=all_labels,
        all_preds=all_preds,
        classes=class_names,
        out_path=paths.figures / "confusion_matrix.png",
        cfg=cfg
    )

    # Training Curves (PNG + NPZ data)
    plot_training_curves(
        train_losses=train_losses,
        val_accuracies=val_accuracies,
        out_dir=paths.figures,
        cfg=cfg
    )

    # Sample Predictions Grid
    show_predictions(
        images=test_images,
        true_labels=test_labels,
        preds=all_preds,
        classes=class_names,
        n=12,
        save_path=paths.figures / "sample_predictions.png",
        cfg=cfg
    )

    # --- 3) Structured Reporting ---
    # Create and save the Excel report
    report = create_structured_report(
        val_accuracies=val_accuracies,
        macro_f1=macro_f1,
        test_acc=test_acc,
        train_losses=train_losses,
        best_path=paths.models / "best_model.pth",
        log_path=paths.logs / "run.log",
        cfg=cfg,
        aug_info=aug_info
    )
    report.save(paths.reports / "training_summary.xlsx")

    logger.info(
        f"Full evaluation complete for {cfg.model_name} on {cfg.dataset_name}. "
        f"Results saved in: {paths.root}"
    )
    
    return macro_f1, test_acc