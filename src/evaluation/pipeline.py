"""
Evaluation and Reporting Package

This package coordinates model inference, performance visualization, 
and structured experiment reporting using a memory-efficient Lazy approach.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
from typing import Tuple, List
import logging
from pathlib import Path

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from src.core import Config, RunPaths, LOGGER_NAME
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
logger = logging.getLogger(LOGGER_NAME)


def run_final_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    train_losses: List[float],
    val_metrics_history: List[dict],
    class_names: List[str],
    paths: RunPaths,
    cfg: Config,
    aug_info: str = "N/A",
    log_path: Path | None = None
) -> Tuple[float, float]:
    """
    Executes the complete evaluation pipeline. 
    
    Coordinates full-set inference (with TTA support), visualizes metrics, 
    and generates the final structured Excel report.
    """
    
    # Resolve device from config
    device = torch.device(cfg.system.device)

    # --- 1) Inference & Metrics ---
    # Performance on the full test set
    all_preds, all_labels, test_metrics, macro_f1 = evaluate_model(
        model,
        test_loader,
        device=device,
        use_tta=cfg.training.use_tta,
        is_anatomical=cfg.dataset.metadata.is_anatomical,
        is_texture_based=cfg.dataset.metadata.is_texture_based,
        cfg=cfg
    )

    # --- 2) Visualizations ---
    # Diagnostic Confusion Matrix
    plot_confusion_matrix(
        all_labels=all_labels,
        all_preds=all_preds,
        classes=class_names,
        out_path=paths.get_fig_path("confusion_matrix.png"),
        cfg=cfg
    )

    # Historical Training Curves
    val_acc_list = [m['accuracy'] for m in val_metrics_history]
    plot_training_curves(
        train_losses=train_losses,
        val_accuracies=val_acc_list,
        out_path=paths.get_fig_path("training_curves.png"), 
        cfg=cfg
    )

    # Lazy-loaded prediction grid (samples from loader)
    show_predictions(
        model=model,
        loader=test_loader,
        device=device,
        classes=class_names,
        save_path=paths.get_fig_path("sample_predictions.png"),
        cfg=cfg
    )

    # --- 3) Structured Reporting ---
    # Aggregates everything into a formatted Excel summary
    final_log = log_path if log_path is not None else (paths.logs / "run.log")

    report = create_structured_report(
        val_metrics=val_metrics_history,
        test_metrics=test_metrics,
        macro_f1=macro_f1,
        train_losses=train_losses,
        best_path=paths.best_model_path,
        log_path=final_log,
        cfg=cfg,
        aug_info=aug_info
    )
    report.save(paths.final_report_path)

    test_acc = test_metrics['accuracy']
    logger.info(f"Final Evaluation Phase Complete. Test Acc: {test_acc:.4f}")
    
    return macro_f1, test_acc