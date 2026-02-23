"""
Evaluation and Reporting Package

This package coordinates model inference, performance visualization,
and structured experiment reporting using a memory-efficient Lazy approach.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch.nn as nn
from torch.utils.data import DataLoader

from ..core import LOGGER_NAME, Config, RunPaths
from ..core.paths import METRIC_ACCURACY, METRIC_AUC

if TYPE_CHECKING:  # pragma: no cover
    from ..tracking import TrackerProtocol

from .evaluator import evaluate_model
from .reporting import create_structured_report
from .visualization import plot_confusion_matrix, plot_training_curves, show_predictions

# Global logger instance
logger = logging.getLogger(LOGGER_NAME)


# EVALUATION PIPELINE
def run_final_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    train_losses: list[float],
    val_metrics_history: list[dict],
    class_names: list[str],
    paths: RunPaths,
    cfg: Config,
    aug_info: str = "N/A",
    log_path: Path | None = None,
    tracker: TrackerProtocol | None = None,
) -> tuple[float, float, float]:
    """
    Execute the complete evaluation pipeline.

    Coordinates full-set inference (with TTA support), visualizes metrics,
    and generates the final structured report.

    Args:
        model: Trained model for evaluation (already on target device).
        test_loader: DataLoader for test set.
        train_losses: Training loss history per epoch.
        val_metrics_history: Validation metrics history per epoch.
        class_names: List of class label strings.
        paths: RunPaths for artifact output.
        cfg: Global configuration manifest.
        aug_info: Augmentation description string for report.
        log_path: Path to session log file for report embedding.
        tracker: Optional experiment tracker for final metrics.

    Returns:
        tuple of (macro_f1, test_acc, test_auc)
    """

    # Resolve device from model (already placed on the correct device by the trainer)
    device = next(model.parameters()).device

    # Filesystem-safe architecture tag (e.g. "timm/model" → "timm_model")
    arch_tag = cfg.architecture.name.replace("/", "_")

    # --- 1) Inference & Metrics ---
    # Performance on the full test set
    all_preds, all_labels, test_metrics, macro_f1 = evaluate_model(
        model,
        test_loader,
        device=device,
        use_tta=cfg.training.use_tta,
        is_anatomical=cfg.dataset.metadata.is_anatomical,
        is_texture_based=cfg.dataset.metadata.is_texture_based,
        cfg=cfg,
    )

    # --- 2) Visualizations ---
    # Diagnostic Confusion Matrix
    if cfg.evaluation.save_confusion_matrix:
        plot_confusion_matrix(
            all_labels=all_labels,
            all_preds=all_preds,
            classes=class_names,
            out_path=paths.get_fig_path(
                f"confusion_matrix_{arch_tag}_{cfg.dataset.resolution}.png"
            ),
            cfg=cfg,
        )

    # Historical Training Curves
    val_acc_list = [m[METRIC_ACCURACY] for m in val_metrics_history]
    plot_training_curves(
        train_losses=train_losses,
        val_accuracies=val_acc_list,
        out_path=paths.get_fig_path(f"training_curves_{arch_tag}_{cfg.dataset.resolution}.png"),
        cfg=cfg,
    )

    # Lazy-loaded prediction grid (samples from loader)
    if cfg.evaluation.save_predictions_grid:
        show_predictions(
            model=model,
            loader=test_loader,
            device=device,
            classes=class_names,
            save_path=paths.get_fig_path(
                f"sample_predictions_{arch_tag}_{cfg.dataset.resolution}.png"
            ),
            cfg=cfg,
        )

    # --- 3) Structured Reporting ---
    # Aggregates everything into a formatted report (xlsx/csv/json)
    final_log = log_path if log_path is not None else (paths.logs / "run.log")

    report = create_structured_report(
        val_metrics=val_metrics_history,
        test_metrics=test_metrics,
        macro_f1=macro_f1,
        train_losses=train_losses,
        best_path=paths.best_model_path,
        log_path=final_log,
        cfg=cfg,
        aug_info=aug_info,
    )
    report.save(paths.final_report_path, fmt=cfg.evaluation.report_format)

    test_acc = test_metrics[METRIC_ACCURACY]
    test_auc = test_metrics.get(METRIC_AUC, 0.0)

    # Log test metrics to experiment tracker
    if tracker is not None:
        tracker.log_test_metrics(test_acc=test_acc, macro_f1=macro_f1)

    logger.info("Final Evaluation Phase Complete.")

    return macro_f1, test_acc, test_auc
