"""
Final Evaluation Pipeline.

Top-level orchestrator that chains inference, visualization, and reporting
into a single ``run_final_evaluation`` call. Coordinates:

1. Test-set inference via ``evaluator.evaluate_model`` (with optional TTA).
2. Artifact generation — confusion matrix, training curves, prediction grid.
3. Structured report (Excel/CSV/JSON) via ``reporting.create_structured_report``.
4. Metric logging to the experiment tracker (MLflow) when enabled.

This module is the last stage of the training lifecycle, invoked by
``ModelTrainer`` after best-weight restoration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch.nn as nn
from torch.utils.data import DataLoader

from ..core import (
    LOGGER_NAME,
    AugmentationConfig,
    DatasetConfig,
    EvaluationConfig,
    LogStyle,
    RunPaths,
    TrainingConfig,
)
from ..core.paths import METRIC_ACCURACY, METRIC_AUC

if TYPE_CHECKING:  # pragma: no cover
    from ..tracking import TrackerProtocol

from .evaluator import evaluate_model
from .plot_context import PlotContext
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
    training: TrainingConfig,
    dataset: DatasetConfig,
    augmentation: AugmentationConfig,
    evaluation: EvaluationConfig,
    arch_name: str,
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
        training: Training sub-config (use_tta, hyperparameters for report).
        dataset: Dataset sub-config (resolution, metadata, normalization).
        augmentation: Augmentation sub-config (TTA transforms).
        evaluation: Evaluation sub-config (plot flags, report format).
        arch_name: Architecture identifier (e.g. ``"resnet_18"``).
        aug_info: Augmentation description string for report.
        log_path: Path to session log file for report embedding.
        tracker: Optional experiment tracker for final metrics.

    Returns:
        tuple[float, float, float]: A 3-tuple of:

            - **macro_f1** -- Macro-averaged F1 score
            - **test_acc** -- Test set accuracy
            - **test_auc** -- Test set AUC (NaN if computation failed)
    """
    # Resolve device from model (already placed on the correct device by the trainer)
    device = next(model.parameters()).device

    # Filesystem-safe architecture tag (e.g. "timm/model" → "timm_model")
    arch_tag = arch_name.replace("/", "_")

    # --- 1) Inference & Metrics ---
    # Performance on the full test set
    all_preds, all_labels, test_metrics, macro_f1 = evaluate_model(
        model,
        test_loader,
        device=device,
        use_tta=training.use_tta,
        is_anatomical=dataset.metadata.is_anatomical,
        is_texture_based=dataset.metadata.is_texture_based,
        aug_cfg=augmentation,
        resolution=dataset.resolution,
    )

    # --- 2) Visualizations ---
    meta = dataset.metadata
    ctx = PlotContext(
        arch_name=arch_name,
        resolution=dataset.resolution,
        fig_dpi=evaluation.fig_dpi,
        plot_style=evaluation.plot_style,
        cmap_confusion=evaluation.cmap_confusion,
        grid_cols=evaluation.grid_cols,
        n_samples=evaluation.n_samples,
        fig_size_predictions=evaluation.fig_size_predictions,
        mean=dataset.mean,
        std=dataset.std,
        use_tta=training.use_tta,
        is_anatomical=meta.is_anatomical if meta else False,
        is_texture_based=meta.is_texture_based if meta else False,
    )

    # Diagnostic Confusion Matrix
    if evaluation.save_confusion_matrix:
        plot_confusion_matrix(
            all_labels=all_labels,
            all_preds=all_preds,
            classes=class_names,
            out_path=paths.get_fig_path(f"confusion_matrix_{arch_tag}_{dataset.resolution}.png"),
            ctx=ctx,
        )

    # Historical Training Curves
    val_acc_list = [m[METRIC_ACCURACY] for m in val_metrics_history]
    plot_training_curves(
        train_losses=train_losses,
        val_accuracies=val_acc_list,
        out_path=paths.get_fig_path(f"training_curves_{arch_tag}_{dataset.resolution}.png"),
        ctx=ctx,
    )

    # Lazy-loaded prediction grid (samples from loader)
    if evaluation.save_predictions_grid:
        show_predictions(
            model=model,
            loader=test_loader,
            device=device,
            classes=class_names,
            save_path=paths.get_fig_path(f"sample_predictions_{arch_tag}_{dataset.resolution}.png"),
            ctx=ctx,
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
        arch_name=arch_name,
        dataset=dataset,
        training=training,
        aug_info=aug_info,
    )
    report.save(paths.final_report_path, fmt=evaluation.report_format)

    test_acc = test_metrics[METRIC_ACCURACY]
    test_auc = test_metrics.get(METRIC_AUC, float("nan"))

    # Log test metrics to experiment tracker
    if tracker is not None:
        tracker.log_test_metrics(test_acc=test_acc, macro_f1=macro_f1)

    logger.info(f"{LogStyle.INDENT}{LogStyle.SUCCESS} Final Evaluation Phase Complete.")

    return macro_f1, test_acc, test_auc
