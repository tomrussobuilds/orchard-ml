"""
Progress and Optimization Logging.

Provides formatted logging utilities for training progress, Optuna optimization,
and pipeline completion summaries.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import optuna

from ..paths import LOGGER_NAME
from .styles import LogStyle

if TYPE_CHECKING:  # pragma: no cover
    import torch

    from ..config import Config
    from ..paths import RunPaths

logger = logging.getLogger(LOGGER_NAME)


def _format_param_value(value: Any) -> str:
    """Format a hyperparameter value for log display."""
    if isinstance(value, float):
        return f"{value:.2e}" if value < 0.001 else f"{value:.4f}"
    return str(value)


def _count_trial_states(study: "optuna.Study") -> tuple[list, list, list]:
    """
    Count trials by state.

    Args:
        study: Optuna study containing trials to count.

    Returns:
        tuple of (completed, pruned, failed) trial lists.
    """
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    return completed, pruned, failed


def log_optimization_header(cfg: "Config", logger_instance: logging.Logger | None = None) -> None:
    """
    Log Optuna optimization configuration details.

    Logs search-specific parameters only (dataset/model already shown in environment).

    Args:
        cfg: Configuration with optuna settings
        logger_instance: Logger instance to use (defaults to module logger)
    """
    log = logger_instance or logger

    # Search configuration (no duplicate header - phase header already shown)
    log.info("")
    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Dataset      : {cfg.dataset.dataset_name}")
    log.info(
        f"{LogStyle.INDENT}{LogStyle.ARROW} Model Search : "
        f"{'Enabled' if cfg.optuna.enable_model_search else 'Disabled'}"
    )
    if cfg.optuna.model_pool is not None:
        log.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} Model Pool   : "
            f"{', '.join(cfg.optuna.model_pool)}"
        )
    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Search Space : {cfg.optuna.search_space_preset}")
    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Trials       : {cfg.optuna.n_trials}")
    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Epochs/Trial : {cfg.optuna.epochs}")
    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Metric       : {cfg.optuna.metric_name}")
    log.info(
        f"{LogStyle.INDENT}{LogStyle.ARROW} Pruning      : "
        f"{'Enabled' if cfg.optuna.enable_pruning else 'Disabled'}"
    )

    if cfg.optuna.enable_early_stopping:
        threshold = cfg.optuna.early_stopping_threshold or "auto"
        log.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} Early Stop   : Enabled "
            f"(threshold={threshold}, patience={cfg.optuna.early_stopping_patience})"
        )

    log.info("")


def log_trial_start(
    trial_number: int, params: dict[str, Any], logger_instance: logging.Logger | None = None
) -> None:
    """
    Log trial start with formatted parameters (grouped by category).

    Args:
        trial_number: Trial index
        params: Sampled hyperparameters
        logger_instance: Logger instance to use (defaults to module logger)
    """
    log = logger_instance or logger

    log.info(f"{LogStyle.LIGHT}")
    log.info(f"Trial {trial_number} Hyperparameters:")

    categories = {
        "Optimization": ["learning_rate", "weight_decay", "momentum", "min_lr"],
        "Regularization": ["mixup_alpha", "label_smoothing", "dropout"],
        "Scheduling": ["scheduler_type", "scheduler_patience", "batch_size"],
        "Augmentation": ["rotation_angle", "jitter_val", "min_scale"],
        "Architecture": ["model_name", "pretrained", "weight_variant"],
    }

    for category_name, keys in categories.items():
        category_params = {k: params[k] for k in keys if k in params}
        if category_params:
            log.info(f"{LogStyle.INDENT}[{category_name}]")
            for key, value in category_params.items():
                log.info(
                    f"{LogStyle.DOUBLE_INDENT}{LogStyle.BULLET} "
                    f"{key:<20} : {_format_param_value(value)}"
                )

    log.info(LogStyle.LIGHT)


def log_optimization_summary(
    study: "optuna.Study",
    cfg: "Config",
    device: "torch.device",
    paths: "RunPaths",
    logger_instance: logging.Logger | None = None,
) -> None:
    """
    Log optimization study completion summary.

    Args:
        study: Completed Optuna study
        cfg: Configuration object
        device: PyTorch device used
        paths: Run paths for artifacts
        logger_instance: Logger instance to use (defaults to module logger)
    """
    log = logger_instance or logger
    completed, pruned, failed = _count_trial_states(study)

    LogStyle.log_phase_header(log, "OPTIMIZATION SUMMARY", LogStyle.DOUBLE)
    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Dataset        : {cfg.dataset.dataset_name}")
    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Search Space   : {cfg.optuna.search_space_preset}")
    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Total Trials   : {len(study.trials)}")
    log.info(f"{LogStyle.INDENT}{LogStyle.SUCCESS} Completed      : {len(completed)}")
    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Pruned         : {len(pruned)}")

    if failed:
        log.info(f"{LogStyle.INDENT}{LogStyle.WARNING} Failed         : {len(failed)}")

    if completed:
        try:
            log.info(
                f"{LogStyle.INDENT}{LogStyle.SUCCESS} "
                f"Best {cfg.optuna.metric_name.upper():<9} : {study.best_value:.6f}"
            )
            log.info(
                f"{LogStyle.INDENT}{LogStyle.SUCCESS} Best Trial     : {study.best_trial.number}"
            )
        except ValueError:  # pragma: no cover
            log.warning(
                f"{LogStyle.INDENT}{LogStyle.WARNING} "
                "Best trial lookup failed (check study integrity)"
            )
    else:
        log.warning(f"{LogStyle.INDENT}{LogStyle.WARNING} No trials completed")

    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Device         : {str(device).upper()}")
    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Artifacts      : {Path(paths.root).name}")
    log.info(f"{LogStyle.DOUBLE}")
    log.info("")


def log_pipeline_summary(
    test_acc: float,
    macro_f1: float,
    best_model_path: Any,
    run_dir: Any,
    duration: str,
    test_auc: float | None = None,
    onnx_path: Any = None,
    logger_instance: logging.Logger | None = None,
) -> None:
    """
    Log final pipeline completion summary.

    Called at the end of the pipeline after all phases complete.
    Consolidates key metrics and artifact locations.

    Args:
        test_acc: Final test accuracy
        macro_f1: Final macro F1 score
        best_model_path: Path to best model checkpoint
        run_dir: Root directory for this run
        duration: Human-readable duration string
        test_auc: Final test AUC (if available)
        onnx_path: Path to ONNX export (if performed)
        logger_instance: Logger instance to use (defaults to module logger)
    """
    log = logger_instance or logger

    LogStyle.log_phase_header(log, "PIPELINE COMPLETE", LogStyle.DOUBLE)
    log.info(f"{LogStyle.INDENT}{LogStyle.SUCCESS} Test Accuracy  : {test_acc:>8.2%}")
    log.info(f"{LogStyle.INDENT}{LogStyle.SUCCESS} Macro F1       : {macro_f1:>8.4f}")
    if test_auc is not None:
        log.info(f"{LogStyle.INDENT}{LogStyle.SUCCESS} Test AUC       : {test_auc:>8.4f}")
    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Best Model     : {Path(best_model_path).name}")
    if onnx_path:
        log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} ONNX Export    : {Path(onnx_path).name}")
    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Run Directory  : {Path(run_dir).name}")
    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Duration       : {duration}")
    log.info(f"{LogStyle.DOUBLE}")
