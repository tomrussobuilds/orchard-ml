"""
Progress and Optimization Logging.

Provides formatted logging utilities for training progress, Optuna optimization,
and pipeline completion summaries.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import optuna

from ..paths import LOGGER_NAME
from ..paths.constants import METRIC_ACCURACY, LogStyle
from .env_reporter import Reporter

if TYPE_CHECKING:  # pragma: no cover
    import torch

    from ..config import Config
    from ..paths import RunPaths

logger = logging.getLogger(LOGGER_NAME)


def _format_param_value(value: Any) -> str:
    """
    Format a hyperparameter value for log display.
    """
    if isinstance(value, float):
        return f"{value:.2e}" if value < 0.001 else f"{value:.4f}"
    return str(value)


def _count_trial_states(
    study: "optuna.Study",
) -> tuple[
    list[optuna.trial.FrozenTrial],
    list[optuna.trial.FrozenTrial],
    list[optuna.trial.FrozenTrial],
]:
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
    I = LogStyle.INDENT  # noqa: E741
    A = LogStyle.ARROW
    log.info("%s%s Dataset      : %s", I, A, cfg.dataset.dataset_name)
    model_search = "Enabled" if cfg.optuna.enable_model_search else "Disabled"  # pragma: no mutate
    log.info("%s%s Model Search : %s", I, A, model_search)
    if cfg.optuna.model_pool is not None:
        log.info("%s%s Model Pool   : %s", I, A, ", ".join(cfg.optuna.model_pool))
    log.info("%s%s Search Space : %s", I, A, cfg.optuna.search_space_preset)
    log.info("%s%s Trials       : %s", I, A, cfg.optuna.n_trials)
    log.info("%s%s Epochs/Trial : %s", I, A, cfg.optuna.epochs)
    log.info("%s%s Metric       : %s", I, A, cfg.training.monitor_metric)
    pruning = "Enabled" if cfg.optuna.enable_pruning else "Disabled"  # pragma: no mutate
    log.info("%s%s Pruning      : %s", I, A, pruning)

    if cfg.optuna.enable_early_stopping:
        threshold = cfg.optuna.early_stopping_threshold or "auto"  # pragma: no mutate
        log.info(
            "%s%s Early Stop   : Enabled (threshold=%s, patience=%s)",
            I,
            A,
            threshold,
            cfg.optuna.early_stopping_patience,
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

    log.info(LogStyle.LIGHT)
    log.info("[Trial %d Hyperparameters]", trial_number)

    categories = {
        "Optimization": ["learning_rate", "weight_decay", "momentum", "min_lr"],
        "Loss": ["criterion_type", "focal_gamma", "label_smoothing"],
        "Regularization": ["mixup_alpha", "dropout"],
        "Scheduling": ["scheduler_type", "scheduler_patience", "batch_size"],
        "Augmentation": ["rotation_angle", "jitter_val", "min_scale"],
        "Architecture": ["model_name", "pretrained", "weight_variant"],
    }

    for category_name, keys in categories.items():
        category_params = {k: params[k] for k in keys if k in params}
        if category_params:
            log.info("%s[%s]", LogStyle.INDENT, category_name)
            for key, value in category_params.items():
                log.info(
                    "%s%s %-20s : %s",
                    LogStyle.DOUBLE_INDENT,
                    LogStyle.BULLET,
                    key,
                    _format_param_value(value),
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

    I = LogStyle.INDENT  # noqa: E741
    A = LogStyle.ARROW
    S = LogStyle.SUCCESS
    W = LogStyle.WARNING

    Reporter.log_phase_header(log, "OPTIMIZATION SUMMARY", LogStyle.DOUBLE)
    log.info("%s%s Dataset        : %s", I, A, cfg.dataset.dataset_name)
    log.info("%s%s Search Space   : %s", I, A, cfg.optuna.search_space_preset)
    log.info("%s%s Total Trials   : %d", I, A, len(study.trials))
    log.info("%s%s Completed      : %d", I, S, len(completed))
    log.info("%s%s Pruned         : %d", I, A, len(pruned))

    if failed:
        log.info("%s%s Failed         : %d", I, W, len(failed))

    if completed:
        try:
            log.info(
                "%s%s Best %-9s : %.6f",
                I,
                S,
                cfg.training.monitor_metric.upper(),
                study.best_value,
            )
            log.info("%s%s Best Trial     : %d", I, S, study.best_trial.number)
        except ValueError:
            log.error("%s%s Best trial lookup failed (check study integrity)", I, W)
    else:
        log.warning("%s%s No trials completed", I, W)

    log.info("%s%s Device         : %s", I, A, str(device).upper())
    log.info("%s%s Artifacts      : %s", I, A, Path(paths.root).name)
    log.info(LogStyle.DOUBLE)
    log.info("")


def log_pipeline_summary(
    test_metrics: Mapping[str, float],
    best_model_path: Path,
    run_dir: Path,
    duration: str,
    onnx_path: Path | None = None,
    logger_instance: logging.Logger | None = None,
) -> None:
    """
    Log final pipeline completion summary.

    Called at the end of the pipeline after all phases complete.
    Consolidates key metrics and artifact locations.

    Args:
        test_metrics: Task-specific test metrics mapping.
        best_model_path: Path to best model checkpoint
        run_dir: Root directory for this run
        duration: Human-readable duration string
        onnx_path: Path to ONNX export (if performed)
        logger_instance: Logger instance to use (defaults to module logger)
    """
    log = logger_instance or logger

    I = LogStyle.INDENT  # noqa: E741
    A = LogStyle.ARROW
    S = LogStyle.SUCCESS

    Reporter.log_phase_header(log, "PIPELINE COMPLETE", LogStyle.DOUBLE)
    for key, value in test_metrics.items():
        label = key.replace("_", " ").title()
        if key == METRIC_ACCURACY:
            log.info("%s%s %-15s: %7.2f%%", I, S, label, value * 100)
        else:
            log.info("%s%s %-15s: %8.4f", I, S, label, value)
    log.info("%s%s Best Model     : %s", I, A, Path(best_model_path).name)
    if onnx_path:
        log.info("%s%s ONNX Export    : %s", I, A, Path(onnx_path).name)
    log.info("%s%s Run Directory  : %s", I, A, Path(run_dir).name)
    log.info("%s%s Duration       : %s", I, A, duration)
    log.info(LogStyle.DOUBLE)
