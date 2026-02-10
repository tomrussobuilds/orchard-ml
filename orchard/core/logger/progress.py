"""
Progress and Optimization Logging.

Provides formatted logging utilities for training progress, Optuna optimization,
and pipeline completion summaries.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict

import optuna

from ..paths import LOGGER_NAME
from .styles import LogStyle

if TYPE_CHECKING:  # pragma: no cover
    import torch

    from ..config import Config
    from ..paths import RunPaths

logger = logging.getLogger(LOGGER_NAME)


def _count_trial_states(study: "optuna.Study") -> tuple[list, list, list]:
    """
    Count trials by state.

    Args:
        study: Optuna study containing trials to count.

    Returns:
        Tuple of (completed, pruned, failed) trial lists.
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
    trial_number: int, params: Dict[str, Any], logger_instance: logging.Logger | None = None
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

    categories = [
        ("Optimization", ["learning_rate", "weight_decay", "momentum", "min_lr"]),
        ("Regularization", ["mixup_alpha", "label_smoothing", "dropout"]),
        ("Scheduling", ["cosine_fraction", "scheduler_patience", "batch_size"]),
        ("Augmentation", ["rotation_angle", "jitter_val", "min_scale"]),
        ("Architecture", ["model_name", "weight_variant"]),
    ]

    for category_name, param_list in categories:
        category_params = {k: v for k, v in params.items() if k in param_list}
        if category_params:
            log.info(f"{LogStyle.INDENT}[{category_name}]")
            for key, value in category_params.items():
                if isinstance(value, float):
                    if value < 0.001:
                        log.info(
                            f"{LogStyle.DOUBLE_INDENT}{LogStyle.BULLET} {key:<20} : {value:.2e}"
                        )
                    else:
                        log.info(
                            f"{LogStyle.DOUBLE_INDENT}{LogStyle.BULLET} {key:<20} : {value:.4f}"
                        )
                else:
                    log.info(f"{LogStyle.DOUBLE_INDENT}{LogStyle.BULLET} {key:<20} : {value}")

    log.info(LogStyle.LIGHT)


def log_study_summary(
    study: "optuna.Study", metric_name: str, logger_instance: logging.Logger | None = None
) -> None:
    """
    Log optimization study completion summary.

    Args:
        study: Completed Optuna study
        metric_name: Name of optimization metric
    """
    log = logger_instance or logger
    completed, pruned, failed = _count_trial_states(study)

    log.info("")
    log.info(LogStyle.DOUBLE)
    log.info(f"{'OPTIMIZATION COMPLETE':^80}")
    log.info(LogStyle.DOUBLE)
    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Total Trials   : {len(study.trials)}")
    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Completed      : {len(completed)}")
    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Pruned         : {len(pruned)}")
    if failed:
        log.info(f"{LogStyle.INDENT}{LogStyle.WARNING} Failed         : {len(failed)}")
    log.info("")

    if completed:
        try:
            log.info(
                f"{LogStyle.INDENT}{LogStyle.SUCCESS} "
                f"Best {metric_name.upper()} : {study.best_value:.6f}"
            )
            log.info(
                f"{LogStyle.INDENT}{LogStyle.SUCCESS} Best Trial     : {study.best_trial.number}"
            )
            log.info(LogStyle.DOUBLE)
            log.info("")

            log.info("Best Hyperparameters:")
            log_trial_params_compact(study.best_trial.number, study.best_params)
        except ValueError:
            log.warning(f"{LogStyle.INDENT}{LogStyle.WARNING} Best trial lookup failed")
            log.info(LogStyle.DOUBLE)
    else:
        log.warning(f"{LogStyle.INDENT}{LogStyle.WARNING} No trials completed successfully")
        log.info(LogStyle.DOUBLE)

    log.info("")


def log_trial_params_compact(
    trial_number: int, params: Dict[str, Any], logger_instance: logging.Logger | None = None
) -> None:
    """
    Compact parameter logging for best trial summary.

    Args:
        trial_number: Trial index
        params: Trial hyperparameters
    """
    log = logger_instance or logger

    categories = [
        ("Optimization", ["learning_rate", "weight_decay", "momentum", "min_lr"]),
        ("Regularization", ["mixup_alpha", "label_smoothing", "dropout"]),
        ("Scheduling", ["cosine_fraction", "scheduler_patience", "batch_size"]),
        ("Augmentation", ["rotation_angle", "jitter_val", "min_scale"]),
        ("Architecture", ["model_name", "weight_variant"]),
    ]

    for category_name, param_list in categories:
        category_params = {k: v for k, v in params.items() if k in param_list}
        if category_params:
            log.info(f"{LogStyle.INDENT}[{category_name}]")
            for key, value in category_params.items():
                if isinstance(value, float):
                    if value < 0.001:  # pragma: no cover
                        log.info(
                            f"{LogStyle.DOUBLE_INDENT}{LogStyle.BULLET} {key:<20} : {value:.2e}"
                        )
                    else:
                        log.info(
                            f"{LogStyle.DOUBLE_INDENT}{LogStyle.BULLET} {key:<20} : {value:.4f}"
                        )
                else:
                    log.info(f"{LogStyle.DOUBLE_INDENT}{LogStyle.BULLET} {key:<20} : {value}")


def log_best_config_export(config_path: Any, logger_instance: logging.Logger | None = None) -> None:
    """
    Log best configuration export information.

    Args:
        config_path: Path to exported YAML config
    """
    log = logger_instance or logger

    log.info(f"{LogStyle.DOUBLE}")
    log.info(f"{'BEST CONFIGURATION EXPORTED':^80}")
    log.info(LogStyle.DOUBLE)
    log.info(f"{LogStyle.INDENT}{LogStyle.SUCCESS} Configuration saved to: {config_path}")
    log.info("")
    log.info(f"{LogStyle.INDENT}To visualize optimization results:")
    log.info(
        f"{LogStyle.DOUBLE_INDENT}firefox "
        f"{config_path.parent.parent}/figures/param_importances.html"
    )
    log.info(f"{LogStyle.DOUBLE}")
    log.info("")


def log_training_summary(
    cfg: "Config",
    test_acc: float,
    macro_f1: float,
    device: "torch.device",
    paths: "RunPaths",
    logger_instance: logging.Logger | None = None,
) -> None:
    """
    Log training pipeline completion summary.

    Args:
        cfg: Configuration object
        test_acc: Final test accuracy
        macro_f1: Final macro F1 score
        device: PyTorch device used
        paths: Run paths for artifacts
        logger_instance: Logger instance to use (defaults to module logger)
    """
    log = logger_instance or logger

    log.info(f"{LogStyle.DOUBLE}")
    log.info(f"{'PIPELINE EXECUTION SUMMARY':^80}")
    log.info(LogStyle.DOUBLE)
    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Dataset      : {cfg.dataset.dataset_name}")
    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Architecture : {cfg.architecture.name}")

    if cfg.architecture.weight_variant:
        log.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} Weight Var.  : {cfg.architecture.weight_variant}"
        )

    log.info(f"{LogStyle.INDENT}{LogStyle.SUCCESS} Test Accuracy: {test_acc:>8.2%}")
    log.info(f"{LogStyle.INDENT}{LogStyle.SUCCESS} Macro F1     : {macro_f1:>8.4f}")
    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Device       : {str(device).upper()}")
    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Artifacts    : {paths.root}")
    log.info(f"{LogStyle.DOUBLE}")
    log.info("")


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

    log.info(f"{LogStyle.DOUBLE}")
    log.info(f"{'OPTIMIZATION SUMMARY':^80}")
    log.info(LogStyle.DOUBLE)
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
    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Artifacts      : {paths.root}")
    log.info(f"{LogStyle.DOUBLE}")
    log.info("")


def log_pipeline_summary(
    test_acc: float,
    macro_f1: float,
    best_model_path: Any,
    run_dir: Any,
    duration: str,
    onnx_path: Any = None,
    logger_instance: logging.Logger | None = None,
) -> None:
    """
    Log final pipeline completion summary.

    Called at the end of forge.py after all phases complete.
    Consolidates key metrics and artifact locations.

    Args:
        test_acc: Final test accuracy
        macro_f1: Final macro F1 score
        best_model_path: Path to best model checkpoint
        run_dir: Root directory for this run
        duration: Human-readable duration string
        onnx_path: Path to ONNX export (if performed)
        logger_instance: Logger instance to use (defaults to module logger)
    """
    log = logger_instance or logger

    log.info("")
    log.info(f"{LogStyle.DOUBLE}")
    log.info(f"{'PIPELINE COMPLETE':^80}")
    log.info(LogStyle.DOUBLE)
    log.info(f"{LogStyle.INDENT}{LogStyle.SUCCESS} Test Accuracy  : {test_acc:>8.2%}")
    log.info(f"{LogStyle.INDENT}{LogStyle.SUCCESS} Macro F1       : {macro_f1:>8.4f}")
    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Best Model     : {best_model_path}")
    if onnx_path:
        log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} ONNX Export    : {onnx_path}")
    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Run Directory  : {run_dir}")
    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Duration       : {duration}")
    log.info(f"{LogStyle.DOUBLE}")
