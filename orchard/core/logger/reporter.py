"""
Telemetry & Environment Reporting Engine.

Provides formatted logging utilities for RootOrchestrator and OptunaOrchestrator.
Centralizes all heavy logging logic for consistent, readable output.

The reporter handles:
    - Hardware capability visualization
    - Dataset metadata resolution reporting
    - Execution strategy and hyperparameter summaries
    - Optimization trial formatting
    - Study completion summaries
"""

# Standard Imports
import logging
from typing import TYPE_CHECKING, Any, Dict

# Third-Party Imports
import optuna
import torch
from pydantic import BaseModel, ConfigDict

# Internal Imports
from ..environment import determine_tta_mode, get_cuda_name, get_vram_info
from ..paths import LOGGER_NAME, RunPaths

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger(LOGGER_NAME)


# LOG STYLE CONSTANTS
class LogStyle:
    """Unified logging style constants for consistent visual hierarchy."""

    # Level 1: Session headers (80 chars)
    HEAVY = "━" * 80

    # Level 2: Major sections (80 chars)
    DOUBLE = "═" * 80

    # Level 3: Subsections / Separators (80 chars)
    LIGHT = "─" * 80

    # Symbols
    ARROW = "»"  # For key-value pairs
    BULLET = "•"  # For list items
    WARNING = "⚠"  # For warnings
    SUCCESS = "✓"  # For success messages

    # Indentation
    INDENT = "  "
    DOUBLE_INDENT = "    "


# EXPERIMENT INITIALIZATION LOGGING
class Reporter(BaseModel):
    """
    Centralized logging and reporting utility for experiment lifecycle events.

    Transforms complex configuration states and hardware objects into
    human-readable logs. Called by Orchestrator during initialization.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def log_initial_status(
        self,
        logger_instance: logging.Logger,
        cfg: "Config",
        paths: "RunPaths",
        device: "torch.device",
        applied_threads: int,
        num_workers: int,
    ) -> None:
        """
        Logs verified baseline environment configuration upon initialization.

        Args:
            logger_instance: Active experiment logger
            cfg: Validated global configuration manifest
            paths: Dynamic path orchestrator for current session
            device: Resolved PyTorch compute device
            applied_threads: Number of intra-op threads assigned
            num_workers: Number of DataLoader workers
        """
        # Newline + Header Block
        logger_instance.info("")
        logger_instance.info(LogStyle.HEAVY)
        logger_instance.info(f"{'ENVIRONMENT INITIALIZATION':^80}")
        logger_instance.info(LogStyle.HEAVY)

        # Hardware Section
        self._log_hardware_section(logger_instance, cfg, device, applied_threads, num_workers)
        logger_instance.info("")

        # Dataset Section
        self._log_dataset_section(logger_instance, cfg)
        logger_instance.info("")

        # Strategy Section
        self._log_strategy_section(logger_instance, cfg, device)
        logger_instance.info("")

        # Hyperparameters Section
        logger_instance.info("[HYPERPARAMETERS]")
        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Epochs':<18}: {cfg.training.epochs}"
        )
        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Batch Size':<18}: {cfg.training.batch_size}"
        )
        lr = cfg.training.learning_rate
        lr_str = f"{lr:.2e}" if isinstance(lr, (float, int)) else str(lr)
        logger_instance.info(f"{LogStyle.INDENT}{LogStyle.ARROW} {'Initial LR':<18}: {lr_str}")
        logger_instance.info("")

        # Filesystem Section
        logger_instance.info("[FILESYSTEM]")
        logger_instance.info(f"{LogStyle.INDENT}{LogStyle.ARROW} {'Run Root':<18}: {paths.root}")

        # Closing separator
        logger_instance.info(LogStyle.HEAVY)
        logger_instance.info("")

    def _log_hardware_section(
        self,
        logger_instance: logging.Logger,
        cfg: "Config",
        device: "torch.device",
        applied_threads: int,
        num_workers: int,
    ) -> None:
        """Logs hardware-specific configuration and GPU metadata."""
        requested_device = cfg.hardware.device.lower()
        active_type = device.type

        logger_instance.info("[HARDWARE]")
        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Active Device':<18}: {str(device).upper()}"
        )

        if requested_device != "cpu" and active_type == "cpu":
            logger_instance.warning(
                f"{LogStyle.INDENT}{LogStyle.WARNING} FALLBACK: Requested '{requested_device}' unavailable, using CPU"
            )

        if active_type == "cuda":
            logger_instance.info(
                f"{LogStyle.INDENT}{LogStyle.ARROW} {'GPU Model':<18}: {get_cuda_name()}"
            )
            logger_instance.info(
                f"{LogStyle.INDENT}{LogStyle.ARROW} {'VRAM Available':<18}: {get_vram_info(device.index or 0)}"
            )

        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'DataLoader':<18}: {num_workers} workers"
        )
        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Compute Threads':<18}: {applied_threads} threads"
        )

    def _log_dataset_section(self, logger_instance: logging.Logger, cfg: "Config") -> None:
        """Logs dataset metadata and characteristics."""
        ds = cfg.dataset
        meta = ds.metadata

        logger_instance.info("[DATASET]")
        logger_instance.info(f"{LogStyle.INDENT}{LogStyle.ARROW} {'Name':<18}: {meta.display_name}")
        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Classes':<18}: {meta.num_classes} categories"
        )
        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Resolution':<18}: {ds.img_size}px (Native: {meta.resolution_str})"
        )
        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Channels':<18}: {meta.in_channels}"
        )
        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Anatomical':<18}: {meta.is_anatomical}"
        )
        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Texture-based':<18}: {meta.is_texture_based}"
        )

    def _log_strategy_section(
        self, logger_instance: logging.Logger, cfg: "Config", device: "torch.device"
    ) -> None:
        """Logs high-level training strategies and models."""
        train = cfg.training
        sys = cfg.hardware
        tta_status = determine_tta_mode(train.use_tta, device.type)

        repro_mode = "Strict" if sys.use_deterministic_algorithms else "Standard"

        logger_instance.info("[STRATEGY]")
        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Architecture':<18}: {cfg.model.name}"
        )
        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Weights':<18}: {'Pretrained' if cfg.model.pretrained else 'Random'}"
        )

        # Add weight variant if present (for ViT)
        if cfg.model.weight_variant:
            logger_instance.info(
                f"{LogStyle.INDENT}{LogStyle.ARROW} {'Weight Variant':<18}: {cfg.model.weight_variant}"
            )

        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Precision':<18}: {'AMP (Mixed)' if train.use_amp else 'FP32'}"
        )
        logger_instance.info(f"{LogStyle.INDENT}{LogStyle.ARROW} {'TTA Mode':<18}: {tta_status}")
        logger_instance.info(f"{LogStyle.INDENT}{LogStyle.ARROW} {'Repro. Mode':<18}: {repro_mode}")
        logger_instance.info(f"{LogStyle.INDENT}{LogStyle.ARROW} {'Global Seed':<18}: {train.seed}")


# OPTIMIZATION LOGGING
def log_optimization_header(cfg: "Config", logger_instance: logging.Logger = None) -> None:
    """
    Log Optuna optimization session header.

    Args:
        cfg: Configuration with optuna settings
        logger_instance: Logger instance to use (defaults to module logger)
    """
    log = logger_instance or logger

    log.info("")
    log.info(LogStyle.DOUBLE)
    log.info(f"{'OPTUNA HYPERPARAMETER OPTIMIZATION':^80}")
    log.info(LogStyle.DOUBLE)
    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Dataset      : {cfg.dataset.dataset_name}")
    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Model        : {cfg.model.name}")

    if cfg.model.weight_variant:
        log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Weight Var.  : {cfg.model.weight_variant}")

    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Search Space : {cfg.optuna.search_space_preset}")
    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Trials       : {cfg.optuna.n_trials}")
    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Epochs/Trial : {cfg.optuna.epochs}")
    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Metric       : {cfg.optuna.metric_name}")
    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Device       : {cfg.hardware.device}")
    log.info(
        f"{LogStyle.INDENT}{LogStyle.ARROW} Pruning      : {'Enabled' if cfg.optuna.enable_pruning else 'Disabled'}"
    )

    if cfg.optuna.enable_early_stopping:
        threshold = cfg.optuna.early_stopping_threshold or "auto"
        log.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} Early Stop   : Enabled (threshold={threshold}, patience={cfg.optuna.early_stopping_patience})"
        )

    log.info(LogStyle.DOUBLE)
    log.info("")


def log_trial_start(
    trial_number: int, params: Dict[str, Any], logger_instance: logging.Logger = None
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
    study: "optuna.Study", metric_name: str, logger_instance: logging.getLogger = None
) -> None:
    """
    Log optimization study completion summary.

    Args:
        study: Completed Optuna study
        metric_name: Name of optimization metric
    """
    log = logger or logger_instance

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

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
                f"{LogStyle.INDENT}{LogStyle.SUCCESS} Best {metric_name.upper()} : {study.best_value:.6f}"
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
    trial_number: int, params: Dict[str, Any], logger_instance: logging.getLogger = None
) -> None:
    """
    Compact parameter logging for best trial summary.

    Args:
        trial_number: Trial index
        params: Trial hyperparameters
    """
    log = logger or logger_instance

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


def log_best_config_export(config_path: Any, logger_instance: logging.getLogger = None) -> None:
    """
    Log best configuration export information.

    Args:
        config_path: Path to exported YAML config
    """
    log = logger or logger_instance

    log.info(f"{LogStyle.DOUBLE}")
    log.info(f"{'BEST CONFIGURATION EXPORTED':^80}")
    log.info(LogStyle.DOUBLE)
    log.info(f"{LogStyle.INDENT}{LogStyle.SUCCESS} Configuration saved to: {config_path}")
    log.info("")
    log.info(f"{LogStyle.INDENT}To train with optimized hyperparameters:")
    log.info(f"{LogStyle.DOUBLE_INDENT}python main.py --config {config_path}")
    log.info("")
    log.info(f"{LogStyle.INDENT}To visualize optimization results:")
    log.info(
        f"{LogStyle.DOUBLE_INDENT}firefox {config_path.parent.parent}/figures/param_importances.html"
    )
    log.info(f"{LogStyle.DOUBLE}")
    log.info("")


def log_training_summary(
    cfg: "Config",
    test_acc: float,
    macro_f1: float,
    device: "torch.device",
    paths: "RunPaths",
    logger_instance: logging.Logger = None,
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
    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Architecture : {cfg.model.name}")

    if cfg.model.weight_variant:
        log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Weight Var.  : {cfg.model.weight_variant}")

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
    logger_instance: logging.Logger = None,
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

    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    pruned = [t for t in study.trials if t.state.name == "PRUNED"]
    failed = [t for t in study.trials if t.state.name == "FAIL"]

    log.info(f"{LogStyle.DOUBLE}")
    log.info(f"{'OPTIMIZATION EXECUTION SUMMARY':^80}")
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
                f"{LogStyle.INDENT}{LogStyle.SUCCESS} Best {cfg.optuna.metric_name.upper():<9} : {study.best_value:.6f}"
            )
            log.info(
                f"{LogStyle.INDENT}{LogStyle.SUCCESS} Best Trial     : {study.best_trial.number}"
            )
        except ValueError:
            log.warning(
                f"{LogStyle.INDENT}{LogStyle.WARNING} Best trial lookup failed (check study integrity)"
            )
    else:
        log.warning(f"{LogStyle.INDENT}{LogStyle.WARNING} No trials completed")

    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Device         : {str(device).upper()}")
    log.info(f"{LogStyle.INDENT}{LogStyle.ARROW} Artifacts      : {paths.root}")
    log.info(f"{LogStyle.DOUBLE}")
    log.info("")
