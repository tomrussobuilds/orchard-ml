"""
Environment Reporter.

Provides formatted logging for experiment initialization and environment
configuration. Transforms complex configuration states and hardware objects
into structured, human-readable log output.

The Reporter is invoked by RootOrchestrator during initialization to produce
a comprehensive baseline status report covering hardware, dataset, strategy,
hyperparameters, and filesystem configuration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

import torch
from pydantic import BaseModel, ConfigDict

from ..environment import determine_tta_mode, get_cuda_name, get_vram_info
from .styles import LogStyle

if TYPE_CHECKING:  # pragma: no cover
    from ..config import Config
    from ..paths import RunPaths


class ReporterProtocol(Protocol):
    """Protocol for environment reporting, allowing mocking in tests."""

    def log_initial_status(
        self,
        logger_instance: logging.Logger,
        cfg: "Config",
        paths: "RunPaths",
        device: torch.device,
        applied_threads: int,
        num_workers: int,
    ) -> None:
        """
        Logs the initial status of the environment.

        Args:
            logger_instance: The logger instance used to log the status.
            cfg: The configuration object containing environment settings.
            paths: The paths object with directories for the run.
            device: The device (e.g., CPU or GPU) to be used for processing.
            applied_threads: The number of threads allocated for processing.
            num_workers: The number of worker processes to use.
        """
        ...  # pragma: no cover


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
        # Header Block
        LogStyle.log_phase_header(logger_instance, "ENVIRONMENT INITIALIZATION")

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

        # Tracking Section (only if configured)
        self._log_tracking_section(logger_instance, cfg)

        # Optimization Section (only if configured)
        self._log_optimization_section(logger_instance, cfg)

        # Export Section (only if configured)
        self._log_export_section(logger_instance, cfg)

        # Filesystem Section
        logger_instance.info("[FILESYSTEM]")
        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Run Root':<18}: {paths.root.name}"
        )
        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Manifest':<18}: config.yaml, requirements.txt"
        )

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
                f"{LogStyle.INDENT}{LogStyle.WARNING} "
                f"FALLBACK: Requested '{requested_device}' unavailable, using CPU"
            )

        if active_type == "cuda":
            logger_instance.info(
                f"{LogStyle.INDENT}{LogStyle.ARROW} {'GPU Model':<18}: {get_cuda_name()}"
            )
            logger_instance.info(
                f"{LogStyle.INDENT}{LogStyle.ARROW} "
                f"{'VRAM Available':<18}: {get_vram_info(device.index or 0)}"
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
            f"{LogStyle.INDENT}{LogStyle.ARROW} "
            f"{'Resolution':<18}: {ds.img_size}px (Native: {meta.resolution_str})"
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
        tta_status = determine_tta_mode(train.use_tta, device.type, cfg.augmentation.tta_mode)

        repro_mode = "Strict" if sys.use_deterministic_algorithms else "Standard"

        logger_instance.info("[STRATEGY]")
        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Architecture':<18}: {cfg.architecture.name}"
        )
        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} "
            f"{'Weights':<18}: {'Pretrained' if cfg.architecture.pretrained else 'Random'}"
        )

        # Add weight variant if present (for ViT)
        if cfg.architecture.weight_variant:
            logger_instance.info(
                f"{LogStyle.INDENT}{LogStyle.ARROW} "
                f"{'Weight Variant':<18}: {cfg.architecture.weight_variant}"
            )

        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} "
            f"{'Precision':<18}: {'AMP (Mixed)' if train.use_amp else 'FP32'}"
        )
        logger_instance.info(f"{LogStyle.INDENT}{LogStyle.ARROW} {'TTA Mode':<18}: {tta_status}")
        logger_instance.info(f"{LogStyle.INDENT}{LogStyle.ARROW} {'Repro. Mode':<18}: {repro_mode}")
        logger_instance.info(f"{LogStyle.INDENT}{LogStyle.ARROW} {'Global Seed':<18}: {train.seed}")

    def _log_tracking_section(self, logger_instance: logging.Logger, cfg: "Config") -> None:
        """Logs tracking configuration if enabled."""
        tracking_cfg = getattr(cfg, "tracking", None)
        if tracking_cfg is None:
            return

        logger_instance.info("[TRACKING]")
        status = "Active" if tracking_cfg.enabled else "Disabled"
        logger_instance.info(f"{LogStyle.INDENT}{LogStyle.ARROW} {'Status':<18}: {status}")
        if tracking_cfg.enabled:
            logger_instance.info(
                f"{LogStyle.INDENT}{LogStyle.ARROW} "
                f"{'Experiment':<18}: {tracking_cfg.experiment_name}"
            )
        logger_instance.info("")

    def _log_optimization_section(self, logger_instance: logging.Logger, cfg: "Config") -> None:
        """Logs optimization configuration if enabled."""
        optuna_cfg = getattr(cfg, "optuna", None)
        if optuna_cfg is None:
            return

        logger_instance.info("[OPTIMIZATION]")
        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Trials':<18}: {optuna_cfg.n_trials}"
        )
        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Epochs/Trial':<18}: {optuna_cfg.epochs}"
        )
        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} "
            f"{'Metric':<18}: {optuna_cfg.metric_name} ({optuna_cfg.direction})"
        )
        logger_instance.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Sampler':<18}: {optuna_cfg.sampler_type.upper()}"
        )
        if optuna_cfg.enable_pruning:
            logger_instance.info(
                f"{LogStyle.INDENT}{LogStyle.ARROW} {'Pruner':<18}: {optuna_cfg.pruner_type}"
            )
        logger_instance.info("")

    def _log_export_section(self, logger_instance: logging.Logger, cfg: "Config") -> None:
        """Logs export configuration if enabled."""
        export_cfg = getattr(cfg, "export", None)
        if export_cfg is None:
            return

        I, A = LogStyle.INDENT, LogStyle.ARROW  # noqa: E741
        logger_instance.info("[EXPORT]")
        logger_instance.info(f"{I}{A} {'Format':<18}: {export_cfg.format.upper()}")
        logger_instance.info(f"{I}{A} {'Opset Version':<18}: {export_cfg.opset_version}")
        logger_instance.info(f"{I}{A} {'Validate':<18}: {export_cfg.validate_export}")
        if export_cfg.quantize:
            logger_instance.info(
                f"{I}{A} {'Quantize':<18}: INT8 ({export_cfg.quantization_backend})"
            )
        logger_instance.info("")
