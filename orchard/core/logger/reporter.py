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

from ..environment import determine_tta_mode, get_accelerator_name, get_vram_info
from ..paths.constants import LogStyle

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

    @staticmethod
    def log_phase_header(
        log: logging.Logger,
        title: str,
        style: str | None = None,
    ) -> None:
        """
        Log a centered phase header with separator lines.

        Args:
            log: Logger instance to write to.
            title: Header text (will be uppercased and centered).
            style: Separator string (defaults to ``LogStyle.HEAVY``).
        """
        sep = style if style is not None else LogStyle.HEAVY
        log.info("")  # pragma: no mutate
        log.info(sep)  # pragma: no mutate
        log.info(title.center(LogStyle.HEADER_WIDTH))  # pragma: no mutate
        log.info(sep)  # pragma: no mutate

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
        Reporter.log_phase_header(
            logger_instance, "ENVIRONMENT INITIALIZATION"
        )  # pragma: no mutate

        I = LogStyle.INDENT  # noqa: E741
        A = LogStyle.ARROW

        # Experiment identifier
        logger_instance.info(  # pragma: no mutate
            "%s%s %-18s: %s", I, A, "Experiment", cfg.run_slug
        )
        logger_instance.info("")  # pragma: no mutate

        # Hardware Section
        self._log_hardware_section(logger_instance, cfg, device, applied_threads, num_workers)
        logger_instance.info("")  # pragma: no mutate

        # Dataset Section
        self._log_dataset_section(logger_instance, cfg)
        logger_instance.info("")  # pragma: no mutate

        # Strategy Section
        self._log_strategy_section(logger_instance, cfg, device)
        logger_instance.info("")  # pragma: no mutate

        # Hyperparameters Section
        logger_instance.info("[HYPERPARAMETERS]")  # pragma: no mutate
        logger_instance.info(  # pragma: no mutate
            "%s%s %-18s: %s", I, A, "Epochs", cfg.training.epochs
        )
        logger_instance.info(  # pragma: no mutate
            "%s%s %-18s: %s", I, A, "Batch Size", cfg.training.batch_size
        )
        lr = cfg.training.learning_rate
        lr_str = f"{lr:.2e}" if isinstance(lr, (float, int)) else str(lr)
        logger_instance.info("%s%s %-18s: %s", I, A, "Initial LR", lr_str)  # pragma: no mutate
        logger_instance.info("")  # pragma: no mutate

        # Tracking Section (only if configured)
        self._log_tracking_section(logger_instance, cfg)

        # Optimization Section (only if configured)
        self._log_optimization_section(logger_instance, cfg)

        # Export Section (only if configured)
        self._log_export_section(logger_instance, cfg)

        # Filesystem Section
        logger_instance.info("[FILESYSTEM]")  # pragma: no mutate
        logger_instance.info(  # pragma: no mutate
            "%s%s %-18s: %s", I, A, "Run Root", paths.root.name
        )
        logger_instance.info(  # pragma: no mutate
            "%s%s %-18s: config.yaml, requirements.txt", I, A, "Manifest"
        )

        # Closing separator
        logger_instance.info(LogStyle.HEAVY)  # pragma: no mutate
        logger_instance.info("")  # pragma: no mutate

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

        I = LogStyle.INDENT  # noqa: E741
        A = LogStyle.ARROW

        logger_instance.info("[HARDWARE]")  # pragma: no mutate
        logger_instance.info(  # pragma: no mutate
            "%s%s %-18s: %s", I, A, "Active Device", str(device).upper()
        )

        if requested_device != "cpu" and active_type == "cpu":
            logger_instance.warning(
                "%s%s FALLBACK: Requested '%s' unavailable, using CPU",
                I,
                LogStyle.WARNING,
                requested_device,
            )

        if active_type in ("cuda", "mps"):
            logger_instance.info(  # pragma: no mutate
                "%s%s %-18s: %s", I, A, "Accelerator", get_accelerator_name()
            )
            if active_type == "cuda":
                logger_instance.info(  # pragma: no mutate
                    "%s%s %-18s: %s", I, A, "VRAM Available", get_vram_info(device.index or 0)
                )

        logger_instance.info(  # pragma: no mutate
            "%s%s %-18s: %d workers", I, A, "DataLoader", num_workers
        )
        logger_instance.info(  # pragma: no mutate
            "%s%s %-18s: %d threads", I, A, "Compute Threads", applied_threads
        )

    def _log_dataset_section(self, logger_instance: logging.Logger, cfg: "Config") -> None:
        """Logs dataset metadata and characteristics."""
        ds = cfg.dataset
        meta = ds.metadata

        I = LogStyle.INDENT  # noqa: E741
        A = LogStyle.ARROW

        logger_instance.info("[DATASET]")  # pragma: no mutate
        logger_instance.info("%s%s %-18s: %s", I, A, "Name", meta.display_name)  # pragma: no mutate
        logger_instance.info(  # pragma: no mutate
            "%s%s %-18s: %d categories", I, A, "Classes", meta.num_classes
        )
        logger_instance.info(  # pragma: no mutate
            "%s%s %-18s: %dpx (Native: %s)", I, A, "Resolution", ds.img_size, meta.resolution_str
        )
        logger_instance.info(
            "%s%s %-18s: %s", I, A, "Channels", meta.in_channels
        )  # pragma: no mutate
        logger_instance.info(  # pragma: no mutate
            "%s%s %-18s: %s", I, A, "Anatomical", meta.is_anatomical
        )
        logger_instance.info(  # pragma: no mutate
            "%s%s %-18s: %s", I, A, "Texture-based", meta.is_texture_based
        )

    def _log_strategy_section(
        self, logger_instance: logging.Logger, cfg: "Config", device: "torch.device"
    ) -> None:
        """Logs high-level training strategies and models."""
        train = cfg.training
        sys = cfg.hardware
        tta_status = determine_tta_mode(train.use_tta, device.type, cfg.augmentation.tta_mode)

        repro_mode = "Strict" if sys.use_deterministic_algorithms else "Standard"

        I = LogStyle.INDENT  # noqa: E741
        A = LogStyle.ARROW

        logger_instance.info("[STRATEGY]")  # pragma: no mutate
        logger_instance.info(  # pragma: no mutate
            "%s%s %-18s: %s", I, A, "Architecture", cfg.architecture.name
        )
        weights = "Pretrained" if cfg.architecture.pretrained else "Random"
        logger_instance.info("%s%s %-18s: %s", I, A, "Weights", weights)  # pragma: no mutate

        # Add weight variant if present (for ViT)
        if cfg.architecture.weight_variant:
            logger_instance.info(  # pragma: no mutate
                "%s%s %-18s: %s", I, A, "Weight Variant", cfg.architecture.weight_variant
            )

        precision = "AMP (Mixed)" if train.use_amp else "FP32"
        logger_instance.info("%s%s %-18s: %s", I, A, "Precision", precision)  # pragma: no mutate
        logger_instance.info("%s%s %-18s: %s", I, A, "TTA Mode", tta_status)  # pragma: no mutate
        logger_instance.info("%s%s %-18s: %s", I, A, "Repro. Mode", repro_mode)  # pragma: no mutate
        logger_instance.info("%s%s %-18s: %s", I, A, "Global Seed", train.seed)  # pragma: no mutate

    def _log_tracking_section(self, logger_instance: logging.Logger, cfg: "Config") -> None:
        """Logs tracking configuration if enabled."""
        tracking_cfg = getattr(cfg, "tracking", None)
        if tracking_cfg is None:
            return

        I = LogStyle.INDENT  # noqa: E741
        A = LogStyle.ARROW

        logger_instance.info("[TRACKING]")  # pragma: no mutate
        status = "Active" if tracking_cfg.enabled else "Disabled"
        logger_instance.info("%s%s %-18s: %s", I, A, "Status", status)  # pragma: no mutate
        if tracking_cfg.enabled:
            logger_instance.info(  # pragma: no mutate
                "%s%s %-18s: %s", I, A, "Experiment", tracking_cfg.experiment_name
            )
        logger_instance.info("")  # pragma: no mutate

    def _log_optimization_section(self, logger_instance: logging.Logger, cfg: "Config") -> None:
        """Logs optimization configuration if enabled."""
        optuna_cfg = getattr(cfg, "optuna", None)
        if optuna_cfg is None:
            return

        I = LogStyle.INDENT  # noqa: E741
        A = LogStyle.ARROW

        logger_instance.info("[OPTIMIZATION]")  # pragma: no mutate
        logger_instance.info(
            "%s%s %-18s: %s", I, A, "Trials", optuna_cfg.n_trials
        )  # pragma: no mutate
        logger_instance.info(  # pragma: no mutate
            "%s%s %-18s: %s", I, A, "Epochs/Trial", optuna_cfg.epochs
        )
        logger_instance.info(  # pragma: no mutate
            "%s%s %-18s: %s (%s)",
            I,
            A,
            "Metric",
            cfg.training.monitor_metric,
            optuna_cfg.direction,
        )
        logger_instance.info(  # pragma: no mutate
            "%s%s %-18s: %s", I, A, "Sampler", optuna_cfg.sampler_type.upper()
        )
        if optuna_cfg.enable_pruning:
            logger_instance.info(  # pragma: no mutate
                "%s%s %-18s: %s", I, A, "Pruner", optuna_cfg.pruner_type
            )
        logger_instance.info("")  # pragma: no mutate

    def _log_export_section(self, logger_instance: logging.Logger, cfg: "Config") -> None:
        """Logs export configuration if enabled."""
        export_cfg = getattr(cfg, "export", None)
        if export_cfg is None:
            return

        IND, A = LogStyle.INDENT, LogStyle.ARROW  # pragma: no mutate
        logger_instance.info("[EXPORT]")  # pragma: no mutate
        logger_instance.info(  # pragma: no mutate
            "%s%s %-18s: %s", IND, A, "Format", export_cfg.format.upper()
        )
        logger_instance.info(  # pragma: no mutate
            "%s%s %-18s: %s", IND, A, "Opset Version", export_cfg.opset_version
        )
        logger_instance.info(  # pragma: no mutate
            "%s%s %-18s: %s", IND, A, "Validate", export_cfg.validate_export
        )
        if export_cfg.quantize:
            logger_instance.info(  # pragma: no mutate
                "%s%s %-18s: %s (%s)",
                IND,
                A,
                "Quantize",
                export_cfg.quantization_type.upper(),
                export_cfg.quantization_backend,
            )
        logger_instance.info("")  # pragma: no mutate
