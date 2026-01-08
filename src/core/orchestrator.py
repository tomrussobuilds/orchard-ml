"""
Environment Orchestration & Lifecycle Management.

This module provides the `RootOrchestrator`, the central authority for 
initializing and managing the experiment's execution state. It synchronizes 
hardware (`CUDA`/`CPU`), filesystem (`RunPaths`), and telemetry (`Logging`) into a 
unified, reproducible context.

Key Responsibilities:
    - Deterministic Seeding: Ensures global `RNG` state is locked, supporting 
      both standard and strict (bit-perfect) reproducibility modes.
    - Resource Guarding: Implements single-instance locking to prevent race 
      conditions on shared hardware or filesystem resources via `InfrastructureManager`.
    - Path Atomicity: Dynamically generates and validates experiment workspaces.
    - Hardware Abstraction: Manages device-specific optimizations including 
      compute thread synchronization and `DataLoader` worker scaling.
    - Lifecycle Safety: Uses the `Context Manager` pattern to guarantee resource 
      cleanup and state persistence even during runtime failures.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

# =========================================================================== #
#                             Third-Party Imports                             #
# =========================================================================== #
import torch

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from .environment import (
    set_seed, to_device_obj, configure_system_libraries, apply_cpu_threads
)
from .config.infrastructure_config import InfrastructureManager
from .io import (
    save_config_as_yaml, load_model_weights
)
from .logger import (
    Logger, Reporter
)
from .paths import (
    RunPaths, setup_static_directories, LOGGER_NAME
)
if TYPE_CHECKING:
    from .config.engine import Config

# =========================================================================== #
#                              Root Orchestrator                              #
# =========================================================================== #

class RootOrchestrator:
    """
    High-level lifecycle controller for the experiment environment.
    
    The `RootOrchestrator` acts as the central state machine for the pipeline. It
    coordinates the transition from a static configuration (`Pydantic`-validated) 
    to a live execution state, synchronizing hardware discovery, filesystem 
    provisioning, and telemetry initialization.

    By leveraging the `Context Manager` pattern, it guarantees that system-level 
    resources—such as kernel-level file locks and telemetry handlers—are 
    safely acquired before execution and released upon termination, ensuring 
    environment atomicity even during runtime failures.

    Attributes:
        cfg (Config): The validated global configuration manifest (`SSOT`).
        infra (InfrastructureManager): Handler for `OS`-level resource guarding.
        reporter (Reporter): Specialized engine for environment telemetry.
        _log_initializer (callable): A strategy function responsible for configuring 
            the logging handlers.
        paths (Optional[RunPaths]): Orchestrator for session-specific directories.
        run_logger (Optional[logging.Logger]): Active logger instance for the session.
        repro_mode (bool): Flag indicating if strict bit-perfect determinism is active.
        num_workers (int): Resolved number of `DataLoader` workers based on policy.
        _device_cache (Optional[torch.device]): Memoized compute device object.
    """
    def __init__(self, cfg: "Config", log_initializer=Logger.setup) -> None:
        """
        Initializes the orchestrator and binds the execution policy.

        Args:
            cfg (Config): The validated global configuration manifest. 
                Acts as the Single Source of Truth (`SSOT`) for all sub-systems.
            log_initializer (callable, optional): A strategy function responsible 
                for configuring the logging handlers. Must accept `name`, 
                `log_dir`, and `level`. Defaults to `Logger.setup`.
        """
        self.cfg = cfg
        self.infra = InfrastructureManager()
        self.reporter = Reporter()
        self._log_initializer = log_initializer
        
        self.paths: Optional[RunPaths] = None
        self.run_logger: Optional[logging.Logger] = None
        self._device_cache: Optional[torch.device] = None
        
        # Policy extraction from the SSOT
        self.repro_mode = self.cfg.system.use_deterministic_algorithms
        self.num_workers = self.cfg.system.effective_num_workers
    
    def __enter__(self) -> "RootOrchestrator":
        """
        Context Manager entry point. 
        Automatically triggers the core service initialization sequence via 
        `initialize_core_services()`.

        Returns:
            RootOrchestrator: The initialized instance, ready for pipeline execution.
        """
        try:
            self.initialize_core_services()
            return self
        except Exception as e:
            self.cleanup()
            raise e

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Context Manager exit point.
        Ensures that system resources are released and lock files are unlinked.

        Args:
            exc_type: The type of the exception raised.
            exc_val: The instance of the exception raised.
            exc_tb: The traceback of the exception raised.

        Returns:
            bool: Always `False` to allow exception propagation.
        """
        self.cleanup()
        return False
     
    # --- PRIVATE LIFECYCLE PHASES ---

    def _phase_1_determinism(self) -> None:
        """
        Enforces global `RNG` seeding and algorithmic determinism policies via `set_seed`.
        """
        set_seed(self.cfg.training.seed, strict=self.repro_mode)

    def _phase_2_hardware_optimization(self) -> int:
        """
        Configures compute thread affinity and accelerator libraries.
        
        Returns:
            int: The number of `CPU` threads successfully applied to the runtime.
        """
        applied_threads = apply_cpu_threads(self.num_workers)
        configure_system_libraries()
        return applied_threads
    
    def _phase_3_filesystem_provisioning(self) -> None:
        """
        Constructs the experiment workspace using `RunPaths`.
        Anchors all relative paths to the validated `PROJECT_ROOT`.
        """
        setup_static_directories()
        self.paths = RunPaths.create(
            dataset_slug=self.cfg.dataset.dataset_name,
            model_name=self.cfg.model.name,
            training_cfg=self.cfg.dump_serialized(),
            base_dir=self.cfg.system.output_dir
        )

    def _phase_4_logging_initialization(self) -> None:
        """
        Bridges the static Logger to the session-specific filesystem.
        Reconfigures handlers to enable file-based persistence in the run directory.
        """
        self.run_logger = self._log_initializer(
            name=LOGGER_NAME,
            log_dir=self.paths.logs,
            level=self.cfg.system.log_level
        )

    def _phase_5_config_persistence(self) -> None:
        """
        Mirrors the hydrated configuration state to the experiment root.
        Ensures auditability by saving a portable YAML manifest of the session.
        """
        save_config_as_yaml(
            self.cfg.dump_portable(),
            self.paths.get_config_path()
        )

    def _phase_6_infrastructure_guarding(self) -> None:
        """
        Secures system-level resource locks via InfrastructureManager.
        Prevents concurrent execution conflicts and manages environment cleanup.
        """
        self.infra.prepare_environment(
            self.cfg,
            logger=self.run_logger
        )

    def _phase_7_environment_reporting(self, applied_threads: int) -> None:
        """
        Emits the baseline environment report to the active logging streams.
        Summarizes hardware, dataset metadata, and resolved execution policies.
        """
        self.reporter.log_initial_status(
            logger=self.run_logger,
            cfg=self.cfg,
            paths=self.paths,
            device=self.get_device(),
            applied_threads=applied_threads,
            num_workers=self.num_workers
        )

    # --- PUBLIC INTERFACE ---

    def initialize_core_services(self) -> RunPaths:
        """
        Executes the linear sequence of environment initialization phases.

        Synchronizes the global state through 7 distinct phases, progressing 
        from deterministic seeding to full environment reporting.

        Returns:
            RunPaths: The verified and provisioned directory structure.
        """
        self._phase_1_determinism()
        applied_threads = self._phase_2_hardware_optimization()
        self._phase_3_filesystem_provisioning()
        
        # New segmented Telemetry & Guarding sequence
        self._phase_4_logging_initialization()
        self._phase_5_config_persistence()
        self._phase_6_infrastructure_guarding()
        self._phase_7_environment_reporting(applied_threads)
        
        return self.paths

    def cleanup(self) -> None:
        """
        Releases system resources and removes the execution lock file.
        
        Guarantees a clean state for subsequent pipeline runs by unlinking 
        `InfrastructureManager` guards and closing active `logging` handlers.
        """
        try:
            self.infra.release_resources(self.cfg, logger=self.run_logger)
        except Exception as e:
            err_msg = f"Failed to release system lock: {e}"
            if self.run_logger:
                for handler in self.run_logger.handlers[:]:
                    handler.close()
                    self.run_logger.removeHandler(handler)
            else:
                logging.error(err_msg)

    def get_device(self) -> torch.device:
        """
        Resolves and caches the optimal computation device (`CUDA`/`CPU`/`MPS`).
        
        Returns:
            torch.device: The `PyTorch` device object for model execution.
        """
        if self._device_cache is None:
            self._device_cache = to_device_obj(device_str=self.cfg.system.device)
        return self._device_cache

    def load_weights(self, model: torch.nn.Module, path: Path) -> None:
        """
        Coordinates weight restoration by bridging the model with system utilities.

        Args:
            model (torch.nn.Module): The model instance to populate.
            path (Path): Filesystem `Path` to the checkpoint file.
        """
        device = self.get_device()
        load_model_weights(model, path, device)
        
        if self.run_logger:
            self.run_logger.info(f" » Checkpoint weights restored from: {path.name}")