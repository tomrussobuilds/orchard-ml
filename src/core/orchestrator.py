"""
Environment Orchestration & Lifecycle Management.

This module provides the `RootOrchestrator`, the central authority for 
initializing and managing the experiment's execution state. It synchronizes 
hardware (CUDA/CPU), filesystem (RunPaths), and telemetry (Logging) into a 
unified, reproducible context.

Key Responsibilities:
    - Deterministic Seeding: Ensures global RNG state is locked, supporting 
      both standard and strict (bit-perfect) reproducibility modes.
    - Resource Guarding: Implements single-instance locking to prevent race 
      conditions on shared hardware or filesystem resources via InfrastructureManager.
    - Path Atomicity: Dynamically generates and validates experiment workspaces.
    - Hardware Abstraction: Manages device-specific optimizations including 
      compute thread synchronization and DataLoader worker scaling.
    - Lifecycle Safety: Uses the Context Manager pattern to guarantee resource 
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
    
    The RootOrchestrator acts as the central state machine for the pipeline. It
    coordinates the transition from a static configuration (Pydantic-validated) 
    to a live execution state, synchronizing hardware discovery, filesystem 
    provisioning, and telemetry initialization.

    By leveraging the Context Manager pattern, it guarantees that system-level 
    resources—such as kernel-level file locks and telemetry handlers—are 
    safely acquired before execution and released upon termination, ensuring 
    environment atomicity even during runtime failures.

    Attributes:
        cfg (Config): The validated global configuration manifest (SSOT).
        infra (InfrastructureManager): Handler for OS-level resource guarding.
        reporter (Reporter): Specialized engine for environment telemetry.
        paths (Optional[RunPaths]): Orchestrator for session-specific directories.
        run_logger (Optional[logging.Logger]): Active logger instance for the session.
        repro_mode (bool): Flag indicating if strict bit-perfect determinism is active.
        num_workers (int): Resolved number of DataLoader workers based on policy.
        _device_cache (Optional[torch.device]): Memoized compute device object.
    """
    def __init__(self, cfg : "Config", log_initializer=Logger.setup) -> None:
        """
        Initializes the orchestrator with the experiment configuration.

        Args:
            cfg (Config): The validated global configuration manifest.
            log_initializer (callable): A strategy function to initialize
            the logging subsystem. Defaults to Logger.setup.
        """
        self.cfg = cfg
        self.infra = InfrastructureManager()
        self.reporter = Reporter()
        self._log_initializer = log_initializer
        self.paths: Optional[RunPaths] = None
        self.run_logger: Optional[logging.Logger] = None
        self._device_cache: Optional[torch.device] = None
        self.repro_mode = self.cfg.system.use_deterministic_algorithms
        self.num_workers = self.cfg.system.effective_num_workers
    
    def __enter__(self) -> "RootOrchestrator":
        """
        Context Manager entry point. 
        Automatically triggers the core service initialization sequence.

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
            bool: Always False to allow exception propagation.
        """
        self.cleanup()
        return False

    def initialize_core_services(self) -> RunPaths:
        """
        Executes the linear initialization sequence for the experiment environment.

        This method synchronizes the system state following a strict protocol:
        1. Policy & Determinism: Sets global RNG seeds and reproducibility flags.
        2. Hardware & Threading: Optimizes CPU/GPU resources and library configs.
        3. Filesystem & Workspace: Provisions static and dynamic run directories.
        4. Telemetry & Persistence: Activates logging, guards, and config mirroring.

        Returns:
            RunPaths: The verified and provisioned path orchestrator for the session.
        """
        # --- PHASE 1: DETERMINISM & REPRODUCIBILITY ---
        set_seed(self.cfg.training.seed, strict=self.repro_mode)

        # --- PHASE 2: HARDWARE & SYSTEM OPTIMIZATION ---
        applied_threads = apply_cpu_threads(self.num_workers)
        configure_system_libraries()

        # --- PHASE 3: FILESYSTEM PROVISIONING ---
        setup_static_directories()
        self.paths = RunPaths.create(
            dataset_slug=self.cfg.dataset.dataset_name,
            model_name=self.cfg.model.name,
            cfg_dict=self.cfg.dump_serialized(),
            base_dir=self.cfg.system.output_dir
        )

        # --- PHASE 4: TELEMETRY, SAFETY & REPORTING ---
        # Initialize logging first to capture subsequent setup events
        self.run_logger = self._log_initializer(
            name=LOGGER_NAME,
            log_dir=self.paths.logs,
            level=self.cfg.system.log_level
        )

        # Secure hardware/process locks
        self.infra.prepare_environment(
            self.cfg,
            logger=self.run_logger
        )
        
        # Mirror the active configuration to the run directory
        save_config_as_yaml(
            self.cfg.dump_portable(),
            self.paths.get_config_path()
        )

        # Emit the final environment baseline report
        self.reporter.log_initial_status(
            logger=self.run_logger,
            cfg=self.cfg,
            paths=self.paths,
            device=self.get_device(),
            applied_threads=applied_threads,
            num_workers=self.num_workers
        )
        
        return self.paths

    def cleanup(self) -> None:
        """
        Releases system resources and removes the execution lock file via InfrastructureManager.
        Guarantees a clean state for subsequent pipeline runs.
        """
        try:
            self.infra.release_resources(self.cfg, logger=self.run_logger)
        except Exception as e:
            err_msg = f"Failed to release system lock: {e}"
            if self.run_logger:
                self.run_logger.error(f" [!] {err_msg}")
            else:
                logging.error(err_msg)

    def get_device(self) -> torch.device:
        """
        Resolves and caches the optimal computation device (CUDA/CPU/MPS).
        
        Returns:
            torch.device: The PyTorch device object for model execution.
        """
        if self._device_cache is None:
            self._device_cache = to_device_obj(device_str=self.cfg.system.device)
        return self._device_cache

    def load_weights(self, model: torch.nn.Module, path: Path) -> None:
        """
        Coordinates weight restoration by bridging the model with system utilities.

        Args:
            model (torch.nn.Module): The model instance to populate.
            path (Path): Filesystem path to the checkpoint file.
        """
        device = self.get_device()
        load_model_weights(model, path, device)
        
        if self.run_logger:
            self.run_logger.info(f" » Checkpoint weights restored from: {path.name}")