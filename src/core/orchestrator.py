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
from typing import Optional

# =========================================================================== #
#                             Third-Party Imports                             #
# =========================================================================== #
import torch

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from .environment import (
    set_seed, to_device_obj, configure_system_libraries, get_num_workers,
    is_repro_mode_requested, apply_cpu_threads
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

# =========================================================================== #
#                              Root Orchestrator                              #
# =========================================================================== #

class RootOrchestrator:
    """
    High-level lifecycle controller for the experiment environment.
    
    The RootOrchestrator acts as the central state machine for the pipeline. It
    manages the transition from static configuration (Pydantic models) to a 
    live execution environment, ensuring atomicity in directory creation, 
    hardware synchronization, and telemetry initialization.

    By leveraging the Context Manager pattern, it guarantees that system-level 
    resources (like file locks) are acquired before execution and safely 
    released upon termination, even in the event of unhandled exceptions.

    Attributes:
        cfg (Config): The immutable global configuration manifest.
        infra (InfrastructureManager): Handler for OS-level resource guarding.
        reporter (Reporter): Specialized utility for environment telemetry.
        paths (Optional[RunPaths]): Orchestrator for session-specific directories.
        run_logger (Optional[logging.Logger]): Active logger instance for the run.
        repro_mode (bool): Whether strict reproducibility is enforced.
        num_workers (int): Resolved number of DataLoader workers based on repro_mode.
        _device_cache (Optional[torch.device]): Memoized compute device.
    """
    
    def __init__(self, cfg, log_initializer=Logger.setup):
        """
        Initializes the orchestrator with the experiment configuration.

        Args:
            cfg (Config): The validated global configuration manifest.
            log_initializer (callable): Function to initialize the logging system.
        """
        self.cfg = cfg
        self.infra = InfrastructureManager()
        self.reporter = Reporter()
        self._log_initializer = log_initializer
        self.paths: Optional[RunPaths] = None
        self.run_logger: Optional[logging.Logger] = None
        self._device_cache: Optional[torch.device] = None
        self.repro_mode: bool = False
        self.num_workers: int = 0
    
    def __enter__(self) -> "RootOrchestrator":
        """
        Context Manager entry point. 
        Automatically triggers the core service initialization sequence.

        Returns:
            RootOrchestrator: The initialized instance ready for pipeline execution.
        """
        self.initialize_core_services()
        return self

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
        Triggers the deterministic sequence of core service initializations.

        This method synchronizes the environment following a strict order of 
        operations to ensure reproducibility, resource optimization, and 
        system safety.

        Returns:
            RunPaths: The verified path orchestrator for the current session.
        """
        # 1. Resolve Reproducibility Policy
        self.repro_mode = is_repro_mode_requested(
            cli_flag=getattr(self.cfg.training, "reproducible", False)
        )

        # 2. Reproducibility & Global Seeding Setup
        set_seed(self.cfg.training.seed, strict=self.repro_mode)

        # 3. Resource & Threading Optimization
        self.num_workers = 0 if self.repro_mode else get_num_workers()
        applied_threads = apply_cpu_threads(self.num_workers)

        # 4. Configure System Libraries
        configure_system_libraries()

        # 5. Static Environment Setup
        setup_static_directories()

        # 6. Dynamic Path Initialization
        self.paths = RunPaths(
            dataset_slug=self.cfg.dataset.dataset_name,
            model_name=self.cfg.model.name,
            base_dir=self.cfg.system.output_dir
        )

        # 7. Logger Initialization
        self.run_logger = self._log_initializer(
            name=LOGGER_NAME,
            log_dir=self.paths.logs,
            level=self.cfg.system.log_level
        )

        # 8. Environment Initialization & Safety
        self.infra.prepare_environment(
            self.cfg,
            logger=self.run_logger
        )
        
        # 9. Metadata Preservation
        save_config_as_yaml(
            self.cfg,
            self.paths.get_config_path()
        )
        
        # 10. Environment Reporting
        if self.repro_mode:
            self.run_logger.info(
                " [!] STRICT DETERMINISM ENABLED: DataLoader workers set to 0."
            )
        
        self.run_logger.debug(
            f"Compute threads synchronized: {applied_threads}"
        )

        self.reporter.log_initial_status(
            logger=self.run_logger,
            cfg=self.cfg,
            paths=self.paths,
            device=self.get_device()
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