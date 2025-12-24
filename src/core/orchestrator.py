"""
Core Environment Orchestrator

This module centralizes the initialization of the system environment, 
coupling static configuration with runtime safety and logging services 
to provide a consistent state for the experiment.
"""
# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import logging
from pathlib import Path

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from .system import (
    set_seed, ensure_single_instance, kill_duplicate_processes, get_cuda_name
)
from .logger import Logger
from .constants import RunPaths, setup_static_directories

class RootOrchestrator:
    """
    Orchestrates the low-level initialization of the project's core services.
    
    It manages the transition from a static Config object to a live runtime 
    environment by setting up paths, logging, and system-level locks.
    """
    
    def __init__(self, cfg):
        """
        Args:
            cfg (Config): The validated global configuration manifest.
        """
        self.cfg = cfg
        self.paths = None
        self.run_logger = None

    def initialize_core_services(self) -> RunPaths:
        """
        Triggers the sequence of core service initializations.

        Steps include:
        1. Setting global seeds for reproducibility.
        2. Ensuring project-level static directory structures.
        3. Initializing unique RunPaths for the current session.
        4. Configuring the logging system with run-specific handlers.
        5. Enforcing single-instance execution and process cleanup.

        Returns:
            RunPaths: The initialized path orchestrator for the current run.
        """
        # 1. Reproducibility setup
        set_seed(self.cfg.training.seed)

        # 2. Static environment setup
        setup_static_directories()

        # 3. Dynamic path initialization
        self.paths = RunPaths(
            dataset_slug=self.cfg.dataset.dataset_name,
            model_name=self.cfg.model_name,
            base_dir=self.cfg.system.output_dir
        )

        # 4. Logger initialization
        Logger.setup(
            name=self.paths.project_id,
            log_dir=self.paths.logs
        )
        self.run_logger = logging.getLogger(self.paths.project_id)

        # 5. Environment initialization and safety
        lock_path = Path("/tmp/medmnist_training.lock")
        ensure_single_instance(
            lock_file=lock_path,
            logger=self.run_logger
        )
        kill_duplicate_processes(
            logger=self.run_logger
        )

        self._log_initial_status()
        
        return self.paths

    def _log_initial_status(self) -> None:
        """Logs the baseline environment configuration."""
        device_str = self.cfg.system.device
        self.run_logger.info(f"Execution Device: {device_str.upper()}")
        
        # Hardware fallback warning logic
        if device_str != "cpu" and not get_cuda_name() and "mps" not in device_str:
            self.run_logger.warning("Hardware Fallback: Requested accelerator not fully detected.")

        self.run_logger.info(f"Run Directory initialized: {self.paths.root}")
        self.run_logger.info(
            f"Hyperparameters: LR={self.cfg.training.learning_rate:.4f}, "
            f"Batch={self.cfg.training.batch_size}, Epochs={self.cfg.training.epochs}"
        )