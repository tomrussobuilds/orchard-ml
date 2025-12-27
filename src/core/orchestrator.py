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
#                             Third-Party Imports                             #
# =========================================================================== #
import numpy as np
import torch

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from .system import (
    set_seed, ensure_single_instance, kill_duplicate_processes, get_cuda_name,
    to_device_obj, load_model_weights, configure_system_libraries,
    release_single_instance, apply_cpu_threads, determine_tta_mode
)
from .io import save_config_as_yaml, validate_npz_keys
from .logger import Logger
from .paths import RunPaths, setup_static_directories

# =========================================================================== #
#                              Root Orchestrator RootOrchestrator             #
# =========================================================================== #

class RootOrchestrator:
    """
    High-level lifecycle controller for the experiment environment.
    
    The RootOrchestrator serves as the Single Source of Truth (SSOT) for the 
    transition between static configuration (Pydantic models) and the live 
    execution state. It enforces system-level constraints, manages directory 
    atomicity, and synchronizes telemetry (logging) across the pipeline.

    Responsibilities:
        - Enforcement of execution exclusivity (Kernel-level locking).
        - Management of run-specific filesystem hierarchies.
        - Global RNG synchronization for bit-perfect reproducibility.
        - Hardware abstraction and stateful weight restoration.
    """
    
    def __init__(self, cfg):
        """
        Initializes the orchestrator with the experiment configuration.

        Args:
            cfg (Config): The validated global configuration manifest.
        """
        self.cfg = cfg
        self.paths = None
        self.run_logger = None
    
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
        Ensures that system resources are released and lock files are unlinked
        regardless of whether the pipeline succeeded or raised an exception.

        Args:
            exc_type: The type of the exception raised (if any).
            exc_val: The instance of the exception raised (if any).
            exc_tb: The traceback of the exception raised (if any).

        Returns:
            bool: Always False to allow exception propagation to the caller.
        """
        self.cleanup()
        return False

    def initialize_core_services(self) -> RunPaths:
        """
        Triggers the deterministic sequence of core service initializations.

        This method synchronizes the environment following a strict order of 
        operations:
            1. System Libraries: Configures matplotlib for headless operation.
            2. RNG Seeding: Locks global state for reproducibility.
            3. Static Layout: Ensures baseline project structure.
            4. Path Mapping: Generates unique session-specific workspace.
            5. Telemetry: Hot-swaps logger to file-persistent handlers.
            6. Guarding: Acquires system locks and purges zombie processes.
            7. Persistence: Saves the validated configuration for reference.

        Returns:
            RunPaths: The verified path orchestrator for the current session.
        """
        # 1. Configure system libraries for the current environment
        configure_system_libraries()

        # 2. Reproducibility setup: Lock global random state
        set_seed(self.cfg.training.seed)

        # 3. Static environment setup: Prepare global folder structure
        setup_static_directories()

        # 4. Dynamic path initialization: Create run-specific folder
        self.paths = RunPaths(
            dataset_slug=self.cfg.dataset.dataset_name,
            model_name=self.cfg.model_name,
            base_dir=self.cfg.system.output_dir
        )

        # 5. Logger initialization: Start file and console logging
        Logger.setup(
            name=self.paths.project_id,
            log_dir=self.paths.logs
        )
        self.run_logger = logging.getLogger(self.paths.project_id)

        # 6. Environment initialization and safety: Lock instance and clean zombies
        kill_duplicate_processes(
            logger=self.run_logger
        )
        ensure_single_instance(
            lock_file=self.cfg.system.lock_file_path,
            logger=self.run_logger
        )
        
        # 7. Metadata preservation: Save validated config as SSOT reference
        save_config_as_yaml(
            config=self.cfg,
            yaml_path=self.paths.get_config_path()
        )
        
        self._log_initial_status()
        
        return self.paths

    def cleanup(self) -> None:
        """
        Releases system resources and removes the lock file.
        To be called in the 'finally' block of main.py.
        """
        # Release the system lock to allow future instances
        try:
            release_single_instance(self.cfg.system.lock_file_path)
            if self.run_logger:
                self.run_logger.info("System lock released cleanly.")
            else:
                logging.info("System lock released cleanly.")
        except Exception as e:
            logging.error(f"Error releasing system lock: {e}")


    def get_device(self) -> torch.device:
        """
        Converts the configuration device string into a live torch.device.

        Returns:
            torch.device: The active computing device (CPU/CUDA/MPS).
        """
        return to_device_obj(self.cfg.system.device)

    def load_weights(self, model: torch.nn.Module, path: Path) -> None:
        """
        Coordinates atomic weight loading and logs the event.

        Args:
            model (torch.nn.Module): The model instance to populate.
            path (Path): Filesystem path to the checkpoint file.
        """
        device = self.get_device()
        load_model_weights(model, path, device)
        self.run_logger.info(
            f"Checkpoint weights successfully restored from: {path.name}"
        )

    def load_raw_dataset(self, path: Path) -> np.lib.npyio.NpzFile:
        """
        Loads and validates MedMNIST NPZ archives for structural integrity.

        Args:
            path (Path): Path to the MedMNIST NPZ file.

        Returns:
            np.lib.npyio.NpzFile: The validated numpy data archive.
        """
        data = np.load(path)
        validate_npz_keys(data)
        return data

    def _log_initial_status(self) -> None:
        """
        Logs the verified baseline environment configuration upon initialization.
        Includes hardware-specific optimization details and data domain mode.
        """
        device_obj = self.get_device()
        device_str = self.cfg.system.device
        self.run_logger.info(f"Execution Device: {str(device_obj).upper()}")

        # Data Domain tracking: Crucial for verifying RGB promotion logic
        mode_str = "RGB-PROMOTED" if self.cfg.dataset.force_rgb else "NATIVE-GRAY"
        self.run_logger.info(f"Data Mode: {mode_str} (Input: {self.cfg.dataset.img_size}px)")

        # Log CPU-specific thread optimizations via system utility
        if device_obj.type == 'cpu':
            optimal_threads = apply_cpu_threads(self.cfg.num_workers)
            self.run_logger.info(f"CPU Optimization: Configured with {optimal_threads} compute threads.")
            self.run_logger.info(f"Worker Strategy: {self.cfg.num_workers} data loaders active.")
        
        # TTA Status via system utility logic
        tta_status = determine_tta_mode(self.cfg.training.use_tta, device_obj.type)
        self.run_logger.info(f"TTA Status: {tta_status}")

        # Hardware fallback warning and metadata logging
        if device_str == "cuda":
            gpu_name = get_cuda_name()
            if gpu_name:
                self.run_logger.info(f"GPU Model: {gpu_name}")
        
        if device_str != "cpu" and device_obj.type == "cpu":
            self.run_logger.warning(
                f"HARDWARE FALLBACK: Requested {device_str}, but it is unavailable. Using CPU."
            )

        self.run_logger.info(f"Run Directory initialized: {self.paths.root}")
        self.run_logger.info(
            f"Hyperparameters: LR={self.cfg.training.learning_rate:.4f}, "
            f"Batch={self.cfg.training.batch_size}, Epochs={self.cfg.training.epochs}"
        )