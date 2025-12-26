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
    to_device_obj, load_model_weights, get_optimal_threads
)
from .io import save_config_as_yaml, validate_npz_keys
from .logger import Logger
from .paths import RunPaths, setup_static_directories

# =========================================================================== #
#                              Root Orchestrator                              #
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

    def initialize_core_services(self) -> RunPaths:
        """
        Triggers the deterministic sequence of core service initializations.

        This method synchronizes the environment following a strict order of 
        operations:
            1. RNG Seeding: Locks global state for reproducibility.
            2. Static Layout: Ensures baseline project structure.
            3. Path Mapping: Generates unique session-specific workspace.
            4. Telemetry: Hot-swaps logger to file-persistent handlers.
            5. Guarding: Acquires system locks and purges zombie processes.
            6. Persistence: Saves the validated configuration for reference.

        Returns:
            RunPaths: The verified path orchestrator for the current session.
        """
        # 1. Reproducibility setup: Lock global random state
        set_seed(self.cfg.training.seed)

        # 2. Static environment setup: Prepare global folder structure
        setup_static_directories()

        # 3. Dynamic path initialization: Create run-specific folder
        self.paths = RunPaths(
            dataset_slug=self.cfg.dataset.dataset_name,
            model_name=self.cfg.model_name,
            base_dir=self.cfg.system.output_dir
        )

        # 4. Logger initialization: Start file and console logging
        Logger.setup(
            name=self.paths.project_id,
            log_dir=self.paths.logs
        )
        self.run_logger = logging.getLogger(self.paths.project_id)

        # 5. Environment initialization and safety: Lock instance and clean zombies
        lock_path = Path("/tmp/medmnist_training.lock")
        ensure_single_instance(
            lock_file=lock_path,
            logger=self.run_logger
        )
        kill_duplicate_processes(
            logger=self.run_logger
        )
        
        # 6. Metadata preservation: Save validated config as SSOT reference
        save_config_as_yaml(
            config=self.cfg,
            yaml_path=self.paths.get_config_path()
        )
        
        self._log_initial_status()
        
        return self.paths
    
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
        Includes hardware-specific optimization details.
        """
        device_obj = self.get_device()
        device_str = self.cfg.system.device
        self.run_logger.info(f"Execution Device: {str(device_obj).upper()}")

        # Log CPU-specific thread optimizations
        if device_obj.type == 'cpu':
            optimal_threads = get_optimal_threads(self.cfg.num_workers)
            torch.set_num_threads(optimal_threads)
            self.run_logger.info(f"CPU Optimization: Configured with {optimal_threads} compute threads.")
            self.run_logger.info(f"Worker Strategy: {self.cfg.num_workers} data loaders active.")
        
        # Hardware fallback warning and metadata logging
        if device_str == "cuda":
            gpu_name = get_cuda_name()
            if gpu_name:
                self.run_logger.info(f"GPU Model: {gpu_name}")
        
        if device_str != "cpu" and self.get_device().type == "cpu":
            self.run_logger.warning(
                f"HARDWARE FALLBACK: Requested {device_str}, but it is unavailable. Using CPU."
            )

        self.run_logger.info(f"Run Directory initialized: {self.paths.root}")
        self.run_logger.info(
            f"Hyperparameters: LR={self.cfg.training.learning_rate:.4f}, "
            f"Batch={self.cfg.training.batch_size}, Epochs={self.cfg.training.epochs}"
        )
    
    def get_tta_status(self) -> str:
        """
        Determines the TTA (Test-Time Augmentation) operational mode based 
        on hardware availability and user configuration.

        Returns:
            str: Description of the TTA execution state.
        """
        if not self.cfg.training.use_tta:
            return "DISABLED"
        
        # Consistent with engine.py logic where CPU mode uses a lighter augmentation set
        if self.get_device().type != "cpu":
            return "FULL (Accelerated - All transforms)"
        
        return "LIGHT (CPU Optimized - Subset of transforms)"