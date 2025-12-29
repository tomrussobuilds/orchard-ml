"""
Environment Orchestration & Lifecycle Management.

This module provides the `RootOrchestrator`, the central authority for 
initializing and managing the experiment's execution state. It synchronizes 
hardware (CUDA/CPU), filesystem (RunPaths), and telemetry (Logging) into a 
unified, reproducible context.

Key Responsibilities:
    - Deterministic Seeding: Ensures global RNG state is locked for reproducibility.
    - Resource Guarding: Implements single-instance locking to prevent race 
      conditions on shared hardware or filesystem resources.
    - Path Atomicity: Dynamically generates and validates experiment workspaces.
    - Hardware Abstraction: Manages device-specific optimizations (CUDA names, 
      CPU threading levels).
    - Lifecycle Safety: Uses the Context Manager pattern to guarantee resource 
      cleanup and state persistence even during runtime failures.

The orchestrator acts as the Single Source of Truth (SSOT) for the pipeline 
environment, ensuring that if an experiment starts, it does so in a 
validated and logged state.
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
from .system import (
    set_seed, ensure_single_instance, get_cuda_name,
    to_device_obj, load_model_weights, configure_system_libraries,
    release_single_instance, apply_cpu_threads, determine_tta_mode
)
from .io import save_config_as_yaml
from .logger import Logger
from .paths import RunPaths, setup_static_directories, LOGGER_NAME

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

    Workflow Sequence:
        1. Library & RNG Seeding (Reproducibility)
        2. Filesystem Layout (RunPaths generation)
        3. Telemetry setup (Logger hot-swap)
        4. Resource Guarding (Process locking)
        5. Config Persistence (Metadata tracking)

    Attributes:
        cfg (Config): The immutable global configuration manifest.
        paths (Optional[RunPaths]): Orchestrator for session-specific directories.
        run_logger (Optional[logging.Logger]): Active logger instance for the run.
        _device_cache (Optional[torch.device]): Memoized compute device.
    """
    
    def __init__(self, cfg):
        """
        Initializes the orchestrator with the experiment configuration.

        Args:
            cfg (Config): The validated global configuration manifest.
        """
        self.cfg = cfg
        self.paths: Optional[RunPaths] = None
        self.run_logger: Optional[logging.Logger] = None
        self._device_cache: Optional[torch.device] = None
    
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
            8. Reporting: Logs the verified environment baseline.

        Returns:
            RunPaths: The verified path orchestrator for the current session.
        """
        # 1. Configure System Libraries for the current environment
        configure_system_libraries()

        # 2. Reproducibility Setup: Lock global random state
        set_seed(self.cfg.training.seed)

        # 3. Static Environment Setup: Prepare global folder structure
        setup_static_directories()

        # 4. Dynamic Path Initialization: Create run-specific folder
        self.paths = RunPaths(
            dataset_slug=self.cfg.dataset.dataset_name,
            model_name=self.cfg.model_name,
            base_dir=self.cfg.system.output_dir
        )

        # 5. Logger Initialization: Start file and console logging
        Logger.setup(
            name=LOGGER_NAME,
            log_dir=self.paths.logs
        )
        self.run_logger = logging.getLogger(LOGGER_NAME)

        # 6. Environment Initialization & Safety: Lock instance and clean zombies
        self.cfg.system.manage_environment()

        ensure_single_instance(
            lock_file=self.cfg.system.lock_file_path,
            logger=self.run_logger
        )
        
        # 7. Metadata Preservation: Save validated config as SSOT reference
        save_config_as_yaml(
            data=self.cfg.model_dump(mode='json'),
            yaml_path=self.paths.get_config_path()
        )
        
        # 8. Environment Reporting: Log the verified baseline status
        self._log_initial_status()
        
        return self.paths

    def cleanup(self) -> None:
        """
        Releases system resources and removes the lock file.
        To be called in the 'finally' block of main.py or via __exit__.
        """
        try:
            release_single_instance(self.cfg.system.lock_file_path)
            message = "System lock released cleanly."
            if self.run_logger:
                self.run_logger.info(message)
            else:
                logging.info(message)
        except Exception as e:
            logging.error(f"Error releasing system lock: {e}")

    def get_device(self) -> torch.device:
        """
        Resolves and caches the optimal computation device based on configuration.
        
        Returns:
            torch.device: The PyTorch device object for model execution.
        """
        if self._device_cache is None:
            self._device_cache = to_device_obj(
                device_str=self.cfg.system.device
            )
        return self._device_cache

    def load_weights(self, model: torch.nn.Module, path: Path) -> None:
        """
        Coordinates weight restoration by bridging the model with system utilities.
        Delegates the low-level weight loading to system utilities while
        providing telemetry.

        Args:
            model (torch.nn.Module): The model instance to populate.
            path (Path): Filesystem path to the checkpoint file.
        """
        device = self.get_device()
        load_model_weights(model, path, device)
        
        if self.run_logger:
            self.run_logger.info(
                f"Checkpoint weights successfully restored from: {path.name}"
            )

    def _log_initial_status(self) -> None:
        """
        Logs the verified baseline environment configuration upon initialization.
        """
        self.run_logger.info(f"--- Environment Status Report ---")
        
        self._log_hardware_section()
        self._log_dataset_section()
        self._log_strategy_section()
        
        self.run_logger.info(f"Run Directory: {self.paths.root}")
        self.run_logger.info(f"---------------------------------")

    def _log_hardware_section(self):
        device_obj = self.get_device()
        device_str_requested = self.cfg.system.device
        
        self.run_logger.info(f"[HARDWARE]")
        self.run_logger.info(f"  Device: {str(device_obj).upper()}")
        
        if device_str_requested != "cpu" and device_obj.type == "cpu":
            self.run_logger.warning(f"  (!) FALLBACK: Requested {device_str_requested} is unavailable.")
        
        if device_obj.type == 'cuda':
            gpu_name = get_cuda_name()
            if gpu_name: self.run_logger.info(f"  GPU: {gpu_name}")
        elif device_obj.type == 'cpu':
            opt_threads = apply_cpu_threads(self.cfg.num_workers)
            self.run_logger.info(f"  Threads: {opt_threads} (Workers: {self.cfg.num_workers})")

    def _log_dataset_section(self):
        ds = self.cfg.dataset
        if ds.effective_in_channels == 3:
            mode_str = "NATIVE-RGB" if ds.in_channels == 3 else "RGB-PROMOTED"
        else:
            mode_str = "NATIVE-GRAY"

        self.run_logger.info(f"[DATASET]")
        self.run_logger.info(f"  Name: {ds.dataset_name} ({ds.img_size}px)")
        self.run_logger.info(f"  Mode: {mode_str}")
        self.run_logger.info(f"  Anatomical: {ds.is_anatomical} | Texture: {ds.is_texture_based}")

    def _log_strategy_section(self):
        train = self.cfg.training
        tta_status = determine_tta_mode(train.use_tta, self.get_device().type)
        
        self.run_logger.info(f"[STRATEGY]")
        self.run_logger.info(f"  TTA: {tta_status}")
        self.run_logger.info(
            f"  Params: Epochs={train.epochs}, Batch={train.batch_size}, LR={train.learning_rate:.4f}"
        )