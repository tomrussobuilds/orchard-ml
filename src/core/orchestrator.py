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
from .environment import (
    set_seed, get_cuda_name, to_device_obj, configure_system_libraries,
    apply_cpu_threads, determine_tta_mode
)
from .processes import (
    ensure_single_instance, release_single_instance
)
from .io import (
    save_config_as_yaml, load_model_weights
)
from .logger import Logger
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
        operations to ensure reproducibility and system safety.

        Returns:
            RunPaths: The verified path orchestrator for the current session.
        """
        # 1. Configure System Libraries
        configure_system_libraries()

        # 2. Reproducibility Setup
        set_seed(self.cfg.training.seed)

        # 3. Static Environment Setup
        setup_static_directories()

        # 4. Dynamic Path Initialization
        self.paths = RunPaths(
            dataset_slug=self.cfg.dataset.dataset_name,
            model_name=self.cfg.model.name,
            base_dir=self.cfg.system.output_dir
        )

        # 5. Logger Initialization
        Logger.setup(name=LOGGER_NAME, log_dir=self.paths.logs)
        self.run_logger = logging.getLogger(LOGGER_NAME)

        # 6. Environment Initialization & Safety
        self.cfg.system.manage_environment()
        ensure_single_instance(lock_file=self.cfg.system.lock_file_path, logger=self.run_logger)
        
        # 7. Metadata Preservation
        save_config_as_yaml(data=self.cfg.model_dump(mode='json'), yaml_path=self.paths.get_config_path())
        
        # 8. Environment Reporting
        self._log_initial_status()
        
        return self.paths

    def cleanup(self) -> None:
        """
        Releases system resources and removes the execution lock file.
        Guarantees a clean state for subsequent pipeline runs.
        """
        try:
            release_single_instance(self.cfg.system.lock_file_path)
            msg = "System resource lock released successfully."
            if self.run_logger: self.run_logger.info(f" » {msg}")
            else: logging.info(msg)
        except Exception as e:
            err_msg = f"Failed to release system lock: {e}"
            if self.run_logger: self.run_logger.error(f" [!] {err_msg}")
            else: logging.error(err_msg)

    def get_device(self) -> torch.device:
        """
        Resolves and caches the optimal computation device (CUDA/CPU).
        
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

    def _log_initial_status(self) -> None:
        """
        Logs the verified baseline environment configuration upon initialization.
        Uses formatted headers for visual consistency with the main pipeline.
        """
        self.run_logger.info(f"\n{'━' * 80}\n{' ENVIRONMENT INITIALIZATION ':^80}\n{'━' * 80}")
        
        self._log_hardware_section()
        self._log_dataset_section()
        self._log_strategy_section()
        
        self.run_logger.info(f" » Run Directory: {self.paths.root}")
        self.run_logger.info(f"{'━' * 80}\n")

    def _log_hardware_section(self):
        """Logs hardware-specific configuration and fallback warnings."""
        device_obj = self.get_device()
        req_dev = self.cfg.system.device
        
        self.run_logger.info(f"[HARDWARE]")
        self.run_logger.info(f"  » Device:      {str(device_obj).upper()}")
        
        if req_dev != "cpu" and device_obj.type == "cpu":
            self.run_logger.warning(f"  [!] FALLBACK: Requested {req_dev} is unavailable.")
        
        if device_obj.type == 'cuda':
            gpu_name = get_cuda_name()
            if gpu_name: self.run_logger.info(f"  » GPU:         {gpu_name}")
        elif device_obj.type == 'cpu':
            opt_threads = apply_cpu_threads(self.cfg.num_workers)
            self.run_logger.info(f"  » Threads:     {opt_threads} (Workers: {self.cfg.num_workers})")

    def _log_dataset_section(self):
        """Logs dataset metadata and channel processing modes."""
        ds = self.cfg.dataset


        self.run_logger.info(f"[DATASET]")
        self.run_logger.info(f"  » Name:        {ds.dataset_name} ({ds.img_size}px)")
        self.run_logger.info(f"  » Mode:        {ds.processing_mode}")
        self.run_logger.info(f"  » Features:    Anatomical={ds.is_anatomical}, Texture={ds.is_texture_based}")

    def _log_strategy_section(self):
        """Logs high-level training and augmentation strategies."""
        train = self.cfg.training
        tta_status = determine_tta_mode(train.use_tta, self.get_device().type)
        
        self.run_logger.info(f"[STRATEGY]")
        self.run_logger.info(f"  » Model:       {self.cfg.model.name} (Pretrained: {self.cfg.model.pretrained})")
        self.run_logger.info(f"  » TTA:         {tta_status}")
        self.run_logger.info(f"  » Hyperparams: Epochs={train.epochs}, Batch={train.batch_size}, LR={train.learning_rate:.4f}")