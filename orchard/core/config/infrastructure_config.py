"""
Infrastructure & Resource Lifecycle Management.

Operational bridge between declarative configuration and physical execution
environment. Manages 'clean-start' and 'graceful-stop' sequences, ensuring
hardware resource optimization and preventing concurrent run collisions via
filesystem locks.

Key Tasks:
    * Process sanitization: Guards against ghost processes and multi-process
      collisions in local environments
    * Environment locking: Mutex strategy for synchronized experimental output access
    * Resource deallocation: GPU/MPS cache flushing and temporary artifact cleanup
"""

import logging

# Standard Imports
import os
from typing import Any, Optional, Protocol

# Third-Party Imports
import torch
from pydantic import BaseModel, ConfigDict

# Internal Imports
from ..environment import DuplicateProcessCleaner, ensure_single_instance, release_single_instance


# INFRASTRUCTURE MANAGER
class HardwareAwareConfig(Protocol):
    """
    Structural contract for configurations exposing hardware manifest.

    Decouples infrastructure management from concrete implementations,
    enabling type-safe access to hardware execution policies.
    """

    hardware: Any


class InfrastructureManager(BaseModel):
    """
    Environment safeguarding and resource management executor.

    Ensures clean execution environment before runs and proper resource
    release after, preventing collisions and leaks.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    def prepare_environment(
        self, cfg: HardwareAwareConfig, logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Prepares execution environment.

        Steps:
            1. Terminate duplicate/zombie processes if allowed
            2. Acquire filesystem lock to prevent concurrent runs

        Args:
            cfg: Configuration with hardware manifest
            logger: Status reporting logger
        """
        log = logger or logging.getLogger("Infrastructure")

        # Process sanitization
        if cfg.hardware.allow_process_kill:
            cleaner = DuplicateProcessCleaner()

            # Skip on shared compute (SLURM, PBS, LSF)
            is_shared = any(env in os.environ for env in ("SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID"))

            if not is_shared:
                num_zombies = cleaner.terminate_duplicates(logger=log)
                log.info(f" » Duplicate processes terminated: {num_zombies}.")
            else:
                log.debug(" » Shared environment detected: skipping process kill.")

        # Concurrency guard
        ensure_single_instance(lock_file=cfg.hardware.lock_file_path, logger=log)
        log.info(f" » Lock acquired at {cfg.hardware.lock_file_path}")

    def release_resources(
        self, cfg: HardwareAwareConfig, logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Releases system and hardware resources gracefully.

        Steps:
            1. Release filesystem lock
            2. Flush hardware memory caches

        Args:
            cfg: Configuration with hardware manifest
            logger: Status reporting logger
        """
        log = logger or logging.getLogger("Infrastructure")

        # Release lock
        try:
            release_single_instance(cfg.hardware.lock_file_path)
            log.info(f" » Lock released at {cfg.hardware.lock_file_path}")
        except Exception as e:
            log.warning(f" » Failed to release lock: {e}")

        # Flush caches
        self._flush_compute_cache(log=log)

    def _flush_compute_cache(self, log: Optional[logging.Logger] = None) -> None:
        """Clears GPU/MPS memory to prevent fragmentation across runs."""
        log = log or logging.getLogger("Infrastructure")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log.debug(" » CUDA cache cleared.")

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
                log.debug(" » MPS cache cleared.")
            except Exception:
                log.debug(" » MPS cache cleanup failed (non-fatal).")
