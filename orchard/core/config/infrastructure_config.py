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

from __future__ import annotations

import logging
import os
from typing import Any, Protocol

import torch
from pydantic import BaseModel, ConfigDict

from ..environment import DuplicateProcessCleaner, ensure_single_instance, release_single_instance


# PROTOCOLS
class InfraManagerProtocol(Protocol):
    """
    Protocol defining infrastructure management interface.

    Enables dependency injection and mocking in tests while ensuring
    consistent lifecycle management across implementations.
    """

    def prepare_environment(self, cfg: "HardwareAwareConfig", logger: logging.Logger) -> None:
        """
        Prepare execution environment before experiment run.

        Args:
            cfg: Configuration with hardware manifest access.
            logger: Logger instance for status reporting.
        """
        ...  # pragma: no cover

    def release_resources(self, cfg: "HardwareAwareConfig", logger: logging.Logger) -> None:
        """
        Release resources allocated during environment preparation.

        Args:
            cfg: Configuration used during resource allocation.
            logger: Logger instance for status reporting.
        """
        ...  # pragma: no cover


class HardwareAwareConfig(Protocol):
    """
    Structural contract for configurations exposing hardware manifest.

    Decouples infrastructure management from concrete Config implementations,
    enabling type-safe access to hardware execution policies.

    Attributes:
        hardware: HardwareConfig instance with device and lock settings.
    """

    hardware: Any


class InfrastructureManager(BaseModel):
    """
    Environment safeguarding and resource management executor.

    Ensures clean execution environment before runs and proper resource
    release after, preventing concurrent experiment collisions and
    GPU memory leaks.

    Lifecycle:
        1. prepare_environment(): Kill zombies, acquire lock
        2. [Experiment runs]
        3. release_resources(): Release lock, flush caches
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    def prepare_environment(
        self, cfg: HardwareAwareConfig, logger: logging.Logger | None = None
    ) -> None:
        """
        Prepare execution environment for experiment run.

        Performs pre-run cleanup and resource acquisition to ensure
        isolated, collision-free experiment execution.

        Steps:
            1. Terminate duplicate/zombie processes (if allow_process_kill=True
               and not in shared compute environment like SLURM/PBS/LSF)
            2. Acquire filesystem lock to prevent concurrent runs using
               the same project name

        Args:
            cfg: Configuration object with hardware.allow_process_kill and
                hardware.lock_file_path attributes.
            logger: Logger for status messages. Defaults to 'Infrastructure'.
        """
        log = logger or logging.getLogger("Infrastructure")

        # Process sanitization
        if cfg.hardware.allow_process_kill:
            cleaner = DuplicateProcessCleaner()

            # Skip on shared compute (SLURM, PBS, LSF) or distributed launchers (torchrun)
            is_shared = any(
                env in os.environ
                for env in ("SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID", "RANK", "LOCAL_RANK")
            )

            if not is_shared:
                num_zombies = cleaner.terminate_duplicates(logger=log)
                log.debug(f" » Duplicate processes terminated: {num_zombies}.")
            else:
                log.debug(" » Shared environment detected: skipping process kill.")

        # Concurrency guard
        ensure_single_instance(lock_file=cfg.hardware.lock_file_path, logger=log)
        log.debug(f" » Lock acquired at {cfg.hardware.lock_file_path}")

    def release_resources(
        self, cfg: HardwareAwareConfig, logger: logging.Logger | None = None
    ) -> None:
        """
        Release system and hardware resources gracefully after experiment.

        Performs cleanup to ensure resources are properly freed and available
        for subsequent runs. Handles errors gracefully to avoid blocking
        experiment completion.

        Steps:
            1. Release filesystem lock at cfg.hardware.lock_file_path
            2. Flush GPU/MPS memory caches to prevent VRAM fragmentation

        Args:
            cfg: Configuration object with hardware.lock_file_path attribute.
            logger: Logger for status messages. Defaults to 'Infrastructure'.

        Note:
            Lock release failures are logged as warnings but do not raise,
            ensuring experiment completion even with cleanup issues.
        """
        log = logger or logging.getLogger("Infrastructure")

        # Release lock
        try:
            release_single_instance(cfg.hardware.lock_file_path)
            log.info("  » System lock released")
        except OSError as e:
            log.warning(f" » Failed to release lock: {e}")

        # Flush caches
        self._flush_compute_cache(log=log)

    def _flush_compute_cache(self, log: logging.Logger | None = None) -> None:
        """
        Clear GPU/MPS memory caches to prevent fragmentation across runs.

        Args:
            log: Logger for debug output (defaults to 'Infrastructure' logger).
        """
        log = log or logging.getLogger("Infrastructure")

        # Full session teardown (see also OptunaObjective._cleanup for per-trial flush)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log.debug(" » CUDA cache cleared.")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
                log.debug(" » MPS cache cleared.")
            except RuntimeError:
                log.debug(" » MPS cache cleanup failed (non-fatal).")
