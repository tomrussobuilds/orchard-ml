"""
Hardware Manifest.

Declarative schema for hardware abstraction and execution policy. Resolves
compute device, enforces determinism constraints, and derives hardware-dependent
execution parameters.

Single Source of Truth (SSOT) for:
    * Device selection (CPU/CUDA/MPS) with automatic resolution
    * Reproducibility and deterministic execution
    * DataLoader parallelism constraints
    * Process-level synchronization (cross-platform lock files)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import cast

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..environment import detect_best_device, get_num_workers
from .types import DeviceType, ProjectSlug


# HARDWARE CONFIGURATION
class HardwareConfig(BaseModel):
    """
    Hardware abstraction and execution policy configuration.

    Manages device selection, reproducibility, process synchronization,
    and DataLoader parallelism for training execution.

    Attributes:
        device: Compute device ('auto', 'cpu', 'cuda', 'mps'). Auto-resolved
            to best available accelerator.
        project_name: Project identifier for lock file naming.
        allow_process_kill: Allow terminating duplicate/zombie processes.
        reproducible: Enable strict deterministic mode (disables workers,
            enables deterministic algorithms).
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    # Core Configuration
    device: DeviceType = Field(
        default="auto", description="Device selection: 'cpu', 'cuda', 'mps', or 'auto'"
    )
    project_name: ProjectSlug = "orchard_ml"
    allow_process_kill: bool = Field(
        default=True, description="Allow terminating duplicate processes for cleanup"
    )

    reproducible: bool = Field(default=False, description="Enable strict deterministic mode")

    @field_validator("device")
    @classmethod
    def resolve_device(cls, v: DeviceType) -> DeviceType:
        """
        Validates and resolves device to available hardware.

        Auto-selects best device if 'auto', falls back to CPU if
        requested accelerator unavailable.

        Args:
            v: Requested device type

        Returns:
            Resolved device string
        """
        if v == "auto":
            return cast(DeviceType, detect_best_device())

        requested = v.lower()

        if requested == "cuda" and not torch.cuda.is_available():
            import warnings

            warnings.warn(
                "CUDA was explicitly requested but is not available. Falling back to CPU.",
                UserWarning,
                stacklevel=2,
            )
            return "cpu"
        if requested == "mps" and not torch.backends.mps.is_available():
            import warnings

            warnings.warn(
                "MPS was explicitly requested but is not available. Falling back to CPU.",
                UserWarning,
                stacklevel=2,
            )
            return "cpu"

        return cast(DeviceType, requested)

    @property
    def lock_file_path(self) -> Path:
        """
        Cross-platform lock file for preventing concurrent experiments.

        Returns:
            Path in system temp directory based on project name
        """
        safe_name = self.project_name.replace("/", "_")
        return Path(tempfile.gettempdir()) / f"{safe_name}.lock"

    @property
    def supports_amp(self) -> bool:
        """
        Check if device supports Automatic Mixed Precision.

        Returns:
            True if device is CUDA or MPS (GPU accelerators), False for CPU.
        """
        device = self.device.lower()
        return device in ("cuda", "mps") or device.startswith(("cuda:", "mps:"))

    @property
    def effective_num_workers(self) -> int:
        """
        Get optimal DataLoader workers respecting reproducibility constraints.

        Returns:
            0 if reproducible mode (avoids non-determinism from multiprocessing),
            otherwise system-detected optimal worker count.
        """
        return 0 if self.reproducible else get_num_workers()

    @property
    def use_deterministic_algorithms(self) -> bool:
        """
        Check if PyTorch should enforce deterministic operations.

        Returns:
            True if reproducible mode is enabled, False otherwise.
        """
        return self.reproducible
