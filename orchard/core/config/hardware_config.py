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

# Standard Imports
import argparse
import tempfile
from pathlib import Path

# Third-Party Imports
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Internal Imports
from ..environment import detect_best_device, get_num_workers
from .types import DeviceType, ProjectSlug


# HARDWARE CONFIGURATION
class HardwareConfig(BaseModel):
    """
    Hardware abstraction and execution policy configuration.

    Manages device selection, reproducibility, process synchronization,
    and DataLoader parallelism.
    """

    model_config = ConfigDict(
        frozen=False,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    # Core Configuration
    device: DeviceType = Field(
        default="auto", description="Device selection: 'cpu', 'cuda', 'mps', or 'auto'"
    )
    project_name: ProjectSlug = "vision_experiment"
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
            return detect_best_device()

        requested = v.lower()

        if requested == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if requested == "mps" and not torch.backends.mps.is_available():
            return "cpu"

        return requested

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
        """Whether device supports Automatic Mixed Precision."""
        device = self.device.lower()
        return device in ("cuda", "mps") or device.startswith(("cuda:", "mps:"))

    @property
    def effective_num_workers(self) -> int:
        """
        Optimal DataLoader workers respecting reproducibility constraints.

        Returns:
            0 if reproducible mode (avoids non-determinism),
            otherwise system-detected optimal count
        """
        return 0 if self.reproducible else get_num_workers()

    @property
    def use_deterministic_algorithms(self) -> bool:
        """Whether PyTorch should enforce deterministic operations."""
        return self.reproducible

    @classmethod
    def for_optuna(cls, **kwargs) -> "HardwareConfig":
        """
        Create HardwareConfig for Optuna trials with reproducibility enabled.
        """
        kwargs["reproducible"] = True
        return cls(**kwargs)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "HardwareConfig":
        """
        Factory from command-line arguments.

        Args:
            args: Parsed argparse namespace

        Returns:
            Configured HardwareConfig instance
        """
        schema_fields = cls.model_fields.keys()
        params = {
            k: getattr(args, k)
            for k in schema_fields
            if hasattr(args, k) and getattr(args, k) is not None
        }
        return cls(**params)
