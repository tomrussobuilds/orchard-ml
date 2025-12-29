"""
System & Infrastructure Configuration Schema.

This module defines the declarative schema for hardware abstraction and 
filesystem orchestration. It acts as the bridge between raw environment 
settings (CLI/YAML) and the physical resources of the machine, ensuring 
Single Source of Truth (SSOT) for device selection and path integrity.

Key Functionalities:
    * Hardware Negotiation: Resolves 'auto' device requests into concrete 
      accelerators (CUDA/MPS) based on runtime availability.
    * Environment Sanitization: Manages process lifecycle (duplicate killing) 
      and ensures lock file consistency across different OS platforms.
    * Artifact Management: Enforces validated directory structures for 
      datasets and experiment outputs via Pydantic-driven path resolution.

The SystemConfig class ensures that the Orchestrator starts on a 
predictable and hardware-aligned environment.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import argparse
import os
import tempfile
from pathlib import Path

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import torch
from pydantic import (
    BaseModel, Field, field_validator, ConfigDict
)

# =========================================================================== #
#                               Internal Imports                              #
# =========================================================================== #
from .types import ValidatedPath, PositiveInt
from ..processes import kill_duplicate_processes
from ..environment import detect_best_device
from ..paths import DATASET_DIR, OUTPUTS_ROOT

# =========================================================================== #
#                             SYSTEM CONFIGURATION                            #
# =========================================================================== #
 
class SystemConfig(BaseModel):
    """
    Manages infrastructure, hardware abstraction, and environment state.
    
    Handles SSOT (Single Source of Truth) for device selection, ensuring 
    requested accelerators (CUDA/MPS) are physically available, and manages 
    experimental artifacts through validated directory paths.
    """
    model_config = ConfigDict(
        frozen=True,
        extra="forbid"
    )
    device: str = Field(
        default_factory=detect_best_device,
        description="Computing device (cpu, cuda, mps or auto)."
    )
    data_dir: ValidatedPath = Field(default=DATASET_DIR)
    output_dir: ValidatedPath = Field(default=OUTPUTS_ROOT)
    save_model: bool = True
    log_interval: PositiveInt = Field(default=10)
    project_name: str = "medmnist_experiment"
    allow_process_kill: bool = Field(
        default=True,
        description="Enable automatic termination of duplicate processes."
    )

    @property
    def lock_file_path(self) -> Path:
        """Dynamically generates a cross-platform lock file path."""
        return Path(tempfile.gettempdir()) / f"{self.project_name}.lock"

    @field_validator("device")
    @classmethod
    def resolve_device(cls, v: str) -> str:
        """
        SSOT Validation: Ensures the requested device actually exists on this system.
        If the requested accelerator (cuda/mps) is unavailable, it self-corrects to 'cpu'.
        """
        if v == "auto":
            return detect_best_device()
        
        requested = v.lower()
        if "cuda" in requested and not torch.cuda.is_available():
            return "cpu"
        if "mps" in requested and not torch.backends.mps.is_available():
            return "cpu"
        return requested
    
    def manage_environment(self) -> None:
        """"
        Handles environment setup tasks such as killing duplicates if enabled.
        Safeguards against termination in shared cluster environments.
        """
        if not self.allow_process_kill:
            return
        is_shared = any(env in os.environ for env in ["SLURM_JOB_ID", "PBS_JOBID"])
        if is_shared:
            return
        kill_duplicate_processes()
    

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "SystemConfig":
        """Map infrastructure and hardware settings."""
        return cls(
            device=getattr(args, 'device', "auto"),
            data_dir=Path(getattr(args, 'data_dir', DATASET_DIR)),
            output_dir=Path(getattr(args, 'output_dir', OUTPUTS_ROOT)),
            save_model=getattr(args, 'save_model', True),
            log_interval=getattr(args, 'log_interval', 10),
            project_name=getattr(args, 'project_name', "medmnist_experiment")
        )