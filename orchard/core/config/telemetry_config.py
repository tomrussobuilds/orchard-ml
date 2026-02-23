"""
Telemetry & Filesystem Manifest.

Declarative schema for filesystem orchestration, logging policy, and
experiment identity. Resolves paths, configures logging, and exports
environment-agnostic manifests.

Single Source of Truth (SSOT) for:
    - Dataset and output directory resolution and anchoring
    - Logging cadence and verbosity
    - Experiment identity and run-level metadata
    - Portable, host-independent configuration serialization

Attributes:
    data_dir: Validated path to dataset directory (default: ./dataset).
    output_dir: Validated path to outputs directory (default: ./outputs).
    log_level: Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..paths import PROJECT_ROOT
from .types import LogFrequency, LogLevel, PositiveInt, ValidatedPath


# TELEMETRY CONFIGURATION
class TelemetryConfig(BaseModel):
    """
    Declarative manifest for telemetry, logging, and filesystem strategy.

    Manages experiment artifacts location, logging behavior, and path
    portability across environments. Frozen after creation for immutability.

    Attributes:
        data_dir: Validated absolute path to dataset directory.
        output_dir: Validated absolute path to outputs directory.
        log_level: Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        io_chunk_size: Streaming chunk size in bytes for checksums and downloads
            (hardcoded in data_io to avoid circular import).
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        json_encoders={Path: lambda v: str(v)},
    )

    # Filesystem
    data_dir: ValidatedPath = Field(default="./dataset")  # type: ignore[assignment]
    output_dir: ValidatedPath = Field(default="./outputs")  # type: ignore[assignment]

    # Telemetry
    log_interval: LogFrequency = Field(default=10)
    log_level: LogLevel = Field(default="INFO")

    # I/O — io_chunk_size is hardcoded in data_io.py to avoid circular import;
    # this field documents the canonical value but is not read at runtime.
    io_chunk_size: PositiveInt = Field(default=8192, description="Streaming chunk size (bytes)")

    @model_validator(mode="before")
    @classmethod
    def handle_empty_config(cls, data: Any) -> Any:
        """
        Handle empty YAML section by returning default dict.

        When YAML contains 'telemetry:' with no values, Pydantic receives None.
        This validator converts None to empty dict, allowing field defaults
        to apply correctly.

        Args:
            data: Raw input data from YAML or dict.

        Returns:
            Empty dict if data is None, otherwise the original data.
        """
        if data is None:
            return {}
        return data

    @property
    def resolved_data_dir(self) -> Path:
        """
        Absolute path to the dataset directory.

        Returns:
            Resolved data_dir Path (already validated and absolute).
        """
        return self.data_dir

    def to_portable_dict(self) -> dict:
        """
        Convert to portable dictionary with environment-agnostic paths.

        Transforms absolute paths to project-relative paths (e.g.,
        '/home/user/project/dataset' -> './dataset') to prevent
        host-specific filesystem leakage in exported configs.

        Returns:
            Dictionary with all paths converted to relative strings.
        """
        data = self.model_dump()

        for field in ("data_dir", "output_dir"):
            full_path = Path(data[field])
            if full_path.is_relative_to(PROJECT_ROOT):
                relative_path = full_path.relative_to(PROJECT_ROOT)
                data[field] = f"./{relative_path}"
            else:
                data[field] = str(full_path)

        return data
