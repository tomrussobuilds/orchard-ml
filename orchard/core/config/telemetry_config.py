"""
Telemetry & Filesystem Configuration.

Declarative schema for filesystem orchestration and logging policy.
Resolves paths, configures logging cadence and verbosity, and exports
environment-agnostic manifests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..paths import PROJECT_ROOT
from .types import LogFrequency, LogLevel, ValidatedPath


# TELEMETRY CONFIGURATION
class TelemetryConfig(BaseModel):
    """
    Declarative manifest for telemetry, logging, and filesystem strategy.

    Manages experiment artifacts location, logging behavior, and path
    portability across environments. Frozen after creation for immutability.

    Attributes:
        data_dir (ValidatedPath): Validated absolute path to dataset directory.
        output_dir (ValidatedPath): Validated absolute path to outputs directory.
        log_interval (LogFrequency): Epoch logging cadence (default: 10).
        log_level (LogLevel): Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Filesystem
    data_dir: ValidatedPath = Field(default="./dataset")  # type: ignore[assignment]
    output_dir: ValidatedPath = Field(default="./outputs")  # type: ignore[assignment]

    # Telemetry
    log_interval: LogFrequency = Field(default=10)
    log_level: LogLevel = Field(default="INFO")

    # I/O streaming chunk size (8192 bytes) is defined as a default parameter
    # in data_io.md5_checksum and galaxy10_converter, not here, to avoid
    # circular imports.  Removed as a config field: it was never read at runtime.

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
