"""
Telemetry & Filesystem Manifest.

Declarative schema for filesystem orchestration, logging policy, and
experiment identity. Resolves paths, configures logging, and exports
environment-agnostic manifests.

Single Source of Truth (SSOT) for:
    * Dataset and output directory resolution and anchoring
    * Logging cadence, verbosity, and persistence policy
    * Experiment identity and run-level metadata
    * Portable, host-independent configuration serialization

Centralizes telemetry and filesystem concerns to ensure traceable,
reproducible artifacts free from host-specific filesystem leakage.
"""

# Standard Imports
import argparse
from pathlib import Path

# Third-Party Imports
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Internal Imports
from ..paths import PROJECT_ROOT
from .types import LogFrequency, LogLevel, ValidatedPath


# TELEMETRY CONFIGURATION
class TelemetryConfig(BaseModel):
    """
    Declarative manifest for telemetry, logging, and filesystem strategy.

    Manages experiment artifacts location, logging behavior, and path
    portability across environments.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        json_encoders={Path: lambda v: str(v)},
    )

    # Filesystem
    data_dir: ValidatedPath = Field(default="./dataset")
    output_dir: ValidatedPath = Field(default="./outputs")
    health_name: str = Field(default="healthcheck")

    # Telemetry
    save_model: bool = True
    log_interval: LogFrequency = Field(default=10)
    log_level: LogLevel = Field(default="INFO")

    @model_validator(mode="before")
    @classmethod
    def handle_empty_config(cls, data):
        """
        Handles empty YAML section (telemetry:) by returning default dict.

        When YAML contains 'telemetry:' with no values, Pydantic receives None.
        This validator converts None to empty dict, allowing defaults to apply.
        """
        if data is None:
            return {}
        return data

    @property
    def resolved_data_dir(self) -> Path:
        """
        Absolute path to the dataset.
        """
        return self.data_dir

    def to_portable_dict(self) -> dict:
        """
        Converts to portable dictionary with environment-agnostic paths.

        Reconciles absolute paths to project-relative paths (e.g.,
        '/home/user/project/dataset' → './dataset') to prevent
        filesystem leakage in exported configs.
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

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TelemetryConfig":
        """
        Factory from CLI arguments.

        Args:
            args: Parsed argparse namespace

        Returns:
            Configured TelemetryConfig instance
        """
        schema_fields = cls.model_fields.keys()
        params = {
            k: getattr(args, k)
            for k in schema_fields
            if hasattr(args, k) and getattr(args, k) is not None
        }
        return cls(**params)
