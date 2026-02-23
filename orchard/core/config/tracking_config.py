"""
Tracking Configuration.

Pydantic sub-config for experiment tracking settings (MLflow).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TrackingConfig(BaseModel):
    """Configuration for MLflow experiment tracking.

    Controls whether MLflow logging is active and under which experiment
    name runs are grouped. When present in the YAML config, tracking
    is enabled by default.

    Attributes:
        enabled: Whether to activate MLflow logging for this run.
        experiment_name: MLflow experiment name (groups related runs).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = Field(default=True, description="Enable MLflow tracking")
    experiment_name: str = Field(
        default="orchard-ml",
        description="MLflow experiment name",
    )
