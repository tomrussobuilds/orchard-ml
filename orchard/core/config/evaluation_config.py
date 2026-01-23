"""
Evaluation Reporting & Visualization Schema.

Defines post-training diagnostic phase requirements: visual artifacts
(confusion matrices, prediction grids) and quantitative data persistence
(Excel/JSON/CSV).

Key Features:
    * Aesthetic standardization: DPI, colormaps, plot styles for reproducible
      comparative analysis across experiments
    * Diagnostic layouts: Configurable prediction grid geometry for inspecting
      model errors and blind spots
    * Tabular export: Validated serialization formats compatible with
      downstream analysis tools
    * Resource efficiency: Inference batch size control for memory optimization

Centralizes reporting parameters to ensure standardized, publication-quality
diagnostic output for every experiment.
"""

# Standard Imports
import argparse

# Third-Party Imports
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Internal Imports
from .types import BatchSize, PositiveInt


# EVALUATION CONFIGURATION
class EvaluationConfig(BaseModel):
    """
    Visual reporting and performance metric persistence configuration.

    Controls inference settings, visualization aesthetics, and export formats
    for confusion matrices and prediction grids.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Inference
    batch_size: BatchSize = Field(default=64, description="Batch size for inference/evaluation")

    # Visualization
    n_samples: PositiveInt = Field(default=12, description="Number of samples in prediction grid")
    fig_dpi: PositiveInt = Field(default=200, description="DPI for saved figures")
    cmap_confusion: str = Field(default="Blues", description="Confusion matrix colormap")
    plot_style: str = Field(default="seaborn-v0_8-muted", description="Matplotlib plot style")
    grid_cols: PositiveInt = Field(default=4, description="Prediction grid columns")
    fig_size_predictions: tuple[PositiveInt, PositiveInt] = Field(
        default=(12, 8), description="Prediction grid size (width, height)"
    )

    # Export
    report_format: str = Field(default="xlsx", description="Report export format (xlsx, csv, json)")
    save_confusion_matrix: bool = Field(
        default=True, description="Save confusion matrix visualization"
    )
    save_predictions_grid: bool = Field(
        default=True, description="Save prediction grid visualization"
    )

    @field_validator("report_format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Validates report format, defaults to xlsx if unsupported."""
        supported = {"xlsx", "csv", "json"}
        normalized = v.lower()
        return normalized if normalized in supported else "xlsx"

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "EvaluationConfig":
        """
        Factory from CLI arguments.

        Args:
            args: Parsed argparse namespace

        Returns:
            Configured EvaluationConfig instance
        """
        params = {
            field: getattr(args, field, cls.model_fields[field].default)
            for field in cls.model_fields
            if hasattr(args, field) or cls.model_fields[field].default is not None
        }
        return cls(**params)
