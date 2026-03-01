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
    * Resource efficiency: Configurable output artifacts for memory optimization

Centralizes reporting parameters to ensure standardized, publication-quality
diagnostic output for every experiment.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ...exceptions import OrchardConfigError
from .types import PositiveInt


# EVALUATION CONFIGURATION
class EvaluationConfig(BaseModel):
    """
    Visual reporting and performance metric persistence configuration.

    Controls inference settings, visualization aesthetics, and export formats
    for confusion matrices and prediction grids.

    Attributes:
        n_samples: Number of samples to display in prediction grid.
        fig_dpi: DPI resolution for saved figure files.
        cmap_confusion: Matplotlib colormap for confusion matrix.
        plot_style: Matplotlib style preset for all visualizations.
        grid_cols: Number of columns in prediction grid layout.
        fig_size_predictions: Figure dimensions (width, height) in inches.
        report_format: Export format for metrics report (xlsx, csv, json).
        save_confusion_matrix: Whether to generate confusion matrix plot.
        save_predictions_grid: Whether to generate prediction samples grid.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

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
        """
        Validate and normalize report format.

        Args:
            v: Requested report format string.

        Returns:
            Normalized format string (xlsx, csv, or json).

        Raises:
            ValueError: If format is not one of xlsx, csv, json.
        """
        supported = {"xlsx", "csv", "json"}
        normalized = v.lower()
        if normalized not in supported:
            raise OrchardConfigError(
                f"Unsupported report_format '{v}'. Choose from: {sorted(supported)}"
            )
        return normalized
