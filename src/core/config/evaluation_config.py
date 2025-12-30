"""
Evaluation Reporting & Visualization Schema.

This module governs the post-training diagnostic phase. It defines the 
aesthetics and structural requirements for performance reports, including 
confusion matrix colormaps, prediction grid layouts, and the serialization 
format for quantitative metrics.

The schema ensures that all visual artifacts (DPI, figure sizes) and 
tabular data are generated consistently across different experiments 
for easier comparative analysis.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import argparse

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
from pydantic import BaseModel, Field, ConfigDict, field_validator

# =========================================================================== #
#                               Internal Imports                              #
# =========================================================================== #
from .types import (
    PositiveInt, BatchSize
)

# =========================================================================== #
#                           EVALUATION CONFIGURATION                          #
# =========================================================================== #

class EvaluationConfig(BaseModel):
    """
    Controls the visual reporting and performance metric persistence.
    
    Sets the aesthetics for confusion matrices, prediction grids, and 
    defines the export format for quantitative tabular reports.
    """
    model_config = ConfigDict(
        frozen=True,
        extra="forbid"
    )
    
    batch_size: BatchSize = Field(
        default=64,
        description="Batch size used during inference/evaluation"
    )
    n_samples: PositiveInt = Field(default=12)
    fig_dpi: PositiveInt = Field(default=200)
    img_size: tuple[int, int] = (10, 10)
    cmap_confusion: str = "Blues"
    plot_style: str = "seaborn-v0_8-muted"
    report_format: str = "xlsx"

    @field_validator("report_format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Ensure the output format is compatible with downstream consumers."""
        supported = ["xlsx", "csv", "json"]
        if v.lower() not in supported:
            raise ValueError(f"Format {v} not supported. Use {supported}")
        return v.lower()
    
    save_confusion_matrix: bool = True
    save_predictions_grid: bool = True
    grid_cols: PositiveInt = Field(default=4)
    fig_size_predictions: tuple[int, int] = (12, 8)


    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "EvaluationConfig":
        """Map evaluation and reporting preferences."""
        return cls(
            batch_size=getattr(args, 'eval_batch_size', 64),
            n_samples=getattr(args, 'n_samples', 12),
            fig_dpi=getattr(args, 'fig_dpi', 200),
            plot_style=getattr(args, 'plot_style', "seaborn-v0_8-muted"),
            report_format=getattr(args, 'report_format', "xlsx")
        )