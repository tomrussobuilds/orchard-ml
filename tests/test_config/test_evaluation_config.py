"""
Test Suite for EvaluationConfig.

Tests inference settings, visualization parameters,
and report export format validation.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orchard.core.config import EvaluationConfig


# EVALUATION CONFIG: DEFAULTS
@pytest.mark.unit
def test_evaluation_config_defaults():
    """Test EvaluationConfig with default values."""
    config = EvaluationConfig()

    assert config.n_samples == 12
    assert config.fig_dpi == 200
    assert config.cmap_confusion == "Blues"
    assert config.plot_style == "seaborn-v0_8-muted"
    assert config.grid_cols == 4
    assert config.fig_size_predictions == (12, 8)
    assert config.report_format == "xlsx"
    assert config.save_confusion_matrix is True
    assert config.save_predictions_grid is True


@pytest.mark.unit
def test_evaluation_config_custom_values():
    """Test EvaluationConfig with custom parameters."""
    config = EvaluationConfig(n_samples=20, fig_dpi=300, grid_cols=5, report_format="csv")

    assert config.n_samples == 20
    assert config.fig_dpi == 300
    assert config.grid_cols == 5
    assert config.report_format == "csv"


# EVALUATION CONFIG: VALIDATION
@pytest.mark.unit
def test_n_samples_positive():
    """Test n_samples must be positive."""

    config = EvaluationConfig(n_samples=1)
    assert config.n_samples == 1

    with pytest.raises(ValidationError):
        EvaluationConfig(n_samples=0)

    with pytest.raises(ValidationError):
        EvaluationConfig(n_samples=-5)


@pytest.mark.unit
def test_fig_dpi_positive():
    """Test fig_dpi must be positive."""

    config = EvaluationConfig(fig_dpi=100)
    assert config.fig_dpi == 100

    with pytest.raises(ValidationError):
        EvaluationConfig(fig_dpi=0)

    with pytest.raises(ValidationError):
        EvaluationConfig(fig_dpi=-50)


@pytest.mark.unit
def test_grid_cols_positive():
    """Test grid_cols must be positive."""

    config = EvaluationConfig(grid_cols=3)
    assert config.grid_cols == 3

    with pytest.raises(ValidationError):
        EvaluationConfig(grid_cols=0)


# EVALUATION CONFIG: FORMAT VALIDATION
@pytest.mark.unit
def test_report_format_validation_valid():
    """Test report_format accepts valid formats."""
    for fmt in ["xlsx", "csv", "json", "XLSX", "CSV", "JSON"]:
        config = EvaluationConfig(report_format=fmt)
        assert config.report_format in ["xlsx", "csv", "json"]


@pytest.mark.unit
def test_report_format_validation_invalid():
    """Test report_format raises ValueError for unsupported formats."""
    with pytest.raises(ValidationError, match="Unsupported report_format"):
        EvaluationConfig(report_format="pdf")

    with pytest.raises(ValidationError, match="Unsupported report_format"):
        EvaluationConfig(report_format="invalid")


@pytest.mark.unit
def test_report_format_case_insensitive():
    """Test report_format is case-insensitive."""
    config = EvaluationConfig(report_format="CSV")
    assert config.report_format == "csv"

    config = EvaluationConfig(report_format="XlSx")
    assert config.report_format == "xlsx"


# EVALUATION CONFIG: VISUALIZATION
@pytest.mark.unit
def test_colormap_string():
    """Test cmap_confusion accepts string values."""
    config = EvaluationConfig(cmap_confusion="Reds")
    assert config.cmap_confusion == "Reds"

    config = EvaluationConfig(cmap_confusion="viridis")
    assert config.cmap_confusion == "viridis"


@pytest.mark.unit
def test_colormap_invalid_rejected():
    """Test cmap_confusion rejects unknown colormap names."""
    with pytest.raises(ValidationError, match="Unknown matplotlib colormap"):
        EvaluationConfig(cmap_confusion="not_a_real_colormap_xyz")


@pytest.mark.unit
def test_plot_style_string():
    """Test plot_style accepts string values."""
    config = EvaluationConfig(plot_style="ggplot")
    assert config.plot_style == "ggplot"

    config = EvaluationConfig(plot_style="seaborn-v0_8-darkgrid")
    assert config.plot_style == "seaborn-v0_8-darkgrid"


@pytest.mark.unit
def test_plot_style_invalid_rejected():
    """Test plot_style rejects unknown style names."""
    with pytest.raises(ValidationError, match="Unknown matplotlib style"):
        EvaluationConfig(plot_style="not_a_real_style_xyz")


@pytest.mark.unit
def test_fig_size_tuple():
    """Test fig_size_predictions accepts tuple of positive ints."""
    config = EvaluationConfig(fig_size_predictions=(16, 10))
    assert config.fig_size_predictions == (16, 10)

    with pytest.raises(ValidationError):
        EvaluationConfig(fig_size_predictions=(0, 8))

    with pytest.raises(ValidationError):
        EvaluationConfig(fig_size_predictions=(12, -5))


# EVALUATION CONFIG: EXPORT FLAGS
@pytest.mark.unit
def test_save_flags_boolean():
    """Test save flags accept boolean values."""
    config = EvaluationConfig(save_confusion_matrix=False, save_predictions_grid=False)

    assert config.save_confusion_matrix is False
    assert config.save_predictions_grid is False


@pytest.mark.unit
def test_save_flags_default_true():
    """Test save flags default to True."""
    config = EvaluationConfig()

    assert config.save_confusion_matrix is True
    assert config.save_predictions_grid is True


# EVALUATION CONFIG: IMMUTABILITY
@pytest.mark.unit
def test_config_is_frozen():
    """Test EvaluationConfig is immutable after creation."""
    config = EvaluationConfig()

    with pytest.raises(ValidationError):
        config.n_samples = 256


@pytest.mark.unit
def test_config_forbids_extra_fields():
    """Test EvaluationConfig rejects unknown fields."""
    with pytest.raises(ValidationError):
        EvaluationConfig(unknown_param="value")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
