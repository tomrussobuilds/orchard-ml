"""
Unit tests for orchestrator visualization functions.

Tests save_plot, generate_visualizations, and _missing_params_filter from
orchard/optimization/orchestrator/visualizers.py.
"""

from __future__ import annotations

import logging
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from orchard.optimization.orchestrator.visualizers import (
    _missing_params_filter,
    save_plot,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def completed_trial():  # type: ignore
    """Mock completed trial."""
    import optuna

    trial = MagicMock()
    trial.state = optuna.trial.TrialState.COMPLETE
    trial.number = 1
    trial.value = 0.95
    trial.params = {"learning_rate": 0.001}
    trial.datetime_start = datetime(2024, 1, 1, 0, 0, 0)
    trial.datetime_complete = datetime(2024, 1, 1, 1, 0, 0)
    return trial


# ---------------------------------------------------------------------------
# save_plot
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_save_plot_success(completed_trial) -> None:  # type: ignore
    """Test save_plot saves HTML file."""
    study = MagicMock()
    study.trials = [completed_trial]

    mock_plot_fn = MagicMock()
    mock_fig = MagicMock()
    mock_plot_fn.return_value = mock_fig

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        save_plot(study, "test", mock_plot_fn, output_dir)

        mock_plot_fn.assert_called_once_with(study)
        mock_fig.write_html.assert_called_once()


@pytest.mark.unit
@patch("orchard.optimization.orchestrator.visualizers.logger")
def test_save_plot_handles_exception(mock_logger: MagicMock, completed_trial) -> None:  # type: ignore
    """Test save_plot logs warning with plot_name and exception on failure."""
    study = MagicMock()
    study.trials = [completed_trial]

    mock_plot_fn = MagicMock(side_effect=ValueError("Plot failed"))

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        save_plot(study, "test_plot", mock_plot_fn, output_dir)

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0]
        # Verify plot_name and exception are passed as format args
        assert call_args[1] == "test_plot"
        assert isinstance(call_args[2], ValueError)


# ---------------------------------------------------------------------------
# generate_visualizations
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_generate_visualizations_no_completed_trials() -> None:
    """Test generate_visualizations skips when no completed trials."""
    from orchard.optimization.orchestrator.visualizers import generate_visualizations

    study = MagicMock()
    study.trials = []

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        generate_visualizations(study, output_dir)

        html_files = list(output_dir.glob("*.html"))
        assert len(html_files) == 0


@pytest.mark.unit
@patch("orchard.optimization.orchestrator.visualizers.has_completed_trials")
def test_generate_visualizations_plotly_not_installed(  # type: ignore
    mock_has_trials: MagicMock, completed_trial
) -> None:
    """Test generate_visualizations handles missing plotly gracefully."""
    from orchard.optimization.orchestrator.visualizers import generate_visualizations

    study = MagicMock()
    study.trials = [completed_trial]
    mock_has_trials.return_value = True

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def _selective_import(name, *args, **kwargs):  # type: ignore
            if name == "optuna.visualization" or name.startswith("optuna.visualization."):
                raise ImportError("No module named 'plotly'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_selective_import):
            generate_visualizations(study, output_dir)


@pytest.mark.unit
@patch("orchard.optimization.orchestrator.visualizers.has_completed_trials")
@patch("orchard.optimization.orchestrator.visualizers.save_plot")
def test_generate_visualizations_creates_all_plots(  # type: ignore
    mock_save_plot: MagicMock, mock_has_trials: MagicMock, completed_trial
) -> None:
    """Test generate_visualizations creates all four plot types."""
    from orchard.optimization.orchestrator.visualizers import generate_visualizations

    study = MagicMock()
    study.trials = [completed_trial]
    mock_has_trials.return_value = True

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        with patch.dict(
            "sys.modules",
            {
                "optuna.visualization": MagicMock(
                    plot_optimization_history=MagicMock(),
                    plot_param_importances=MagicMock(),
                    plot_slice=MagicMock(),
                    plot_parallel_coordinate=MagicMock(),
                )
            },
        ):
            generate_visualizations(study, output_dir)

            assert mock_save_plot.call_count == 4

            for call in mock_save_plot.call_args_list:
                call_study, plot_name, plot_fn, call_dir = call[0]
                assert call_study is study
                assert plot_name in {
                    "optimization_history",
                    "param_importances",
                    "slice",
                    "parallel_coordinate",
                }
                assert callable(plot_fn)
                assert call_dir is output_dir


# ---------------------------------------------------------------------------
# Filter management in save_plot
# ---------------------------------------------------------------------------


@pytest.mark.unit
@patch("orchard.optimization.orchestrator.visualizers.logging")
def test_save_plot_applies_and_removes_filter(mock_logging: MagicMock, completed_trial) -> None:  # type: ignore
    """Test save_plot adds/removes _missing_params_filter on the correct logger."""
    study = MagicMock()
    study.trials = [completed_trial]

    mock_plot_fn = MagicMock()
    mock_fig = MagicMock()
    mock_plot_fn.return_value = mock_fig

    mock_pc_logger = MagicMock()
    mock_logging.getLogger.return_value = mock_pc_logger

    with tempfile.TemporaryDirectory() as tmpdir:
        save_plot(study, "test", mock_plot_fn, Path(tmpdir))

        mock_pc_logger.addFilter.assert_called_once_with(_missing_params_filter)
        mock_pc_logger.removeFilter.assert_called_once_with(_missing_params_filter)


@pytest.mark.unit
@patch("orchard.optimization.orchestrator.visualizers.logging")
def test_save_plot_removes_filter_on_exception(mock_logging: MagicMock, completed_trial) -> None:  # type: ignore
    """Test save_plot removes filter even when plot_fn raises."""
    study = MagicMock()
    study.trials = [completed_trial]

    mock_plot_fn = MagicMock(side_effect=RuntimeError("boom"))
    mock_pc_logger = MagicMock()
    mock_logging.getLogger.return_value = mock_pc_logger

    with tempfile.TemporaryDirectory() as tmpdir:
        save_plot(study, "test", mock_plot_fn, Path(tmpdir))

        mock_pc_logger.addFilter.assert_called_once_with(_missing_params_filter)
        mock_pc_logger.removeFilter.assert_called_once_with(_missing_params_filter)


# ---------------------------------------------------------------------------
# _missing_params_filter
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_missing_params_filter_suppresses_matching() -> None:
    """Test _MissingParamsFilter suppresses 'missing parameters' messages."""
    record = logging.LogRecord(
        name="optuna.visualization._parallel_coordinate",
        level=logging.WARNING,
        pathname="",
        lineno=0,
        msg="Your study has trials with missing parameters.",
        args=(),
        exc_info=None,
    )
    assert _missing_params_filter.filter(record) is False


@pytest.mark.unit
def test_missing_params_filter_passes_other_messages() -> None:
    """Test _MissingParamsFilter passes unrelated messages through."""
    record = logging.LogRecord(
        name="optuna.visualization._parallel_coordinate",
        level=logging.WARNING,
        pathname="",
        lineno=0,
        msg="Some other warning message.",
        args=(),
        exc_info=None,
    )
    assert _missing_params_filter.filter(record) is True
