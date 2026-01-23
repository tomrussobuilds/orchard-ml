"""
Minimal Test Suite for Orchestrator Submodules.

Quick tests to eliminate codecov warnings for newly created modules.
Focuses on testing the most critical functions in each module.
"""

# Standard Imports
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

# Third-Party Imports
import optuna
import pytest

from orchard.optimization.orchestrator.builders import (
    build_callbacks,
    build_pruner,
    build_sampler,
)

# Internal Imports
from orchard.optimization.orchestrator.config import (
    PRUNER_REGISTRY,
    SAMPLER_REGISTRY,
    map_param_to_config_path,
)
from orchard.optimization.orchestrator.exporters import (
    build_best_config_dict,
    build_trial_data,
)
from orchard.optimization.orchestrator.utils import (
    get_completed_trials,
    has_completed_trials,
)
from orchard.optimization.orchestrator.visualizers import (
    save_plot,
)


# FIXTURES
@pytest.fixture
def mock_cfg():
    """Minimal config mock."""
    cfg = MagicMock()
    cfg.optuna.sampler_type = "tpe"
    cfg.optuna.enable_pruning = True
    cfg.optuna.pruner_type = "median"
    cfg.training.epochs = 50
    cfg.model_dump = MagicMock(
        return_value={
            "training": {"epochs": 50},
            "model": {},
            "augmentation": {},
        }
    )
    return cfg


@pytest.fixture
def completed_trial():
    """Mock completed trial."""
    trial = MagicMock()
    trial.state = optuna.trial.TrialState.COMPLETE
    trial.number = 1
    trial.value = 0.95
    trial.params = {"learning_rate": 0.001}
    trial.datetime_start = datetime(2024, 1, 1, 0, 0, 0)
    trial.datetime_complete = datetime(2024, 1, 1, 1, 0, 0)
    return trial


#                    TESTS: config.py                                         #
@pytest.mark.unit
def test_sampler_registry_has_tpe():
    """Test SAMPLER_REGISTRY contains TPE."""
    assert "tpe" in SAMPLER_REGISTRY


@pytest.mark.unit
def test_pruner_registry_has_median():
    """Test PRUNER_REGISTRY contains Median."""
    assert "median" in PRUNER_REGISTRY


@pytest.mark.unit
def test_map_param_to_config_path_training():
    """Test mapping training parameter."""
    section, key = map_param_to_config_path("learning_rate")
    assert section == "training"
    assert key == "learning_rate"


@pytest.mark.unit
def test_map_param_to_config_path_model():
    """Test mapping model parameter."""
    section, key = map_param_to_config_path("dropout")
    assert section == "model"
    assert key == "dropout"


#                    TESTS: builders.py                                       #
@pytest.mark.unit
def test_build_sampler_tpe(mock_cfg):
    """Test building TPE sampler."""
    sampler = build_sampler("tpe", mock_cfg)
    assert isinstance(sampler, optuna.samplers.TPESampler)


@pytest.mark.unit
def test_build_pruner_median(mock_cfg):
    """Test building Median pruner."""
    pruner = build_pruner(True, "median", mock_cfg)
    assert isinstance(pruner, optuna.pruners.MedianPruner)


@pytest.mark.unit
def test_build_pruner_disabled(mock_cfg):
    """Test disabled pruning returns NopPruner."""
    pruner = build_pruner(False, "median", mock_cfg)
    assert isinstance(pruner, optuna.pruners.BasePruner)


@pytest.mark.unit
@patch("orchard.optimization.orchestrator.builders.get_early_stopping_callback")
def test_build_callbacks(mock_callback_fn, mock_cfg):
    """Test building callbacks list."""
    mock_callback_fn.return_value = None
    callbacks = build_callbacks(mock_cfg)
    assert isinstance(callbacks, list)


#                    TESTS: utils.py                                          #
@pytest.mark.unit
def test_get_completed_trials(completed_trial):
    """Test extracting completed trials."""
    study = MagicMock()
    study.trials = [completed_trial]

    completed = get_completed_trials(study)
    assert len(completed) == 1


@pytest.mark.unit
def test_has_completed_trials_true(completed_trial):
    """Test has_completed_trials returns True."""
    study = MagicMock()
    study.trials = [completed_trial]

    assert has_completed_trials(study) is True


@pytest.mark.unit
def test_has_completed_trials_false():
    """Test has_completed_trials returns False."""
    study = MagicMock()
    study.trials = []

    assert has_completed_trials(study) is False


#                    TESTS: exporters.py                                      #
@pytest.mark.unit
def test_build_trial_data(completed_trial):
    """Test building trial data dict."""
    data = build_trial_data(completed_trial)

    assert data["number"] == 1
    assert data["value"] == 0.95
    assert data["state"] == "COMPLETE"
    assert "datetime_start" in data
    assert "duration_seconds" in data


@pytest.mark.unit
def test_build_best_config_dict(mock_cfg):
    """Test building config dict from params."""
    params = {"learning_rate": 0.001, "dropout": 0.5}

    config_dict = build_best_config_dict(params, mock_cfg)

    assert config_dict["training"]["learning_rate"] == 0.001
    assert config_dict["model"]["dropout"] == 0.5
    assert config_dict["training"]["epochs"] == 50


#                    TESTS: visualizers.py                                    #
@pytest.mark.unit
def test_save_plot_success(completed_trial):
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
def test_save_plot_handles_exception(completed_trial):
    """Test save_plot handles exceptions gracefully."""
    study = MagicMock()
    study.trials = [completed_trial]

    mock_plot_fn = MagicMock(side_effect=Exception("Plot failed"))

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Should not raise
        save_plot(study, "test", mock_plot_fn, output_dir)


@pytest.mark.unit
def test_generate_visualizations_no_completed_trials():
    """Test generate_visualizations skips when no completed trials (lines 68-91)."""
    from orchard.optimization.orchestrator.visualizers import generate_visualizations

    study = MagicMock()
    study.trials = []  # No completed trials

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Should return early without trying to import plotly
        generate_visualizations(study, output_dir)

        # No HTML files should be created
        html_files = list(output_dir.glob("*.html"))
        assert len(html_files) == 0


@pytest.mark.unit
@patch("orchard.optimization.orchestrator.visualizers.has_completed_trials")
def test_generate_visualizations_plotly_not_installed(mock_has_trials, completed_trial):
    """Test generate_visualizations handles missing plotly gracefully (lines 68-91)."""
    from orchard.optimization.orchestrator.visualizers import generate_visualizations

    study = MagicMock()
    study.trials = [completed_trial]
    mock_has_trials.return_value = True

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Mock the import to raise ImportError
        with patch("builtins.__import__", side_effect=ImportError("No module named 'plotly'")):
            # Should handle ImportError gracefully
            generate_visualizations(study, output_dir)


@pytest.mark.unit
@patch("orchard.optimization.orchestrator.visualizers.has_completed_trials")
@patch("orchard.optimization.orchestrator.visualizers.save_plot")
def test_generate_visualizations_creates_all_plots(
    mock_save_plot, mock_has_trials, completed_trial
):
    """Test generate_visualizations creates all four plot types (lines 68-91)."""
    from orchard.optimization.orchestrator.visualizers import generate_visualizations

    study = MagicMock()
    study.trials = [completed_trial]
    mock_has_trials.return_value = True

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Mock the optuna.visualization imports
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

            # Should call save_plot 4 times (one for each plot type)
            assert mock_save_plot.call_count == 4

            # Verify the plot names
            plot_names = [call[0][1] for call in mock_save_plot.call_args_list]
            assert "optimization_history" in plot_names
            assert "param_importances" in plot_names
            assert "slice" in plot_names
            assert "parallel_coordinate" in plot_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
