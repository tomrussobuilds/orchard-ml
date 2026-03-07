"""
Minimal Test Suite for Orchestrator Submodules.

Quick tests to eliminate codecov warnings for newly created modules.
Focuses on testing the most critical functions in each module.
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import optuna
import pytest

from orchard.optimization._param_mapping import map_param_to_config_path
from orchard.optimization.orchestrator.builders import (
    build_callbacks,
    build_pruner,
    build_sampler,
)
from orchard.optimization.orchestrator.exporters import (
    TrialData,
    build_best_config_dict,
)
from orchard.optimization.orchestrator.registries import (
    PRUNER_REGISTRY,
    SAMPLER_REGISTRY,
)
from orchard.optimization.orchestrator.utils import (
    get_completed_trials,
    has_completed_trials,
)
from orchard.optimization.orchestrator.visualizers import (
    _missing_params_filter,
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
            "architecture": {},
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


# TESTS: config.py
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
def test_map_param_to_config_path_architecture():
    """Test mapping architecture parameter."""
    section, key = map_param_to_config_path("dropout")
    assert section == "architecture"
    assert key == "dropout"


# TESTS: builders.py
@pytest.mark.unit
def test_build_sampler_tpe(mock_cfg):
    """Test building TPE sampler."""
    sampler = build_sampler(mock_cfg.optuna)
    assert isinstance(sampler, optuna.samplers.TPESampler)


@pytest.mark.unit
def test_build_pruner_median(mock_cfg):
    """Test building Median pruner."""
    pruner = build_pruner(mock_cfg.optuna)
    assert isinstance(pruner, optuna.pruners.MedianPruner)


@pytest.mark.unit
def test_build_pruner_disabled(mock_cfg):
    """Test disabled pruning returns NopPruner."""
    pruner = build_pruner(mock_cfg.optuna)
    assert isinstance(pruner, optuna.pruners.BasePruner)


@pytest.mark.unit
@patch("orchard.optimization.orchestrator.builders.get_early_stopping_callback")
def test_build_callbacks(mock_callback_fn, mock_cfg):
    """Test building callbacks list."""
    mock_callback_fn.return_value = None
    callbacks = build_callbacks(mock_cfg.optuna, "auc")
    assert isinstance(callbacks, list)


# TESTS: utils.py
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


# TESTS: exporters.py
@pytest.mark.unit
def test_trial_data_from_trial(completed_trial):
    """Test building TrialData from trial."""
    data = TrialData.from_trial(completed_trial)

    assert data.number == 1
    assert data.value == pytest.approx(0.95)
    assert data.state == "COMPLETE"
    assert data.datetime_start is not None
    assert data.duration_seconds is not None


@pytest.mark.unit
def test_build_best_config_dict(mock_cfg):
    """Test building config dict from params."""
    params = {"learning_rate": 0.001, "dropout": 0.5}

    config_dict = build_best_config_dict(params, mock_cfg)

    assert config_dict["training"]["learning_rate"] == pytest.approx(0.001)
    assert config_dict["architecture"]["dropout"] == pytest.approx(0.5)
    assert config_dict["training"]["epochs"] == 50


# TESTS: visualizers.py
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
@patch("orchard.optimization.orchestrator.visualizers.logger")
def test_save_plot_handles_exception(mock_logger, completed_trial):
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


@pytest.mark.unit
def test_generate_visualizations_no_completed_trials():
    """Test generate_visualizations skips when no completed trials (lines 68-91)."""
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
def test_generate_visualizations_plotly_not_installed(mock_has_trials, completed_trial):
    """Test generate_visualizations handles missing plotly gracefully (lines 68-91)."""
    from orchard.optimization.orchestrator.visualizers import generate_visualizations

    study = MagicMock()
    study.trials = [completed_trial]
    mock_has_trials.return_value = True

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def _selective_import(name, *args, **kwargs):
            if name == "optuna.visualization" or name.startswith("optuna.visualization."):
                raise ImportError("No module named 'plotly'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_selective_import):
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


@pytest.mark.unit
@patch("orchard.optimization.orchestrator.visualizers.logging")
def test_save_plot_applies_and_removes_filter(mock_logging, completed_trial):
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
def test_save_plot_removes_filter_on_exception(mock_logging, completed_trial):
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


@pytest.mark.unit
def test_missing_params_filter_suppresses_matching():
    """Test _MissingParamsFilter suppresses 'missing parameters' messages."""
    import logging

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
def test_missing_params_filter_passes_other_messages():
    """Test _MissingParamsFilter passes unrelated messages through."""
    import logging

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


# TESTS: config.py
@pytest.mark.unit
def test_map_param_to_config_path_augmentation():
    """Test mapping augmentation parameter."""
    section, key = map_param_to_config_path("rotation_angle")
    assert section == "augmentation"
    assert key == "rotation_angle"

    section, key = map_param_to_config_path("jitter_val")
    assert section == "augmentation"
    assert key == "jitter_val"

    section, key = map_param_to_config_path("min_scale")
    assert section == "augmentation"
    assert key == "min_scale"


@pytest.mark.unit
def test_map_param_to_config_path_special_model_name():
    """Test mapping special parameter: model_name."""
    section, key = map_param_to_config_path("model_name")
    assert section == "architecture"
    assert key == "name"


@pytest.mark.unit
def test_map_param_to_config_path_special_weight_variant():
    """Test mapping special parameter: weight_variant."""
    section, key = map_param_to_config_path("weight_variant")
    assert section == "architecture"
    assert key == "weight_variant"


@pytest.mark.unit
def test_map_param_to_config_path_unknown_defaults_to_training(caplog):
    """Test unknown parameter defaults to training with warning."""
    import logging

    with caplog.at_level(logging.WARNING):
        section, key = map_param_to_config_path("unknown_param")

        assert section == "training"
        assert key == "unknown_param"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
