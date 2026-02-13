"""
Test Suite for Exporting Configurations and Study Information.

This module contains test cases for verifying the export functionality of various
configurations such as best config, study summary, and top trials. It ensures
correct behavior of configuration serialization and export to formats such as YAML, JSON, and Excel.
"""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import optuna
import pandas as pd
import pytest
from pydantic import ValidationError

from orchard.core import Config, RunPaths
from orchard.optimization import export_best_config, export_study_summary, export_top_trials
from orchard.optimization.orchestrator.exporters import (
    build_best_trial_data,
    build_top_trials_dataframe,
    build_trial_data,
)


# FIXTURES
@pytest.fixture
def study():
    """Fixture for creating a mock Optuna Study object."""
    study = MagicMock(spec=optuna.Study)
    trial_mock = MagicMock(spec=optuna.trial.FrozenTrial)
    trial_mock.state = optuna.trial.TrialState.COMPLETE
    trial_mock.params = {"param1": 0.1, "param2": 0.2}
    trial_mock.value = 0.8
    trial_mock.number = 1
    trial_mock.datetime_start = None
    trial_mock.datetime_complete = None

    study.best_trial = trial_mock
    study.best_params = trial_mock.params
    study.trials = [trial_mock]
    study.study_name = "test_study"
    study.direction = optuna.study.StudyDirection.MAXIMIZE
    return study


@pytest.fixture
def paths(tmpdir):
    """Fixture for creating a mock RunPaths object."""
    paths = MagicMock(spec=RunPaths)
    paths.reports = tmpdir.mkdir("reports")
    return paths


@pytest.fixture
def config():
    """Fixture for creating a valid Config object with ModelConfig and TrainingConfig."""
    model_config = {
        "name": "resnet_18_adapted",
        "pretrained": True,
        "dropout": 0.2,
        "weight_variant": None,
    }

    training_config = {"epochs": 10, "mixup_epochs": 5}

    return Config(model=model_config, training=training_config)


# TESTS
@pytest.mark.unit
def test_export_study_summary(study, paths):
    """Test export of study summary to JSON."""
    export_study_summary(study, paths)

    output_path = paths.reports / "study_summary.json"
    assert output_path.exists()

    with open(output_path, "r") as f:
        summary = json.load(f)
        assert "study_name" in summary
        assert "direction" in summary
        assert "trials" in summary
        assert len(summary["trials"]) == 1
        assert "best_trial" in summary


@pytest.mark.unit
def test_export_top_trials(study, paths):
    """Test export of top trials to Excel."""
    export_top_trials(study, paths, metric_name="accuracy", top_k=1)

    output_path = paths.reports / "top_10_trials.xlsx"
    assert output_path.exists()

    df = pd.read_excel(output_path)
    assert "Rank" in df.columns
    assert "Trial" in df.columns
    assert "ACCURACY" in df.columns


@pytest.mark.unit
def test_export_best_config_invalid_config(study, paths):
    """Test handling of invalid config during export."""

    invalid_model_config = "invalid_model"

    invalid_training_config = {"epochs": -10}

    with pytest.raises(ValidationError):
        invalid_config = Config(model=invalid_model_config, training=invalid_training_config)
        export_best_config(study, invalid_config, paths)

    output_path = paths.reports / "best_config.yaml"
    assert not output_path.exists()


@pytest.mark.unit
def test_export_study_summary_no_completed_trials(study, paths):
    """Test export when no completed trials exist."""
    study.trials = []
    export_study_summary(study, paths)

    output_path = paths.reports / "study_summary.json"
    assert output_path.exists()

    with open(output_path, "r") as f:
        summary = json.load(f)
        assert "n_completed" in summary
        assert summary["n_completed"] == 0
        assert "trials" in summary
        assert len(summary["trials"]) == 0


@pytest.mark.unit
def test_export_top_trials_no_completed_trials(study, paths):
    """Test export when no completed trials exist."""
    study.trials = []
    export_top_trials(study, paths, metric_name="accuracy", top_k=1)

    output_path = paths.reports / "top_10_trials.xlsx"
    assert not output_path.exists()


@pytest.mark.unit
def test_build_best_trial_data_value_error():
    """Test build_best_trial_data handles ValueError from study.best_trial."""
    study = MagicMock(spec=optuna.Study)
    study.best_trial.side_effect = ValueError("No trials")

    completed = []

    result = build_best_trial_data(study, completed)

    assert result is None


@pytest.mark.unit
def test_build_best_trial_data_value_error_with_completed_trials():
    """Test build_best_trial_data handles ValueError even with completed trials."""
    study = MagicMock(spec=optuna.Study)

    trial_mock = MagicMock(spec=optuna.trial.FrozenTrial)
    trial_mock.state = optuna.trial.TrialState.COMPLETE
    completed = [trial_mock]

    from unittest.mock import PropertyMock

    type(study).best_trial = PropertyMock(side_effect=ValueError("Corrupted study"))

    result = build_best_trial_data(study, completed)

    assert result is None


@pytest.mark.unit
def test_build_best_trial_data_value_error_direct_call():
    """Test build_best_trial_data ValueError exception path directly."""

    class BrokenStudy:
        @property
        def best_trial(self):
            raise ValueError("No best trial available")

    study = BrokenStudy()

    trial_mock = MagicMock(spec=optuna.trial.FrozenTrial)
    trial_mock.state = optuna.trial.TrialState.COMPLETE
    completed = [trial_mock]

    result = build_best_trial_data(study, completed)

    assert result is None


@pytest.mark.unit
def test_build_trial_data_without_timestamps():
    """Test build_trial_data when datetime fields are None."""
    trial = MagicMock(spec=optuna.trial.FrozenTrial)
    trial.number = 5
    trial.value = 0.95
    trial.params = {"lr": 0.001}
    trial.state = optuna.trial.TrialState.COMPLETE
    trial.datetime_start = None
    trial.datetime_complete = None

    result = build_trial_data(trial)

    assert result["number"] == 5
    assert result["value"] == pytest.approx(0.95)
    assert result["params"] == {"lr": 0.001}
    assert result["state"] == "COMPLETE"
    assert result["datetime_start"] is None
    assert result["datetime_complete"] is None
    assert result["duration_seconds"] is None


@pytest.mark.unit
def test_build_trial_data_with_only_start_time():
    """Test build_trial_data when only start time is available."""
    trial = MagicMock(spec=optuna.trial.FrozenTrial)
    trial.number = 5
    trial.value = 0.95
    trial.params = {"lr": 0.001}
    trial.state = optuna.trial.TrialState.RUNNING
    trial.datetime_start = datetime(2024, 1, 1, 12, 0, 0)
    trial.datetime_complete = None

    result = build_trial_data(trial)

    assert result["datetime_start"] is not None
    assert result["datetime_complete"] is None
    assert result["duration_seconds"] is None


@pytest.mark.unit
def test_build_top_trials_dataframe_without_duration():
    """Test build_top_trials_dataframe when trials have no timestamps."""
    trial1 = MagicMock(spec=optuna.trial.FrozenTrial)
    trial1.number = 1
    trial1.value = 0.95
    trial1.params = {"lr": 0.001, "dropout": 0.2}
    trial1.datetime_start = None
    trial1.datetime_complete = None

    trial2 = MagicMock(spec=optuna.trial.FrozenTrial)
    trial2.number = 2
    trial2.value = 0.93
    trial2.params = {"lr": 0.002, "dropout": 0.3}
    trial2.datetime_start = None
    trial2.datetime_complete = None

    sorted_trials = [trial1, trial2]

    df = build_top_trials_dataframe(sorted_trials, "auc")

    assert len(df) == 2
    assert "Rank" in df.columns
    assert "Trial" in df.columns
    assert "AUC" in df.columns
    assert "Duration (s)" not in df.columns


@pytest.mark.unit
def test_build_top_trials_dataframe_with_mixed_durations():
    """Test build_top_trials_dataframe with some trials having duration, some not."""
    trial1 = MagicMock(spec=optuna.trial.FrozenTrial)
    trial1.number = 1
    trial1.value = 0.95
    trial1.params = {"lr": 0.001}
    trial1.datetime_start = datetime(2024, 1, 1, 12, 0, 0)
    trial1.datetime_complete = datetime(2024, 1, 1, 12, 5, 30)
    trial2 = MagicMock(spec=optuna.trial.FrozenTrial)
    trial2.number = 2
    trial2.value = 0.93
    trial2.params = {"lr": 0.002}
    trial2.datetime_start = None
    trial2.datetime_complete = None

    sorted_trials = [trial1, trial2]

    df = build_top_trials_dataframe(sorted_trials, "accuracy")

    assert len(df) == 2
    assert "Duration (s)" in df.columns
    assert df.loc[0, "Duration (s)"] == 330
    assert pd.isna(df.loc[1, "Duration (s)"]) or "Duration (s)" not in df.iloc[1]


@pytest.mark.unit
def test_export_best_config_no_completed_trials_integration(minimal_config, paths):
    """Test export_best_config returns None when no completed trials exist."""
    study = MagicMock()

    trial1 = MagicMock()
    trial1.state = optuna.trial.TrialState.FAIL

    trial2 = MagicMock()
    trial2.state = optuna.trial.TrialState.PRUNED

    study.trials = [trial1, trial2]

    with patch("orchard.optimization.orchestrator.exporters.logger") as mock_logger:
        result = export_best_config(study, minimal_config, paths)
        assert result is None

        mock_logger.warning.assert_called_once()


@pytest.mark.unit
def test_export_best_config_success_path(minimal_config, paths, tmp_path):
    """Test export_best_config creates YAML when trials exist."""
    study = MagicMock()
    study.best_params = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "weight_decay": 0.0001,
    }

    trial = MagicMock()
    trial.state = optuna.trial.TrialState.COMPLETE
    trial.value = 0.95
    study.trials = [trial]

    paths.reports = tmp_path / "reports"
    paths.reports.mkdir(parents=True, exist_ok=True)

    with patch("orchard.optimization.orchestrator.exporters.build_best_config_dict") as mock_build:
        with patch("orchard.optimization.orchestrator.exporters.save_config_as_yaml") as mock_save:
            with patch("orchard.optimization.orchestrator.exporters.log_best_config_export"):
                mock_build.return_value = {
                    "training": {"learning_rate": 0.001, "batch_size": 32},
                    "dataset": {"name": "test"},
                    "architecture": {"name": "resnet"},
                }

                result = export_best_config(study, minimal_config, paths)

                assert result == paths.reports / "best_config.yaml"

                mock_build.assert_called_once_with(study.best_params, minimal_config)
                mock_save.assert_called_once()


@pytest.mark.unit
def test_export_top_trials_all_type_branches(paths, tmp_path):
    """Test that all formatting branches (float, int, bool, string) are covered."""
    study = MagicMock(spec=optuna.Study)
    study.direction = optuna.study.StudyDirection.MAXIMIZE

    trial = MagicMock(spec=optuna.trial.FrozenTrial)
    trial.state = optuna.trial.TrialState.COMPLETE
    trial.number = 1
    trial.value = 0.9567
    trial.params = {
        "learning_rate": 0.001234,
        "batch_size": 32,
        "dropout": 0.25,
        "use_amp": True,
        "optimizer": "adamw",
    }
    trial.datetime_start = datetime(2024, 1, 1, 12, 0, 0)
    trial.datetime_complete = datetime(2024, 1, 1, 12, 5, 30)

    study.trials = [trial]

    paths.reports = tmp_path / "reports"
    paths.reports.mkdir(parents=True, exist_ok=True)

    export_top_trials(study, paths, metric_name="auc", top_k=1)

    output_path = paths.reports / "top_10_trials.xlsx"
    assert output_path.exists()

    df = pd.read_excel(output_path)
    assert len(df) == 1

    assert df.loc[0, "Rank"] == 1
    assert df.loc[0, "Trial"] == 1
    assert df.loc[0, "AUC"] == pytest.approx(0.9567)
    assert df.loc[0, "learning_rate"] == pytest.approx(0.001234)
    assert df.loc[0, "batch_size"] == 32
    assert df.loc[0, "Duration (s)"] == 330


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
