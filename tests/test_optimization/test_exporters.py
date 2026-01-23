"""
Test Suite for Exporting Configurations and Study Information.

This module contains test cases for verifying the export functionality of various
configurations such as best config, study summary, and top trials. It ensures
correct behavior of configuration serialization and export to formats such as YAML, JSON, and Excel.
"""

# Standard Imports
import json
from unittest.mock import MagicMock

# Third-Party Imports
import optuna
import pandas as pd
import pytest
from pydantic import ValidationError

# Internal Imports
from orchard.core import Config, RunPaths
from orchard.optimization import export_best_config, export_study_summary, export_top_trials
from orchard.optimization.orchestrator.exporters import (
    build_best_trial_data,
    build_top_trials_dataframe,
    build_trial_data,
)


#                                Fixtures                                     #
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
    study.trials = [trial_mock]  # Add the mock trial to the study
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


@pytest.mark.unit
def test_export_study_summary(study, paths):
    """Test export of study summary to JSON."""
    export_study_summary(study, paths, metric_name="accuracy")

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

    # Test invalid model: Pass a string instead of a ModelConfig
    invalid_model_config = "invalid_model"  # Invalid config

    # Test invalid epochs: Set epochs to a negative number
    invalid_training_config = {"epochs": -10}  # Negative epoch value

    with pytest.raises(ValidationError):
        # Trying to create a Config instance with invalid model and epochs
        invalid_config = Config(model=invalid_model_config, training=invalid_training_config)
        export_best_config(study, invalid_config, paths)

    output_path = paths.reports / "best_config.yaml"
    assert not output_path.exists()


@pytest.mark.unit
def test_export_study_summary_no_completed_trials(study, paths):
    """Test export when no completed trials exist."""
    study.trials = []
    export_study_summary(study, paths, metric_name="accuracy")

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
    # Create a study that raises ValueError when accessing best_trial
    study = MagicMock(spec=optuna.Study)
    study.best_trial.side_effect = ValueError("No trials")

    completed = []  # No completed trials

    result = build_best_trial_data(study, completed)

    # Should return None when no completed trials
    assert result is None


@pytest.mark.unit
def test_build_best_trial_data_value_error_with_completed_trials():
    """Test build_best_trial_data handles ValueError even with completed trials."""
    # Create a real-ish study mock
    study = MagicMock(spec=optuna.Study)

    # Mock a completed trial
    trial_mock = MagicMock(spec=optuna.trial.FrozenTrial)
    trial_mock.state = optuna.trial.TrialState.COMPLETE
    completed = [trial_mock]

    # Configure best_trial to raise ValueError when accessed
    # Using PropertyMock for proper property mocking
    from unittest.mock import PropertyMock

    type(study).best_trial = PropertyMock(side_effect=ValueError("Corrupted study"))

    result = build_best_trial_data(study, completed)

    # Should return None when ValueError is raised
    assert result is None


@pytest.mark.unit
def test_build_best_trial_data_value_error_direct_call():
    """Test build_best_trial_data ValueError exception path directly."""

    # Another approach: use a custom class
    class BrokenStudy:
        @property
        def best_trial(self):
            raise ValueError("No best trial available")

    study = BrokenStudy()

    # Mock a completed trial
    trial_mock = MagicMock(spec=optuna.trial.FrozenTrial)
    trial_mock.state = optuna.trial.TrialState.COMPLETE
    completed = [trial_mock]

    result = build_best_trial_data(study, completed)

    # Should catch ValueError and return None
    assert result is None


@pytest.mark.unit
def test_build_trial_data_without_timestamps():
    """Test build_trial_data when datetime fields are None."""
    trial = MagicMock(spec=optuna.trial.FrozenTrial)
    trial.number = 5
    trial.value = 0.95
    trial.params = {"lr": 0.001}
    trial.state = optuna.trial.TrialState.COMPLETE
    trial.datetime_start = None  # No start time
    trial.datetime_complete = None  # No complete time

    result = build_trial_data(trial)

    assert result["number"] == 5
    assert result["value"] == 0.95
    assert result["params"] == {"lr": 0.001}
    assert result["state"] == "COMPLETE"
    assert result["datetime_start"] is None
    assert result["datetime_complete"] is None
    assert result["duration_seconds"] is None  # Duration should be None


@pytest.mark.unit
def test_build_trial_data_with_only_start_time():
    """Test build_trial_data when only start time is available."""
    from datetime import datetime

    trial = MagicMock(spec=optuna.trial.FrozenTrial)
    trial.number = 5
    trial.value = 0.95
    trial.params = {"lr": 0.001}
    trial.state = optuna.trial.TrialState.RUNNING
    trial.datetime_start = datetime(2024, 1, 1, 12, 0, 0)
    trial.datetime_complete = None  # Trial still running

    result = build_trial_data(trial)

    assert result["datetime_start"] is not None
    assert result["datetime_complete"] is None
    assert result["duration_seconds"] is None  # No duration without complete time


@pytest.mark.unit
def test_build_top_trials_dataframe_without_duration():
    """Test build_top_trials_dataframe when trials have no timestamps."""
    trial1 = MagicMock(spec=optuna.trial.FrozenTrial)
    trial1.number = 1
    trial1.value = 0.95
    trial1.params = {"lr": 0.001, "dropout": 0.2}
    trial1.datetime_start = None  # No timestamps
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
    assert "Duration (s)" not in df.columns  # Should not be added when no timestamps


@pytest.mark.unit
def test_build_top_trials_dataframe_with_mixed_durations():
    """Test build_top_trials_dataframe with some trials having duration, some not."""
    from datetime import datetime, timedelta

    # Trial with duration
    trial1 = MagicMock(spec=optuna.trial.FrozenTrial)
    trial1.number = 1
    trial1.value = 0.95
    trial1.params = {"lr": 0.001}
    trial1.datetime_start = datetime(2024, 1, 1, 12, 0, 0)
    trial1.datetime_complete = datetime(2024, 1, 1, 12, 5, 30)  # 5.5 minutes

    # Trial without duration
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

    # First trial should have duration
    assert df.loc[0, "Duration (s)"] == 330  # 5 minutes 30 seconds

    # Second trial should not have duration (NaN or missing)
    # Since not all trials have duration, pandas will handle it appropriately
    assert pd.isna(df.loc[1, "Duration (s)"]) or "Duration (s)" not in df.iloc[1]
