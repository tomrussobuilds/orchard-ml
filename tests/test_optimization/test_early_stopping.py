"""
Unit tests for the StudyEarlyStoppingCallback class.

These tests validate the functionality of the early stopping callback in an Optuna study.
They ensure that early stopping occurs under appropriate conditions and that all internal
states, such as patience and threshold checks, are correctly handled.
"""

from unittest.mock import MagicMock

import pytest
from optuna.trial import Trial, TrialState

from orchard.optimization import StudyEarlyStoppingCallback, get_early_stopping_callback


# TEST CASES
@pytest.mark.unit
def test_initialization_invalid_direction():
    """Test initialization with an invalid direction."""
    with pytest.raises(ValueError):
        StudyEarlyStoppingCallback(threshold=0.9999, direction="invalid", patience=3)


@pytest.mark.unit
def test_initialization_valid():
    """Test valid initialization."""
    callback = StudyEarlyStoppingCallback(threshold=0.9999, direction="maximize", patience=3)
    assert callback.threshold == 0.9999
    assert callback.direction == "maximize"
    assert callback.patience == 3


@pytest.mark.unit
def test_callback_threshold_not_met():
    """Test callback when the threshold is not met."""
    callback = StudyEarlyStoppingCallback(threshold=0.9999, direction="maximize", patience=3)

    trial = MagicMock(spec=Trial)
    trial.state = TrialState.COMPLETE
    trial.value = 0.995

    study_mock = MagicMock()
    callback(study=study_mock, trial=trial)

    assert callback._count == 0
    study_mock.stop.assert_not_called()


@pytest.mark.unit
def test_callback_threshold_met_once():
    """Test callback when the threshold is met once."""
    callback = StudyEarlyStoppingCallback(threshold=0.9999, direction="maximize", patience=3)

    trial = MagicMock(spec=Trial)
    trial.state = TrialState.COMPLETE
    trial.value = 0.9999
    trial.number = 5

    study_mock = MagicMock()
    study_mock.user_attrs = {"n_trials": 10}

    callback(study=study_mock, trial=trial)

    assert callback._count == 1
    study_mock.stop.assert_not_called()


@pytest.mark.unit
def test_callback_trials_saved():
    """Test callback when total trials are available, and we calculate the trials saved."""
    callback = StudyEarlyStoppingCallback(threshold=0.9999, direction="maximize", patience=3)

    trial = MagicMock(spec=Trial)
    trial.state = TrialState.COMPLETE
    trial.value = 0.9999
    trial.number = 5

    study_mock = MagicMock()
    study_mock.user_attrs = {"n_trials": 10}

    callback._count = 3

    callback(study=study_mock, trial=trial)

    trials_saved = 10 - (trial.number + 1)
    study_mock.stop.assert_called_once()

    assert trials_saved == 4


@pytest.mark.unit
def test_callback_patience_reached():
    """Test callback when patience is reached."""
    callback = StudyEarlyStoppingCallback(threshold=0.9999, direction="maximize", patience=3)

    study_mock = MagicMock()

    trial1 = MagicMock(spec=Trial)
    trial1.state = TrialState.COMPLETE
    trial1.value = 0.9999
    callback(study=study_mock, trial=trial1)

    trial2 = MagicMock(spec=Trial)
    trial2.state = TrialState.COMPLETE
    trial2.value = 0.9999
    callback(study=study_mock, trial=trial2)

    trial3 = MagicMock(spec=Trial)
    trial3.state = TrialState.COMPLETE
    trial3.value = 0.9999
    callback(study=study_mock, trial=trial3)

    assert callback._count == 3
    study_mock.stop.assert_called_once()


@pytest.mark.unit
def test_callback_threshold_not_met_after_patience():
    """Test callback when threshold is not met after patience is reached."""
    callback = StudyEarlyStoppingCallback(threshold=0.9999, direction="maximize", patience=3)

    trial1 = MagicMock(spec=Trial)
    trial1.state = TrialState.COMPLETE
    trial1.value = 0.995
    study_mock = MagicMock()
    callback(study=study_mock, trial=trial1)

    trial2 = MagicMock(spec=Trial)
    trial2.state = TrialState.COMPLETE
    trial2.value = 0.995
    callback(study=study_mock, trial=trial2)

    trial3 = MagicMock(spec=Trial)
    trial3.state = TrialState.COMPLETE
    trial3.value = 0.995
    callback(study=study_mock, trial=trial3)

    assert callback._count == 0
    study_mock.stop.assert_not_called()


@pytest.mark.unit
def test_callback_trial_state_not_complete():
    """Test callback when the trial state is not COMPLETE (e.g., PRUNED)."""
    callback = StudyEarlyStoppingCallback(threshold=0.9999, direction="maximize", patience=3)

    trial = MagicMock(spec=Trial)
    trial.state = TrialState.PRUNED
    trial.value = 0.9999
    trial.number = 5

    study_mock = MagicMock()
    callback(study=study_mock, trial=trial)

    assert callback._count == 0
    study_mock.stop.assert_not_called()


@pytest.mark.unit
def test_callback_inactive_when_disabled():
    """Test callback behavior when disabled."""
    callback = StudyEarlyStoppingCallback(
        threshold=0.9999, direction="maximize", patience=3, enabled=False
    )

    trial = MagicMock(spec=Trial)
    trial.state = TrialState.COMPLETE
    trial.value = 0.9999
    study_mock = MagicMock()
    callback(study=study_mock, trial=trial)

    assert callback._count == 0
    study_mock.stop.assert_not_called()


@pytest.mark.unit
def test_get_early_stopping_callback_valid():
    """Test the get_early_stopping_callback factory function."""
    callback = get_early_stopping_callback(
        metric_name="auc", direction="maximize", threshold=None, patience=3, enabled=True
    )
    assert isinstance(callback, StudyEarlyStoppingCallback)


@pytest.mark.unit
def test_get_early_stopping_callback_invalid_metric():
    """Test the get_early_stopping_callback factory function with an invalid metric."""
    callback = get_early_stopping_callback(
        metric_name="invalid_metric", direction="maximize", threshold=None, patience=3, enabled=True
    )
    assert callback is None


@pytest.mark.unit
def test_callback_trial_value_none():
    """Test callback when trial.value is None (edge case for type safety)."""
    callback = StudyEarlyStoppingCallback(threshold=0.9999, direction="maximize", patience=3)
    callback._count = 2  # Simulate some previous successes

    trial = MagicMock(spec=Trial)
    trial.state = TrialState.COMPLETE
    trial.value = None  # Edge case: completed but no value

    study_mock = MagicMock()
    callback(study=study_mock, trial=trial)

    # Should reset count and not stop
    assert callback._count == 0
    study_mock.stop.assert_not_called()
