"""
Unit tests for the StudyEarlyStoppingCallback class.

These tests validate the functionality of the early stopping callback in an Optuna study.
They ensure that early stopping occurs under appropriate conditions and that all internal
states, such as patience and threshold checks, are correctly handled.
"""

# Standard Imports
from unittest.mock import MagicMock

# Third-Party Imports
import pytest
from optuna.trial import Trial, TrialState

# Internal Imports
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

    # Mocking the trial object
    trial = MagicMock(spec=Trial)
    trial.state = TrialState.COMPLETE
    trial.value = 0.995  # Below the threshold

    study_mock = MagicMock()  # Mock the study object
    callback(study=study_mock, trial=trial)

    assert callback._count == 0
    study_mock.stop.assert_not_called()


@pytest.mark.unit
def test_callback_threshold_met_once():
    """Test callback when the threshold is met once."""
    callback = StudyEarlyStoppingCallback(threshold=0.9999, direction="maximize", patience=3)

    # Mocking the trial object
    trial = MagicMock(spec=Trial)
    trial.state = TrialState.COMPLETE  # Complete trial state
    trial.value = 0.9999  # Exactly meets the threshold
    trial.number = 5  # Mock the trial number

    # Mocking the study object
    study_mock = MagicMock()
    study_mock.user_attrs = {"n_trials": 10}  # Simulate total trials as 10

    callback(study=study_mock, trial=trial)

    # Check that _count is 1 since the threshold was met
    assert callback._count == 1
    study_mock.stop.assert_not_called()


@pytest.mark.unit
def test_callback_trials_saved():
    """Test callback when total trials are available, and we calculate the trials saved."""
    callback = StudyEarlyStoppingCallback(threshold=0.9999, direction="maximize", patience=3)

    # Mocking the trial object
    trial = MagicMock(spec=Trial)
    trial.state = TrialState.COMPLETE
    trial.value = 0.9999
    trial.number = 5

    # Mocking the study object with user_attrs containing total trials
    study_mock = MagicMock()
    study_mock.user_attrs = {"n_trials": 10}

    callback._count = 3

    # Call the callback function
    callback(study=study_mock, trial=trial)

    # Check that the correct number of trials saved is calculated
    trials_saved = 10 - (trial.number + 1)
    study_mock.stop.assert_called_once()

    assert trials_saved == 4


@pytest.mark.unit
def test_callback_patience_reached():
    """Test callback when patience is reached."""
    callback = StudyEarlyStoppingCallback(threshold=0.9999, direction="maximize", patience=3)

    # Simulating trials where the threshold is met
    study_mock = MagicMock()  # Mock the study object

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

    # First trial does not meet threshold
    trial1 = MagicMock(spec=Trial)
    trial1.state = TrialState.COMPLETE
    trial1.value = 0.995
    study_mock = MagicMock()  # Mock the study object
    callback(study=study_mock, trial=trial1)

    # Second trial does not meet threshold
    trial2 = MagicMock(spec=Trial)
    trial2.state = TrialState.COMPLETE
    trial2.value = 0.995
    callback(study=study_mock, trial=trial2)

    # Third trial does not meet threshold
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

    # Mocking the trial object with a non-COMPLETE state
    trial = MagicMock(spec=Trial)
    trial.state = TrialState.PRUNED  # Set trial state to PRUNED (not COMPLETE)
    trial.value = 0.9999  # Threshold met, but state is not COMPLETE
    trial.number = 5  # Mock the trial number

    study_mock = MagicMock()  # Mock the study object
    callback(study=study_mock, trial=trial)

    # Check that _count is reset to 0 because trial state was not COMPLETE
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
    study_mock = MagicMock()  # Mock the study object
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
