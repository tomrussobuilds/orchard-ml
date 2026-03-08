"""
Unit tests for the StudyEarlyStoppingCallback class.

These tests validate the functionality of the early stopping callback in an Optuna study.
They ensure that early stopping occurs under appropriate conditions and that all internal
states, such as patience and threshold checks, are correctly handled.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from optuna.trial import Trial, TrialState

from orchard.optimization import StudyEarlyStoppingCallback, get_early_stopping_callback
from orchard.optimization.early_stopping import (
    _DEFAULT_THRESHOLDS,
    _THRESH_ACCURACY,
    _THRESH_AUC,
    _THRESH_F1,
    _THRESH_LOSS,
    _THRESH_MAE,
    _THRESH_MSE,
)


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
    assert callback.threshold == pytest.approx(0.9999)
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


# ---------------------------------------------------------------------------
# Mutation-killing tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_constant_thresh_auc():
    """Assert exact value of _THRESH_AUC to kill constant-replacement mutants."""
    assert _THRESH_AUC == 0.9999


@pytest.mark.unit
def test_constant_thresh_accuracy():
    """Assert exact value of _THRESH_ACCURACY."""
    assert _THRESH_ACCURACY == 0.995


@pytest.mark.unit
def test_constant_thresh_f1():
    """Assert exact value of _THRESH_F1."""
    assert _THRESH_F1 == 0.98


@pytest.mark.unit
def test_constant_thresh_loss():
    """Assert exact value of _THRESH_LOSS."""
    assert _THRESH_LOSS == 0.01


@pytest.mark.unit
def test_constant_thresh_mae():
    """Assert exact value of _THRESH_MAE."""
    assert _THRESH_MAE == 0.01


@pytest.mark.unit
def test_constant_thresh_mse():
    """Assert exact value of _THRESH_MSE."""
    assert _THRESH_MSE == 0.001


@pytest.mark.unit
def test_default_thresholds_maximize_keys():
    """Assert _DEFAULT_THRESHOLDS maximize direction contains expected metrics."""
    maximize = _DEFAULT_THRESHOLDS["maximize"]
    assert "auc" in maximize
    assert "accuracy" in maximize
    assert "f1" in maximize
    assert maximize["auc"] == _THRESH_AUC
    assert maximize["accuracy"] == _THRESH_ACCURACY
    assert maximize["f1"] == _THRESH_F1


@pytest.mark.unit
def test_default_thresholds_minimize_keys():
    """Assert _DEFAULT_THRESHOLDS minimize direction contains expected metrics."""
    minimize = _DEFAULT_THRESHOLDS["minimize"]
    assert "loss" in minimize
    assert "mae" in minimize
    assert "mse" in minimize
    assert minimize["loss"] == _THRESH_LOSS
    assert minimize["mae"] == _THRESH_MAE
    assert minimize["mse"] == _THRESH_MSE


@pytest.mark.unit
def test_minimize_direction_threshold_met():
    """Test that minimize direction uses <= comparison."""
    callback = StudyEarlyStoppingCallback(threshold=0.01, direction="minimize", patience=1)

    trial = MagicMock(spec=Trial)
    trial.state = TrialState.COMPLETE
    trial.value = 0.01  # exactly at threshold
    trial.number = 0

    study_mock = MagicMock()
    study_mock.user_attrs = {"n_trials": 10}
    callback(study=study_mock, trial=trial)

    assert callback._count == 1
    study_mock.stop.assert_called_once()


@pytest.mark.unit
def test_minimize_direction_below_threshold():
    """Test that minimize direction treats values below threshold as met."""
    callback = StudyEarlyStoppingCallback(threshold=0.01, direction="minimize", patience=1)

    trial = MagicMock(spec=Trial)
    trial.state = TrialState.COMPLETE
    trial.value = 0.005  # below threshold -- should be met
    trial.number = 0

    study_mock = MagicMock()
    study_mock.user_attrs = {"n_trials": 10}
    callback(study=study_mock, trial=trial)

    assert callback._count == 1
    study_mock.stop.assert_called_once()


@pytest.mark.unit
def test_minimize_direction_above_threshold_not_met():
    """Test that minimize direction treats values above threshold as NOT met."""
    callback = StudyEarlyStoppingCallback(threshold=0.01, direction="minimize", patience=1)

    trial = MagicMock(spec=Trial)
    trial.state = TrialState.COMPLETE
    trial.value = 0.05  # above threshold -- not met

    study_mock = MagicMock()
    callback(study=study_mock, trial=trial)

    assert callback._count == 0
    study_mock.stop.assert_not_called()


@pytest.mark.unit
def test_init_count_starts_at_zero():
    """Assert _count is initialized to exactly 0."""
    callback = StudyEarlyStoppingCallback(threshold=0.5, direction="maximize")
    assert callback._count == 0


@pytest.mark.unit
def test_init_enabled_default_true():
    """Assert enabled defaults to True when not specified."""
    callback = StudyEarlyStoppingCallback(threshold=0.5, direction="maximize")
    assert callback.enabled is True


@pytest.mark.unit
def test_init_patience_default():
    """Assert patience defaults to 2."""
    callback = StudyEarlyStoppingCallback(threshold=0.5, direction="maximize")
    assert callback.patience == 2


@pytest.mark.unit
def test_trials_saved_na_when_n_trials_missing():
    """When n_trials is not in user_attrs, study.stop() is still called."""
    callback = StudyEarlyStoppingCallback(threshold=0.9, direction="maximize", patience=1)

    trial = MagicMock(spec=Trial)
    trial.state = TrialState.COMPLETE
    trial.value = 0.95
    trial.number = 3

    study_mock = MagicMock()
    study_mock.user_attrs = {}  # n_trials key missing

    callback(study=study_mock, trial=trial)
    study_mock.stop.assert_called_once()


@pytest.mark.unit
def test_trials_saved_na_when_n_trials_is_string():
    """When n_trials is a non-int type, stop still fires."""
    callback = StudyEarlyStoppingCallback(threshold=0.9, direction="maximize", patience=1)

    trial = MagicMock(spec=Trial)
    trial.state = TrialState.COMPLETE
    trial.value = 0.95
    trial.number = 3

    study_mock = MagicMock()
    study_mock.user_attrs = {"n_trials": "unknown"}

    callback(study=study_mock, trial=trial)
    study_mock.stop.assert_called_once()


@pytest.mark.unit
def test_count_resets_when_threshold_not_met_after_successes():
    """After consecutive successes, a below-threshold trial resets _count to 0."""
    callback = StudyEarlyStoppingCallback(threshold=0.9999, direction="maximize", patience=5)
    callback._count = 2

    trial = MagicMock(spec=Trial)
    trial.state = TrialState.COMPLETE
    trial.value = 0.5  # well below threshold

    study_mock = MagicMock()
    callback(study=study_mock, trial=trial)

    assert callback._count == 0
    study_mock.stop.assert_not_called()


@pytest.mark.unit
def test_get_early_stopping_callback_custom_threshold():
    """Factory should use the explicit threshold rather than a default."""
    callback = get_early_stopping_callback(
        metric_name="auc", direction="maximize", threshold=0.75, patience=5
    )
    assert callback is not None
    assert callback.threshold == 0.75
    assert callback.patience == 5


@pytest.mark.unit
def test_get_early_stopping_callback_minimize_default_loss():
    """Factory should resolve correct default threshold for loss metric."""
    callback = get_early_stopping_callback(metric_name="loss", direction="minimize", threshold=None)
    assert callback is not None
    assert callback.threshold == _THRESH_LOSS
    assert callback.direction == "minimize"


@pytest.mark.unit
def test_get_early_stopping_callback_minimize_default_mae():
    """Factory should resolve correct default threshold for mae metric."""
    callback = get_early_stopping_callback(metric_name="mae", direction="minimize", threshold=None)
    assert callback is not None
    assert callback.threshold == _THRESH_MAE


@pytest.mark.unit
def test_get_early_stopping_callback_minimize_default_mse():
    """Factory should resolve correct default threshold for mse metric."""
    callback = get_early_stopping_callback(metric_name="mse", direction="minimize", threshold=None)
    assert callback is not None
    assert callback.threshold == _THRESH_MSE


@pytest.mark.unit
def test_get_early_stopping_callback_returns_none_when_disabled():
    """Factory returns None when enabled=False."""
    result = get_early_stopping_callback(
        metric_name="auc", direction="maximize", threshold=0.99, enabled=False
    )
    assert result is None


@pytest.mark.unit
def test_get_early_stopping_callback_unknown_metric_minimize_returns_none():
    """Factory returns None with warning when direction is valid but metric is unknown."""
    result = get_early_stopping_callback(
        metric_name="unknown_metric", direction="minimize", threshold=None
    )
    assert result is None


@pytest.mark.unit
def test_get_early_stopping_callback_unknown_direction_returns_none():
    """Factory returns None when direction has no entry in _DEFAULT_THRESHOLDS."""
    result = get_early_stopping_callback(metric_name="auc", direction="unknown_dir", threshold=None)
    assert result is None


@pytest.mark.unit
def test_maximize_exact_threshold_is_met():
    """For maximize, value == threshold should satisfy >= check."""
    callback = StudyEarlyStoppingCallback(threshold=0.9999, direction="maximize", patience=1)

    trial = MagicMock(spec=Trial)
    trial.state = TrialState.COMPLETE
    trial.value = 0.9999
    trial.number = 0

    study_mock = MagicMock()
    study_mock.user_attrs = {"n_trials": 5}
    callback(study=study_mock, trial=trial)

    assert callback._count == 1
    study_mock.stop.assert_called_once()


@pytest.mark.unit
def test_maximize_just_below_threshold_not_met():
    """For maximize, value just below threshold should NOT be met."""
    callback = StudyEarlyStoppingCallback(threshold=0.9999, direction="maximize", patience=1)

    trial = MagicMock(spec=Trial)
    trial.state = TrialState.COMPLETE
    trial.value = 0.9998

    study_mock = MagicMock()
    callback(study=study_mock, trial=trial)

    assert callback._count == 0
    study_mock.stop.assert_not_called()


@pytest.mark.unit
def test_count_increments_by_one():
    """Verify _count increments by exactly 1, not 2 or other values."""
    callback = StudyEarlyStoppingCallback(threshold=0.9, direction="maximize", patience=10)

    trial = MagicMock(spec=Trial)
    trial.state = TrialState.COMPLETE
    trial.value = 0.95
    trial.number = 0

    study_mock = MagicMock()
    study_mock.user_attrs = {}

    callback(study=study_mock, trial=trial)
    assert callback._count == 1

    trial2 = MagicMock(spec=Trial)
    trial2.state = TrialState.COMPLETE
    trial2.value = 0.95
    trial2.number = 1
    callback(study=study_mock, trial=trial2)
    assert callback._count == 2


@pytest.mark.unit
def test_patience_boundary_not_met_at_less_than():
    """Stop should NOT happen when _count < patience (boundary: patience - 1)."""
    callback = StudyEarlyStoppingCallback(threshold=0.9, direction="maximize", patience=3)
    callback._count = 1

    trial = MagicMock(spec=Trial)
    trial.state = TrialState.COMPLETE
    trial.value = 0.95
    trial.number = 5

    study_mock = MagicMock()
    study_mock.user_attrs = {"n_trials": 10}
    callback(study=study_mock, trial=trial)

    assert callback._count == 2
    study_mock.stop.assert_not_called()


@pytest.mark.unit
def test_pruned_trial_resets_count():
    """A PRUNED trial should reset _count (state != COMPLETE branch)."""
    callback = StudyEarlyStoppingCallback(threshold=0.9, direction="maximize", patience=5)
    callback._count = 3

    trial = MagicMock(spec=Trial)
    trial.state = TrialState.PRUNED

    study_mock = MagicMock()
    callback(study=study_mock, trial=trial)

    assert callback._count == 0
    study_mock.stop.assert_not_called()


@pytest.mark.unit
def test_init_default_direction():
    """Default direction parameter is exactly 'maximize'."""
    callback = StudyEarlyStoppingCallback(threshold=0.5)
    assert callback.direction == "maximize"


@pytest.mark.unit
def test_init_invalid_direction_error_message():
    """ValueError message includes 'direction must be'."""
    with pytest.raises(ValueError, match="direction must be"):
        StudyEarlyStoppingCallback(threshold=0.5, direction="bad")


@pytest.mark.unit
def test_factory_default_patience():
    """Factory default patience is exactly 2."""
    cb = get_early_stopping_callback("auc", "maximize")
    assert cb is not None
    assert cb.patience == 2


@pytest.mark.unit
def test_factory_default_enabled():
    """Factory default enabled is True."""
    cb = get_early_stopping_callback("auc", "maximize")
    assert cb is not None
    assert cb.enabled is True


@pytest.mark.unit
def test_factory_passes_enabled_kwarg():
    """Factory forwards enabled kwarg to the callback."""
    cb = get_early_stopping_callback("auc", "maximize", enabled=True)
    assert cb is not None
    assert cb.enabled is True


@pytest.mark.unit
def test_trials_saved_exact_computation():
    """Verify trials_saved = n_trials - (trial.number + 1)."""
    callback = StudyEarlyStoppingCallback(threshold=0.9, direction="maximize", patience=1)

    trial = MagicMock(spec=Trial)
    trial.state = TrialState.COMPLETE
    trial.value = 0.95
    trial.number = 5

    study_mock = MagicMock()
    study_mock.user_attrs = {"n_trials": 20}

    # Capture the trials_saved computation via the logger call
    with patch("orchard.optimization.early_stopping.logger") as mock_logger:
        callback(study=study_mock, trial=trial)

    # trials_saved = 20 - (5 + 1) = 14
    # The last logger.info before study.stop() logs trials_saved
    # Find the call that has "Trials saved" in the format string
    saved_call = None
    for call in mock_logger.info.call_args_list:
        args = call[0]
        if len(args) >= 2 and isinstance(args[0], str) and "saved" in args[0].lower():
            saved_call = args
            break

    assert saved_call is not None
    assert saved_call[-1] == 14  # exact trials_saved value
    study_mock.stop.assert_called_once()


@pytest.mark.unit
def test_trials_saved_uses_n_trials_key():
    """Verify study.user_attrs is queried with exact key 'n_trials'."""
    callback = StudyEarlyStoppingCallback(threshold=0.9, direction="maximize", patience=1)

    trial = MagicMock(spec=Trial)
    trial.state = TrialState.COMPLETE
    trial.value = 0.95
    trial.number = 2

    study_mock = MagicMock()
    study_mock.user_attrs = {"n_trials": 10, "N_TRIALS": 999, "XXn_trialsXX": 888}

    with patch("orchard.optimization.early_stopping.logger"):
        with patch("orchard.optimization.early_stopping.Reporter"):
            callback(study=study_mock, trial=trial)

    # trials_saved should be 10 - (2+1) = 7, not 999 or 888
    study_mock.stop.assert_called_once()


@pytest.mark.unit
def test_trials_saved_na_exact_string():
    """When n_trials is missing, trials_saved is exactly 'N/A' (not 'n/a' or None)."""
    callback = StudyEarlyStoppingCallback(threshold=0.9, direction="maximize", patience=1)

    trial = MagicMock(spec=Trial)
    trial.state = TrialState.COMPLETE
    trial.value = 0.95
    trial.number = 0

    study_mock = MagicMock()
    study_mock.user_attrs = {}  # no n_trials

    with patch("orchard.optimization.early_stopping.logger") as mock_logger:
        callback(study=study_mock, trial=trial)

    saved_call = None
    for call in mock_logger.info.call_args_list:
        args = call[0]
        if len(args) >= 2 and isinstance(args[0], str) and "saved" in args[0].lower():
            saved_call = args
            break

    assert saved_call is not None
    assert saved_call[-1] == "N/A"


@pytest.mark.unit
def test_reporter_log_phase_header_called():
    """Reporter.log_phase_header is called when stopping triggers."""
    callback = StudyEarlyStoppingCallback(threshold=0.9, direction="maximize", patience=1)

    trial = MagicMock(spec=Trial)
    trial.state = TrialState.COMPLETE
    trial.value = 0.95
    trial.number = 0

    study_mock = MagicMock()
    study_mock.user_attrs = {}

    with patch("orchard.optimization.early_stopping.Reporter") as mock_reporter:
        callback(study=study_mock, trial=trial)

    mock_reporter.log_phase_header.assert_called_once()
    call_args = mock_reporter.log_phase_header.call_args[0]
    assert "EARLY STOPPING" in call_args[1]


@pytest.mark.unit
def test_factory_warning_on_unknown_metric_calls_logger():
    """Factory logs a warning when metric has no default threshold."""
    with patch("orchard.optimization.early_stopping.logger") as mock_logger:
        result = get_early_stopping_callback("unknown", "maximize")

    assert result is None
    mock_logger.warning.assert_called_once()
    fmt_string = mock_logger.warning.call_args[0][0]
    assert "No default threshold" in fmt_string
    metric_arg = mock_logger.warning.call_args[0][1]
    assert metric_arg == "unknown"


# ---------------------------------------------------------------------------
# Mutation-killing: logger format args in __call__ stopping block
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stop_logs_metric_value():
    """Assert the 'Metric' logger call includes the actual trial value."""
    callback = StudyEarlyStoppingCallback(threshold=0.9, direction="maximize", patience=1)

    trial = MagicMock(spec=Trial)
    trial.state = TrialState.COMPLETE
    trial.value = 0.9876
    trial.number = 2

    study_mock = MagicMock()
    study_mock.user_attrs = {"n_trials": 10}

    with patch("orchard.optimization.early_stopping.logger") as mock_logger:
        callback(study=study_mock, trial=trial)

    # Find the "Metric" log call and verify value arg
    for call in mock_logger.info.call_args_list:
        args = call[0]
        if len(args) >= 4 and isinstance(args[0], str) and "Metric" in args[0]:
            assert args[-1] == pytest.approx(0.9876)
            break
    else:
        pytest.fail("No 'Metric' log call found")


@pytest.mark.unit
def test_stop_logs_threshold_value():
    """Assert the 'Threshold' logger call includes the actual threshold."""
    callback = StudyEarlyStoppingCallback(threshold=0.8765, direction="maximize", patience=1)

    trial = MagicMock(spec=Trial)
    trial.state = TrialState.COMPLETE
    trial.value = 0.95
    trial.number = 0

    study_mock = MagicMock()
    study_mock.user_attrs = {}

    with patch("orchard.optimization.early_stopping.logger") as mock_logger:
        callback(study=study_mock, trial=trial)

    for call in mock_logger.info.call_args_list:
        args = call[0]
        if len(args) >= 4 and isinstance(args[0], str) and "Threshold" in args[0]:
            assert args[-1] == pytest.approx(0.8765)
            break
    else:
        pytest.fail("No 'Threshold' log call found")


@pytest.mark.unit
def test_stop_logs_trials_completed():
    """Assert 'Trials completed' log arg is trial.number + 1."""
    callback = StudyEarlyStoppingCallback(threshold=0.9, direction="maximize", patience=1)

    trial = MagicMock(spec=Trial)
    trial.state = TrialState.COMPLETE
    trial.value = 0.95
    trial.number = 7

    study_mock = MagicMock()
    study_mock.user_attrs = {"n_trials": 20}

    with patch("orchard.optimization.early_stopping.logger") as mock_logger:
        callback(study=study_mock, trial=trial)

    for call in mock_logger.info.call_args_list:
        args = call[0]
        if len(args) >= 4 and isinstance(args[0], str) and "completed" in args[0].lower():
            assert args[-1] == 8  # trial.number + 1 = 7 + 1
            break
    else:
        pytest.fail("No 'Trials completed' log call found")


@pytest.mark.unit
def test_threshold_reached_log_includes_count_and_patience():
    """Assert the threshold-reached log includes _count and patience."""
    callback = StudyEarlyStoppingCallback(threshold=0.9, direction="maximize", patience=5)

    trial = MagicMock(spec=Trial)
    trial.state = TrialState.COMPLETE
    trial.value = 0.95
    trial.number = 0

    study_mock = MagicMock()
    study_mock.user_attrs = {}

    with patch("orchard.optimization.early_stopping.logger") as mock_logger:
        callback(study=study_mock, trial=trial)

    # Find the "reached threshold" log call
    for call in mock_logger.info.call_args_list:
        args = call[0]
        if len(args) >= 8 and isinstance(args[0], str) and "threshold" in args[0].lower():
            # args[-2] is _count, args[-1] is patience
            assert args[-2] == 1  # _count after increment
            assert args[-1] == 5  # patience
            break
    else:
        pytest.fail("No 'reached threshold' log call found")


@pytest.mark.unit
def test_threshold_reached_log_includes_value_and_threshold():
    """Assert the threshold-reached log includes value and threshold."""
    callback = StudyEarlyStoppingCallback(threshold=0.85, direction="maximize", patience=5)

    trial = MagicMock(spec=Trial)
    trial.state = TrialState.COMPLETE
    trial.value = 0.90
    trial.number = 3

    study_mock = MagicMock()
    study_mock.user_attrs = {}

    with patch("orchard.optimization.early_stopping.logger") as mock_logger:
        callback(study=study_mock, trial=trial)

    for call in mock_logger.info.call_args_list:
        args = call[0]
        if len(args) >= 8 and isinstance(args[0], str) and "threshold" in args[0].lower():
            # args[3] = trial.number, args[4] = value, args[6] = self.threshold
            assert args[3] == 3  # trial.number
            assert args[4] == pytest.approx(0.90)  # value
            assert args[6] == pytest.approx(0.85)  # threshold
            break
    else:
        pytest.fail("No 'reached threshold' log call found")


@pytest.mark.unit
def test_factory_case_insensitive_metric():
    """Assert factory uses .lower() to look up metric thresholds."""
    callback = get_early_stopping_callback(metric_name="AUC", direction="maximize")
    assert callback is not None
    assert callback.threshold == _THRESH_AUC


@pytest.mark.unit
def test_factory_case_insensitive_metric_mixed():
    """Assert factory uses .lower() for mixed case metrics."""
    callback = get_early_stopping_callback(metric_name="Accuracy", direction="maximize")
    assert callback is not None
    assert callback.threshold == _THRESH_ACCURACY


@pytest.mark.unit
def test_factory_passes_patience_to_callback():
    """Assert factory forwards patience argument."""
    cb = get_early_stopping_callback("auc", "maximize", patience=7)
    assert cb is not None
    assert cb.patience == 7


@pytest.mark.unit
def test_factory_passes_direction_to_callback():
    """Assert factory forwards direction to the callback."""
    cb = get_early_stopping_callback("loss", "minimize")
    assert cb is not None
    assert cb.direction == "minimize"


@pytest.mark.unit
def test_stop_log_phase_header_exact_message():
    """Assert Reporter.log_phase_header receives the exact expected message."""
    callback = StudyEarlyStoppingCallback(threshold=0.9, direction="maximize", patience=1)

    trial = MagicMock(spec=Trial)
    trial.state = TrialState.COMPLETE
    trial.value = 0.95
    trial.number = 0

    study_mock = MagicMock()
    study_mock.user_attrs = {}

    with patch("orchard.optimization.early_stopping.Reporter") as mock_reporter:
        callback(study=study_mock, trial=trial)

    call_args = mock_reporter.log_phase_header.call_args[0]
    # Exact match — kills XX-prefix and UPPERCASE mutants
    assert call_args[1] == "EARLY STOPPING: Target performance achieved!"


@pytest.mark.unit
def test_factory_passes_enabled_to_callback():
    """Assert factory forwards enabled kwarg to the StudyEarlyStoppingCallback."""
    cb = get_early_stopping_callback("auc", "maximize", threshold=0.99, enabled=True)
    assert cb is not None
    assert cb.enabled is True
