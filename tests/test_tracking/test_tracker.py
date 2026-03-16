"""
Test Suite for Experiment Tracking Module.

Tests cover the create_tracker factory, NoOpTracker interface,
and tracker integration points in trainer, evaluation, and optimization.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from orchard.tracking import NoOpTracker, create_tracker
from orchard.tracking.tracker import _flatten_dict

# --- _flatten_dict TESTS ---


@pytest.mark.unit
def test_flatten_dict_empty() -> None:
    """_flatten_dict returns empty dict for empty input."""
    assert _flatten_dict({}) == {}


@pytest.mark.unit
def test_flatten_dict_flat() -> None:
    """_flatten_dict returns same keys for a flat dict."""
    d = {"a": 1, "b": "hello", "c": 3.14}
    assert _flatten_dict(d) == {"a": 1, "b": "hello", "c": 3.14}


@pytest.mark.unit
def test_flatten_dict_nested() -> None:
    """_flatten_dict flattens nested dicts with dot-separated keys."""
    d = {"training": {"lr": 0.01, "epochs": 10}, "model": "resnet"}
    result = _flatten_dict(d)
    assert result == {"training.lr": 0.01, "training.epochs": 10, "model": "resnet"}


@pytest.mark.unit
def test_flatten_dict_deeply_nested() -> None:
    """_flatten_dict handles multiple nesting levels."""
    d = {"a": {"b": {"c": 42}}}
    assert _flatten_dict(d) == {"a.b.c": 42}


@pytest.mark.unit
def test_flatten_dict_custom_separator() -> None:
    """_flatten_dict respects custom separator."""
    d = {"a": {"b": 1}}
    assert _flatten_dict(d, sep="/") == {"a/b": 1}


@pytest.mark.unit
def test_flatten_dict_default_parent_key_is_empty() -> None:
    """Kill mutant: parent_key default must be empty string, not arbitrary."""
    d = {"x": 1}
    result = _flatten_dict(d)
    assert result == {"x": 1}


@pytest.mark.unit
def test_flatten_dict_default_sep_is_dot() -> None:
    """Kill mutant: sep default must be exactly '.'."""
    d = {"a": {"b": 1}}
    result = _flatten_dict(d)
    assert result == {"a.b": 1}


# --- MLFLOW TRACKER INIT TESTS ---


@pytest.mark.unit
def test_mlflow_tracker_default_experiment_name_exact() -> None:
    """Kill mutant: default experiment_name must be exactly 'orchard-ml'."""
    from unittest.mock import patch

    from orchard.tracking.tracker import MLflowTracker

    with patch("orchard.tracking.tracker._MLFLOW_AVAILABLE", True):
        tracker = MLflowTracker()

    assert tracker.experiment_name == "orchard-ml"
    assert len(tracker.experiment_name) == len("orchard-ml")


@pytest.mark.unit
def test_mlflow_tracker_parent_run_id_init_none() -> None:
    """Kill mutant: _parent_run_id must init to None, not empty string."""
    from unittest.mock import patch

    from orchard.tracking.tracker import MLflowTracker

    with patch("orchard.tracking.tracker._MLFLOW_AVAILABLE", True):
        tracker = MLflowTracker()

    assert tracker._parent_run_id is None


# --- FACTORY TESTS ---


@pytest.mark.unit
def test_create_tracker_no_tracking_config() -> None:
    """create_tracker returns NoOpTracker when cfg has no tracking attribute."""
    cfg = MagicMock(spec=[])
    tracker = create_tracker(cfg)
    assert isinstance(tracker, NoOpTracker)


@pytest.mark.unit
def test_create_tracker_tracking_none() -> None:
    """create_tracker returns NoOpTracker when cfg.tracking is None."""
    cfg = MagicMock()
    cfg.tracking = None
    tracker = create_tracker(cfg)
    assert isinstance(tracker, NoOpTracker)


@pytest.mark.unit
def test_create_tracker_tracking_disabled() -> None:
    """create_tracker returns NoOpTracker when tracking is disabled."""
    cfg = MagicMock()
    cfg.tracking.enabled = False
    tracker = create_tracker(cfg)
    assert isinstance(tracker, NoOpTracker)


# --- NOOP TRACKER TESTS ---


@pytest.mark.unit
def test_noop_tracker_start_run() -> None:
    """NoOpTracker.start_run completes without error."""
    tracker = NoOpTracker()
    tracker.start_run(cfg=MagicMock(), run_name="test", tracking_uri="file:///mock")


@pytest.mark.unit
def test_noop_tracker_log_epoch() -> None:
    """NoOpTracker.log_epoch completes without error."""
    tracker = NoOpTracker()
    tracker.log_epoch(epoch=1, train_loss=0.5, val_metrics={"loss": 0.3}, lr=0.01)


@pytest.mark.unit
def test_noop_tracker_log_test_metrics() -> None:
    """NoOpTracker.log_test_metrics completes without error."""
    tracker = NoOpTracker()
    tracker.log_test_metrics({"accuracy": 0.95, "f1": 0.90})


@pytest.mark.unit
def test_noop_tracker_log_artifact(tmp_path: Path) -> None:
    """NoOpTracker.log_artifact completes without error."""
    tracker = NoOpTracker()
    tracker.log_artifact(tmp_path / "fake.txt")


@pytest.mark.unit
def test_noop_tracker_log_artifacts_dir(tmp_path: Path) -> None:
    """NoOpTracker.log_artifacts_dir completes without error."""
    tracker = NoOpTracker()
    tracker.log_artifacts_dir(tmp_path)


@pytest.mark.unit
def test_noop_tracker_optuna_trial() -> None:
    """NoOpTracker nested trial methods complete without error."""
    tracker = NoOpTracker()
    tracker.start_optuna_trial(trial_number=0, params={"lr": 0.01})
    tracker.end_optuna_trial(best_metric=0.95)


@pytest.mark.unit
def test_noop_tracker_end_run() -> None:
    """NoOpTracker.end_run completes without error."""
    tracker = NoOpTracker()
    tracker.end_run()


@pytest.mark.unit
def test_create_tracker_mlflow_not_installed() -> None:
    """create_tracker returns NoOpTracker with warning when mlflow is missing."""
    from unittest.mock import patch

    cfg = MagicMock()
    cfg.tracking.enabled = True
    cfg.tracking.experiment_name = "test"

    with patch("orchard.tracking.tracker._MLFLOW_AVAILABLE", False):
        tracker = create_tracker(cfg)

    assert isinstance(tracker, NoOpTracker)


@pytest.mark.unit
def test_create_tracker_mlflow_available() -> None:
    """create_tracker returns MLflowTracker when mlflow is available."""
    from unittest.mock import patch

    from orchard.tracking.tracker import MLflowTracker

    cfg = MagicMock()
    cfg.tracking.enabled = True
    cfg.tracking.experiment_name = "my_experiment"

    with patch("orchard.tracking.tracker._MLFLOW_AVAILABLE", True):
        tracker = create_tracker(cfg)

    assert isinstance(tracker, MLflowTracker)
    assert tracker.experiment_name == "my_experiment"
