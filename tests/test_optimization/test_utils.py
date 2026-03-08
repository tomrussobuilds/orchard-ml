"""
Unit tests for orchestrator utility functions.

Tests get_completed_trials and has_completed_trials from
orchard/optimization/orchestrator/utils.py.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import optuna
import pytest

from orchard.optimization.orchestrator.utils import (
    get_completed_trials,
    has_completed_trials,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


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
