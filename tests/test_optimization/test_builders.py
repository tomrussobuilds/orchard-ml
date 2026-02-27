"""
Unit tests for Optuna factory functions (samplers, pruners, callbacks).

Tests the builder functions that construct Optuna components from configuration
strings, ensuring proper error handling and component instantiation.
"""

from __future__ import annotations

import pytest
from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner, PercentilePruner
from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler

from orchard.optimization.orchestrator.builders import (
    build_callbacks,
    build_pruner,
    build_sampler,
)


@pytest.mark.unit
def test_build_sampler_tpe(mock_optuna_cfg):
    """Test building TPE sampler."""
    mock_optuna_cfg.sampler_type = "tpe"
    sampler = build_sampler(mock_optuna_cfg)
    assert isinstance(sampler, TPESampler)


@pytest.mark.unit
def test_build_sampler_random(mock_optuna_cfg):
    """Test building random sampler."""
    mock_optuna_cfg.sampler_type = "random"
    sampler = build_sampler(mock_optuna_cfg)
    assert isinstance(sampler, RandomSampler)


@pytest.mark.unit
def test_build_sampler_cmaes(mock_optuna_cfg):
    """Test building CMA-ES sampler."""
    mock_optuna_cfg.sampler_type = "cmaes"
    sampler = build_sampler(mock_optuna_cfg)
    assert isinstance(sampler, CmaEsSampler)


@pytest.mark.unit
def test_build_sampler_invalid(mock_optuna_cfg):
    """Test building sampler with invalid type raises ValueError."""
    mock_optuna_cfg.sampler_type = "invalid_sampler"

    with pytest.raises(ValueError) as exc_info:
        build_sampler(mock_optuna_cfg)

    assert "Unknown sampler: invalid_sampler" in str(exc_info.value)
    assert "Valid options:" in str(exc_info.value)


@pytest.mark.unit
def test_build_pruner_disabled(mock_optuna_cfg):
    """Test that pruner returns NopPruner when pruning is disabled."""
    mock_optuna_cfg.enable_pruning = False
    mock_optuna_cfg.pruner_type = "median"

    pruner = build_pruner(mock_optuna_cfg)
    assert isinstance(pruner, NopPruner)


@pytest.mark.unit
def test_build_pruner_median(mock_optuna_cfg):
    """Test building median pruner."""
    mock_optuna_cfg.enable_pruning = True
    mock_optuna_cfg.pruner_type = "median"

    pruner = build_pruner(mock_optuna_cfg)
    assert isinstance(pruner, MedianPruner)


@pytest.mark.unit
def test_build_pruner_percentile(mock_optuna_cfg):
    """Test building percentile pruner."""
    mock_optuna_cfg.enable_pruning = True
    mock_optuna_cfg.pruner_type = "percentile"

    pruner = build_pruner(mock_optuna_cfg)
    assert isinstance(pruner, PercentilePruner)


@pytest.mark.unit
def test_build_pruner_hyperband(mock_optuna_cfg):
    """Test building hyperband pruner."""
    mock_optuna_cfg.enable_pruning = True
    mock_optuna_cfg.pruner_type = "hyperband"

    pruner = build_pruner(mock_optuna_cfg)
    assert isinstance(pruner, HyperbandPruner)


@pytest.mark.unit
def test_build_pruner_invalid(mock_optuna_cfg):
    """Test building pruner with invalid type raises ValueError."""
    mock_optuna_cfg.enable_pruning = True
    mock_optuna_cfg.pruner_type = "invalid_pruner"

    with pytest.raises(ValueError) as exc_info:
        build_pruner(mock_optuna_cfg)

    assert "Unknown pruner: invalid_pruner" in str(exc_info.value)
    assert "Valid options:" in str(exc_info.value)


@pytest.mark.unit
def test_build_callbacks_with_early_stopping(mock_optuna_cfg):
    """Test building callbacks when early stopping is enabled."""
    mock_optuna_cfg.enable_early_stopping = True
    mock_optuna_cfg.early_stopping_threshold = 0.95
    mock_optuna_cfg.early_stopping_patience = 5
    mock_optuna_cfg.direction = "maximize"
    callbacks = build_callbacks(mock_optuna_cfg, "auc")

    assert len(callbacks) == 1
    assert callbacks[0] is not None


@pytest.mark.unit
def test_build_callbacks_without_early_stopping(mock_optuna_cfg):
    """Test building callbacks when early stopping is disabled."""
    mock_optuna_cfg.enable_early_stopping = False

    callbacks = build_callbacks(mock_optuna_cfg, "auc")

    assert len(callbacks) == 0


# FIXTURE
@pytest.fixture
def mock_optuna_cfg():
    """Provide a mock OptunaConfig object for testing builders."""
    from unittest.mock import MagicMock

    optuna_mock = MagicMock()
    optuna_mock.sampler_type = "tpe"
    optuna_mock.enable_pruning = True
    optuna_mock.pruner_type = "median"
    optuna_mock.enable_early_stopping = False
    optuna_mock.early_stopping_threshold = 0.95
    optuna_mock.early_stopping_patience = 5
    optuna_mock.direction = "maximize"

    return optuna_mock
