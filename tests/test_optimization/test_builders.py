"""
Unit tests for Optuna factory functions (samplers, pruners, callbacks).

Tests the builder functions that construct Optuna components from configuration
strings, ensuring proper error handling and component instantiation.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner, PercentilePruner
from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler

from orchard.optimization.orchestrator.builders import (
    build_callbacks,
    build_pruner,
    build_sampler,
)


@pytest.mark.unit
def test_build_sampler_tpe(mock_optuna_cfg: MagicMock) -> None:
    """Test building TPE sampler."""
    mock_optuna_cfg.sampler_type = "tpe"
    sampler = build_sampler(mock_optuna_cfg)
    assert isinstance(sampler, TPESampler)


@pytest.mark.unit
def test_build_sampler_random(mock_optuna_cfg: MagicMock) -> None:
    """Test building random sampler."""
    mock_optuna_cfg.sampler_type = "random"
    sampler = build_sampler(mock_optuna_cfg)
    assert isinstance(sampler, RandomSampler)


@pytest.mark.unit
def test_build_sampler_cmaes(mock_optuna_cfg: MagicMock) -> None:
    """Test building CMA-ES sampler."""
    mock_optuna_cfg.sampler_type = "cmaes"
    sampler = build_sampler(mock_optuna_cfg)
    assert isinstance(sampler, CmaEsSampler)


@pytest.mark.unit
def test_build_sampler_invalid(mock_optuna_cfg: MagicMock) -> None:
    """Test building sampler with invalid type raises ValueError."""
    mock_optuna_cfg.sampler_type = "invalid_sampler"

    with pytest.raises(ValueError) as exc_info:
        build_sampler(mock_optuna_cfg)

    assert "Unknown sampler: invalid_sampler" in str(exc_info.value)
    assert "Valid options:" in str(exc_info.value)


@pytest.mark.unit
def test_build_pruner_disabled(mock_optuna_cfg: MagicMock) -> None:
    """Test that pruner returns NopPruner when pruning is disabled."""
    mock_optuna_cfg.enable_pruning = False
    mock_optuna_cfg.pruner_type = "median"

    pruner = build_pruner(mock_optuna_cfg)
    assert isinstance(pruner, NopPruner)


@pytest.mark.unit
def test_build_pruner_median(mock_optuna_cfg: MagicMock) -> None:
    """Test building median pruner."""
    mock_optuna_cfg.enable_pruning = True
    mock_optuna_cfg.pruner_type = "median"

    pruner = build_pruner(mock_optuna_cfg)
    assert isinstance(pruner, MedianPruner)


@pytest.mark.unit
def test_build_pruner_percentile(mock_optuna_cfg: MagicMock) -> None:
    """Test building percentile pruner."""
    mock_optuna_cfg.enable_pruning = True
    mock_optuna_cfg.pruner_type = "percentile"

    pruner = build_pruner(mock_optuna_cfg)
    assert isinstance(pruner, PercentilePruner)


@pytest.mark.unit
def test_build_pruner_hyperband(mock_optuna_cfg: MagicMock) -> None:
    """Test building hyperband pruner."""
    mock_optuna_cfg.enable_pruning = True
    mock_optuna_cfg.pruner_type = "hyperband"

    pruner = build_pruner(mock_optuna_cfg)
    assert isinstance(pruner, HyperbandPruner)


@pytest.mark.unit
def test_build_pruner_invalid(mock_optuna_cfg: MagicMock) -> None:
    """Test building pruner with invalid type raises ValueError."""
    mock_optuna_cfg.enable_pruning = True
    mock_optuna_cfg.pruner_type = "invalid_pruner"

    with pytest.raises(ValueError) as exc_info:
        build_pruner(mock_optuna_cfg)

    assert "Unknown pruner: invalid_pruner" in str(exc_info.value)
    assert "Valid options:" in str(exc_info.value)


@pytest.mark.unit
def test_build_callbacks_with_early_stopping(mock_optuna_cfg: MagicMock) -> None:
    """Test building callbacks when early stopping is enabled."""
    mock_optuna_cfg.enable_early_stopping = True
    mock_optuna_cfg.early_stopping_threshold = 0.95
    mock_optuna_cfg.early_stopping_patience = 5
    callbacks = build_callbacks(mock_optuna_cfg, "auc", "maximize")

    assert len(callbacks) == 1
    assert callbacks[0] is not None


@pytest.mark.unit
def test_build_callbacks_without_early_stopping(mock_optuna_cfg: MagicMock) -> None:
    """Test building callbacks when early stopping is disabled."""
    mock_optuna_cfg.enable_early_stopping = False

    callbacks = build_callbacks(mock_optuna_cfg, "auc", "maximize")

    assert len(callbacks) == 0


# FIXTURE
@pytest.fixture
def mock_optuna_cfg() -> None:
    """Provide a mock OptunaConfig object for testing builders."""
    from unittest.mock import MagicMock

    optuna_mock = MagicMock()
    optuna_mock.sampler_type = "tpe"
    optuna_mock.enable_pruning = True
    optuna_mock.pruner_type = "median"
    optuna_mock.enable_early_stopping = False
    optuna_mock.early_stopping_threshold = 0.95
    optuna_mock.early_stopping_patience = 5

    return optuna_mock  # type: ignore


# ---------------------------------------------------------------------------
# Mutation-killing tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_sampler_returns_sampler_instance(mock_optuna_cfg: MagicMock) -> None:
    """Assert build_sampler returns an actual sampler object, not None."""
    mock_optuna_cfg.sampler_type = "tpe"
    result = build_sampler(mock_optuna_cfg)
    assert result is not None


@pytest.mark.unit
def test_build_sampler_invalid_includes_sampler_name(mock_optuna_cfg: MagicMock) -> None:
    """Assert error message includes the invalid sampler name."""
    mock_optuna_cfg.sampler_type = "bogus"
    with pytest.raises(ValueError, match="bogus"):
        build_sampler(mock_optuna_cfg)


@pytest.mark.unit
def test_build_pruner_returns_pruner_instance(mock_optuna_cfg: MagicMock) -> None:
    """Assert build_pruner returns a pruner, not None."""
    mock_optuna_cfg.enable_pruning = True
    mock_optuna_cfg.pruner_type = "median"
    result = build_pruner(mock_optuna_cfg)
    assert result is not None


@pytest.mark.unit
def test_build_pruner_disabled_returns_nop_not_none(mock_optuna_cfg: MagicMock) -> None:
    """Assert disabled pruning returns NopPruner specifically, not None."""
    mock_optuna_cfg.enable_pruning = False
    result = build_pruner(mock_optuna_cfg)
    assert isinstance(result, NopPruner)
    assert not isinstance(result, MedianPruner)


@pytest.mark.unit
def test_build_pruner_invalid_includes_pruner_name(mock_optuna_cfg: MagicMock) -> None:
    """Assert error message includes the invalid pruner name."""
    mock_optuna_cfg.enable_pruning = True
    mock_optuna_cfg.pruner_type = "bogus"
    with pytest.raises(ValueError, match="bogus"):
        build_pruner(mock_optuna_cfg)


@pytest.mark.unit
def test_build_callbacks_returns_list(mock_optuna_cfg: MagicMock) -> None:
    """Assert build_callbacks always returns a list."""
    mock_optuna_cfg.enable_early_stopping = False
    result = build_callbacks(mock_optuna_cfg, "auc", "maximize")
    assert isinstance(result, list)


@pytest.mark.unit
def test_build_callbacks_forwards_direction(mock_optuna_cfg: MagicMock) -> None:
    """Assert build_callbacks passes direction to early stopping factory."""
    from orchard.optimization.early_stopping import StudyEarlyStoppingCallback

    mock_optuna_cfg.enable_early_stopping = True
    mock_optuna_cfg.early_stopping_threshold = 0.99
    mock_optuna_cfg.early_stopping_patience = 3
    callbacks = build_callbacks(mock_optuna_cfg, "loss", "minimize")
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], StudyEarlyStoppingCallback)
    assert callbacks[0].direction == "minimize"


@pytest.mark.unit
def test_build_callbacks_forwards_threshold(mock_optuna_cfg: MagicMock) -> None:
    """Assert build_callbacks passes threshold to early stopping callback."""
    from orchard.optimization.early_stopping import StudyEarlyStoppingCallback

    mock_optuna_cfg.enable_early_stopping = True
    mock_optuna_cfg.early_stopping_threshold = 0.77
    mock_optuna_cfg.early_stopping_patience = 2

    callbacks = build_callbacks(mock_optuna_cfg, "auc", "maximize")
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], StudyEarlyStoppingCallback)
    assert callbacks[0].threshold == pytest.approx(0.77)


@pytest.mark.unit
def test_build_callbacks_forwards_patience(mock_optuna_cfg: MagicMock) -> None:
    """Assert build_callbacks passes patience to early stopping callback."""
    from orchard.optimization.early_stopping import StudyEarlyStoppingCallback

    mock_optuna_cfg.enable_early_stopping = True
    mock_optuna_cfg.early_stopping_threshold = 0.9
    mock_optuna_cfg.early_stopping_patience = 8

    callbacks = build_callbacks(mock_optuna_cfg, "auc", "maximize")
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], StudyEarlyStoppingCallback)
    assert callbacks[0].patience == 8


@pytest.mark.unit
def test_build_callbacks_forwards_monitor_metric(mock_optuna_cfg: MagicMock) -> None:
    """Assert build_callbacks passes monitor_metric to the factory."""
    from unittest.mock import patch as _patch

    mock_optuna_cfg.enable_early_stopping = True
    mock_optuna_cfg.early_stopping_threshold = 0.9
    mock_optuna_cfg.early_stopping_patience = 2

    with _patch(
        "orchard.optimization.orchestrator.builders.get_early_stopping_callback"
    ) as mock_factory:
        mock_factory.return_value = None
        build_callbacks(mock_optuna_cfg, "custom_metric", "maximize")

    call_kwargs = mock_factory.call_args[1]
    assert call_kwargs["metric_name"] == "custom_metric"


@pytest.mark.unit
def test_build_callbacks_empty_when_factory_returns_none(mock_optuna_cfg: MagicMock) -> None:
    """Assert empty list when early stopping factory returns None."""
    from unittest.mock import patch as _patch

    mock_optuna_cfg.enable_early_stopping = True

    with _patch(
        "orchard.optimization.orchestrator.builders.get_early_stopping_callback",
        return_value=None,
    ):
        callbacks = build_callbacks(mock_optuna_cfg, "auc", "maximize")

    assert callbacks == []
