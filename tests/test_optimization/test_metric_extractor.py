"""
Mutation-killing tests for MetricExtractor.

Targets surviving mutants in orchard/optimization/objective/metric_extractor.py
by asserting exact attribute values, direction-aware behavior, NaN handling,
and error messages.
"""

from __future__ import annotations

import math

import pytest

from orchard.optimization.objective.metric_extractor import MetricExtractor

# ---------------------------------------------------------------------------
# __init__ attribute storage
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_init_stores_metric_name():
    """Assert metric_name is stored exactly."""
    ext = MetricExtractor(metric_name="custom")
    assert ext.metric_name == "custom"


@pytest.mark.unit
def test_init_stores_direction():
    """Assert direction is stored exactly."""
    ext = MetricExtractor(metric_name="x", direction="minimize")
    assert ext.direction == "minimize"


@pytest.mark.unit
def test_init_default_direction_is_maximize():
    """Assert default direction is 'maximize'."""
    ext = MetricExtractor(metric_name="x")
    assert ext.direction == "maximize"


@pytest.mark.unit
def test_init_is_maximize_true():
    """Assert _is_maximize is True for 'maximize' direction."""
    ext = MetricExtractor(metric_name="x", direction="maximize")
    assert ext._is_maximize is True


@pytest.mark.unit
def test_init_is_maximize_false():
    """Assert _is_maximize is False for 'minimize' direction."""
    ext = MetricExtractor(metric_name="x", direction="minimize")
    assert ext._is_maximize is False


@pytest.mark.unit
def test_init_best_metric_maximize():
    """Assert best_metric starts at -inf for maximize."""
    ext = MetricExtractor(metric_name="x", direction="maximize")
    assert ext.best_metric == -float("inf")
    assert math.isinf(ext.best_metric)
    assert ext.best_metric < 0


@pytest.mark.unit
def test_init_best_metric_minimize():
    """Assert best_metric starts at +inf for minimize."""
    ext = MetricExtractor(metric_name="x", direction="minimize")
    assert ext.best_metric == float("inf")
    assert math.isinf(ext.best_metric)
    assert ext.best_metric > 0


# ---------------------------------------------------------------------------
# extract()
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extract_returns_exact_value():
    """Assert extract() returns the exact metric value."""
    ext = MetricExtractor(metric_name="auc")
    result = ext.extract({"auc": 0.876, "loss": 0.3})
    assert result == pytest.approx(0.876)


@pytest.mark.unit
def test_extract_missing_metric_raises_key_error():
    """Assert KeyError is raised for missing metric."""
    ext = MetricExtractor(metric_name="f1")
    with pytest.raises(KeyError, match="f1"):
        ext.extract({"loss": 0.5, "accuracy": 0.9})


@pytest.mark.unit
def test_extract_error_message_includes_available():
    """Assert error message includes list of available metric names."""
    ext = MetricExtractor(metric_name="f1")
    with pytest.raises(KeyError) as exc_info:
        ext.extract({"loss": 0.5, "accuracy": 0.9})
    msg = str(exc_info.value)
    assert "Available" in msg
    # Verify the actual keys are listed (kills available=None mutant)
    assert "loss" in msg
    assert "accuracy" in msg


@pytest.mark.unit
def test_extract_error_message_includes_metric_name():
    """Assert error message includes the missing metric name."""
    ext = MetricExtractor(metric_name="recall")
    with pytest.raises(KeyError, match="recall"):
        ext.extract({"precision": 0.8})


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_reset_maximize():
    """Assert reset() restores best_metric to -inf for maximize."""
    ext = MetricExtractor(metric_name="auc", direction="maximize")
    ext.update_best(0.95)
    ext.reset()
    assert ext.best_metric == -float("inf")


@pytest.mark.unit
def test_reset_minimize():
    """Assert reset() restores best_metric to +inf for minimize."""
    ext = MetricExtractor(metric_name="loss", direction="minimize")
    ext.update_best(0.1)
    ext.reset()
    assert ext.best_metric == float("inf")


# ---------------------------------------------------------------------------
# update_best() — maximize
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_update_best_maximize_improvement():
    """Assert higher values replace best in maximize mode."""
    ext = MetricExtractor(metric_name="auc", direction="maximize")
    result = ext.update_best(0.8)
    assert result == pytest.approx(0.8)
    assert ext.best_metric == pytest.approx(0.8)

    result = ext.update_best(0.9)
    assert result == pytest.approx(0.9)
    assert ext.best_metric == pytest.approx(0.9)


@pytest.mark.unit
def test_update_best_maximize_no_improvement():
    """Assert lower values do NOT replace best in maximize mode."""
    ext = MetricExtractor(metric_name="auc", direction="maximize")
    ext.update_best(0.9)
    result = ext.update_best(0.8)
    assert result == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# update_best() — minimize
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_update_best_minimize_improvement():
    """Assert lower values replace best in minimize mode."""
    ext = MetricExtractor(metric_name="loss", direction="minimize")
    result = ext.update_best(0.5)
    assert result == pytest.approx(0.5)

    result = ext.update_best(0.3)
    assert result == pytest.approx(0.3)


@pytest.mark.unit
def test_update_best_minimize_no_improvement():
    """Assert higher values do NOT replace best in minimize mode."""
    ext = MetricExtractor(metric_name="loss", direction="minimize")
    ext.update_best(0.3)
    result = ext.update_best(0.5)
    assert result == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# update_best() — NaN handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_update_best_nan_ignored_maximize():
    """Assert NaN does not poison best_metric in maximize mode."""
    ext = MetricExtractor(metric_name="auc", direction="maximize")
    ext.update_best(0.85)
    result = ext.update_best(float("nan"))
    assert result == pytest.approx(0.85)
    assert ext.best_metric == pytest.approx(0.85)


@pytest.mark.unit
def test_update_best_nan_ignored_minimize():
    """Assert NaN does not poison best_metric in minimize mode."""
    ext = MetricExtractor(metric_name="loss", direction="minimize")
    ext.update_best(0.2)
    result = ext.update_best(float("nan"))
    assert result == pytest.approx(0.2)


@pytest.mark.unit
def test_update_best_nan_as_first_value():
    """Assert NaN as first value does not update best_metric."""
    ext = MetricExtractor(metric_name="auc", direction="maximize")
    result = ext.update_best(float("nan"))
    assert result == -float("inf")
    assert ext.best_metric == -float("inf")


# ---------------------------------------------------------------------------
# update_best() — return value
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_update_best_returns_best_not_current():
    """Assert update_best returns best, not the current metric value."""
    ext = MetricExtractor(metric_name="auc", direction="maximize")
    ext.update_best(0.9)
    # Worse value: should return best (0.9), not current (0.7)
    result = ext.update_best(0.7)
    assert result == pytest.approx(0.9)


@pytest.mark.unit
def test_update_best_equal_value():
    """Assert equal value does not change best_metric."""
    ext = MetricExtractor(metric_name="auc", direction="maximize")
    ext.update_best(0.9)
    result = ext.update_best(0.9)
    assert result == pytest.approx(0.9)
