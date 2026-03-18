"""
Test Suite for Task Registry and Task Protocols.

Tests for register_task, get_task, get_registry, TaskComponents,
and runtime_checkable protocol compliance.
"""

from __future__ import annotations

from types import MappingProxyType
from unittest.mock import MagicMock

import pytest

from orchard.core.task_protocols import (
    TaskCriterionFactory,
    TaskEvalPipeline,
    TaskTrainingStep,
    TaskValidationMetrics,
)
from orchard.core.task_registry import (
    _TASK_REGISTRY,
    TaskComponents,
    get_registry,
    get_task,
    register_task,
)
from orchard.exceptions import OrchardConfigError

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_registry() -> None:  # type: ignore
    """Ensure each test starts with a clean registry and restores it after."""
    saved = dict(_TASK_REGISTRY)
    _TASK_REGISTRY.clear()
    yield
    _TASK_REGISTRY.clear()
    _TASK_REGISTRY.update(saved)


def _make_components() -> TaskComponents:
    """Create a TaskComponents bundle with MagicMock adapters."""
    return TaskComponents(
        criterion_factory=MagicMock(spec=TaskCriterionFactory),
        training_step=MagicMock(spec=TaskTrainingStep),
        validation_metrics=MagicMock(spec=TaskValidationMetrics),
        eval_pipeline=MagicMock(spec=TaskEvalPipeline),
        fallback_metrics={"loss": 999.0},
        early_stopping_thresholds={"accuracy": 0.995},
    )


# ── register_task ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_register_task_stores_components() -> None:
    """register_task adds components to the internal registry."""
    components = _make_components()
    register_task("test_task", components)

    assert "test_task" in _TASK_REGISTRY
    assert _TASK_REGISTRY["test_task"] is components


@pytest.mark.unit
def test_register_task_duplicate_raises() -> None:
    """register_task raises OrchardConfigError on duplicate registration."""
    register_task("dup", _make_components())

    with pytest.raises(OrchardConfigError, match=r"already registered.*\bDuplicate registration"):
        register_task("dup", _make_components())


@pytest.mark.unit
def test_register_task_error_mentions_task_name() -> None:
    """Duplicate registration error message includes the task name."""
    register_task("segmentation", _make_components())

    with pytest.raises(OrchardConfigError, match="segmentation"):
        register_task("segmentation", _make_components())


# ── get_task ──────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_get_task_returns_registered_components() -> None:
    """get_task returns the exact components bundle that was registered."""
    components = _make_components()
    register_task("cls", components)

    result = get_task("cls")

    assert result is components


@pytest.mark.unit
def test_get_task_unknown_raises() -> None:
    """get_task raises OrchardConfigError for unregistered task types."""
    with pytest.raises(OrchardConfigError, match="Unknown task_type"):
        get_task("nonexistent")


@pytest.mark.unit
def test_get_task_error_lists_available() -> None:
    """get_task error message lists registered task types."""
    register_task("alpha", _make_components())
    register_task("beta", _make_components())

    with pytest.raises(OrchardConfigError, match=r"\['alpha', 'beta'\]"):
        get_task("gamma")


@pytest.mark.unit
def test_get_task_error_empty_registry() -> None:
    """get_task error on empty registry shows empty list."""
    with pytest.raises(OrchardConfigError, match=r"\[\]"):
        get_task("anything")


# ── get_registry ──────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_get_registry_returns_mapping_proxy() -> None:
    """get_registry returns a MappingProxyType."""
    register_task("x", _make_components())

    result = get_registry()

    assert isinstance(result, MappingProxyType)


@pytest.mark.unit
def test_get_registry_is_immutable() -> None:
    """Returned registry proxy rejects mutation."""
    register_task("x", _make_components())
    registry = get_registry()

    with pytest.raises(TypeError):
        registry["y"] = _make_components()  # type: ignore


@pytest.mark.unit
def test_get_registry_reflects_registrations() -> None:
    """get_registry includes all registered tasks."""
    c1 = _make_components()
    c2 = _make_components()
    register_task("a", c1)
    register_task("b", c2)

    registry = get_registry()

    assert registry["a"] is c1
    assert registry["b"] is c2
    assert len(registry) == 2


@pytest.mark.unit
def test_get_registry_empty() -> None:
    """get_registry returns empty proxy when no tasks are registered."""
    assert len(get_registry()) == 0


# ── TaskComponents frozen ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_task_components_frozen() -> None:
    """TaskComponents is frozen — attribute assignment raises."""
    components = _make_components()

    with pytest.raises(AttributeError):
        components.criterion_factory = MagicMock()  # type: ignore


# ── Protocol isinstance checks ────────────────────────────────────────────────


@pytest.mark.unit
def test_classification_criterion_adapter_satisfies_protocol() -> None:
    """ClassificationCriterionAdapter passes isinstance check."""
    from orchard.tasks.classification.criterion_adapter import (
        ClassificationCriterionAdapter,
    )

    adapter = ClassificationCriterionAdapter()

    assert isinstance(adapter, TaskCriterionFactory)


@pytest.mark.unit
def test_classification_metrics_adapter_satisfies_protocol() -> None:
    """ClassificationMetricsAdapter passes isinstance check."""
    from orchard.tasks.classification.metrics_adapter import (
        ClassificationMetricsAdapter,
    )

    adapter = ClassificationMetricsAdapter()

    assert isinstance(adapter, TaskValidationMetrics)


@pytest.mark.unit
def test_classification_eval_adapter_satisfies_protocol() -> None:
    """ClassificationEvalPipelineAdapter passes isinstance check."""
    from orchard.tasks.classification.evaluation_adapter import (
        ClassificationEvalPipelineAdapter,
    )

    adapter = ClassificationEvalPipelineAdapter()

    assert isinstance(adapter, TaskEvalPipeline)


@pytest.mark.unit
def test_classification_training_step_adapter_satisfies_protocol() -> None:
    """ClassificationTrainingStepAdapter passes isinstance check."""
    from orchard.tasks.classification.training_step_adapter import (
        ClassificationTrainingStepAdapter,
    )

    adapter = ClassificationTrainingStepAdapter()

    assert isinstance(adapter, TaskTrainingStep)


@pytest.mark.unit
def test_object_without_method_fails_protocol() -> None:
    """An object missing the required method does not satisfy TaskCriterionFactory."""

    class _Empty:
        pass

    assert not isinstance(_Empty(), TaskCriterionFactory)


# ── Integration: classification registration ──────────────────────────────────


@pytest.mark.unit
def test_classification_registered_on_import() -> None:
    """Importing orchard.tasks registers the 'classification' task."""
    register_task("classification", _make_components())

    task = get_task("classification")

    assert task.criterion_factory is not None
    assert task.training_step is not None
    assert task.validation_metrics is not None
    assert task.eval_pipeline is not None
