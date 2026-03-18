"""
Task Component Registry.

Maps ``task_type`` strings to frozen bundles of task-specific strategy
implementations. Each bundle (:class:`TaskComponents`) carries the three
pluggable dimensions defined in :mod:`orchard.core.task_protocols`:

- criterion factory
- validation metrics
- eval pipeline (inference + visualization + reporting)

The registry is populated at import time by ``orchard.tasks.<task_type>``
packages and exposed as an immutable :class:`~types.MappingProxyType` via
:func:`get_registry`.

Usage::

    from orchard.core.task_registry import get_task

    task = get_task("classification")
    criterion = task.criterion_factory.get_criterion(training_cfg)
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from ..exceptions import OrchardConfigError

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from .task_protocols import (
        TaskCriterionFactory,
        TaskEvalPipeline,
        TaskTrainingStep,
        TaskValidationMetrics,
    )


@dataclass(frozen=True)
class TaskComponents:
    """
    Immutable bundle of task-specific strategy implementations.

    Attributes:
        criterion_factory: Builds the loss function for this task.
        training_step: Executes the forward pass and computes training loss.
        validation_metrics: Computes per-epoch validation metrics.
        eval_pipeline: Orchestrates inference, visualization, and reporting.
        fallback_metrics: Metrics returned when validation fails during Optuna
            trials. Must contain at least the monitored metric key.
    """

    criterion_factory: TaskCriterionFactory
    training_step: TaskTrainingStep
    validation_metrics: TaskValidationMetrics
    eval_pipeline: TaskEvalPipeline
    fallback_metrics: Mapping[str, float]


# Internal mutable store — never exposed directly.
_TASK_REGISTRY: dict[str, TaskComponents] = {}


def register_task(task_type: str, components: TaskComponents) -> None:
    """
    Register a task-specific component bundle.

    Args:
        task_type: Identifier matching a ``TaskType`` literal
            (e.g. ``"classification"``).
        components: Frozen bundle of strategy implementations.

    Raises:
        OrchardConfigError: If ``task_type`` is already registered.
    """
    if task_type in _TASK_REGISTRY:
        raise OrchardConfigError(
            f"Task '{task_type}' is already registered. "
            "Duplicate registration indicates a plugin conflict."
        )
    _TASK_REGISTRY[task_type] = components


def get_task(task_type: str) -> TaskComponents:
    """
    Retrieve the component bundle for a given task type.

    Args:
        task_type: Registered task identifier.

    Returns:
        The corresponding :class:`TaskComponents` bundle.

    Raises:
        OrchardConfigError: If ``task_type`` has not been registered.
    """
    if task_type not in _TASK_REGISTRY:
        available = sorted(_TASK_REGISTRY.keys())
        raise OrchardConfigError(f"Unknown task_type '{task_type}'. Registered: {available}")
    return _TASK_REGISTRY[task_type]


def get_registry() -> MappingProxyType[str, TaskComponents]:
    """
    Return an immutable view of the full task registry.

    Returns:
        Read-only mapping of task type strings to component bundles.
    """
    return MappingProxyType(_TASK_REGISTRY)
