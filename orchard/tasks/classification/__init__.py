"""
Classification Task Adapters.

Exports the three classification strategy adapters. Registration in the
task registry is handled by :mod:`orchard.tasks`, which owns the
relationship between adapters and ``core.task_registry``.
"""

from __future__ import annotations

from .criterion_adapter import ClassificationCriterionAdapter
from .evaluation_adapter import ClassificationEvalPipelineAdapter
from .metrics_adapter import ClassificationMetricsAdapter

__all__ = [
    "ClassificationCriterionAdapter",
    "ClassificationEvalPipelineAdapter",
    "ClassificationMetricsAdapter",
]
