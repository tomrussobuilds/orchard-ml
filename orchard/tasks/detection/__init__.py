"""
Detection Task Adapters.

Provides task-specific adapters for object detection, implementing
the protocols defined in :mod:`orchard.core.task_protocols`.
"""

from __future__ import annotations

from .criterion_adapter import DetectionCriterionAdapter
from .evaluation_adapter import DetectionEvalPipelineAdapter
from .metrics_adapter import DetectionMetricsAdapter
from .training_step_adapter import DetectionTrainingStepAdapter

__all__ = [
    "DetectionCriterionAdapter",
    "DetectionEvalPipelineAdapter",
    "DetectionMetricsAdapter",
    "DetectionTrainingStepAdapter",
]
