"""
Task Strategy Packages.

Each sub-package exports its adapter classes. Registration in the
core task registry is handled by :mod:`orchard` (the top-level init),
which is the natural junction point between ``core`` and ``tasks``.
"""

from __future__ import annotations

from .classification import (
    ClassificationCriterionAdapter,
    ClassificationEvalPipelineAdapter,
    ClassificationMetricsAdapter,
)

__all__ = [
    "ClassificationCriterionAdapter",
    "ClassificationEvalPipelineAdapter",
    "ClassificationMetricsAdapter",
]
