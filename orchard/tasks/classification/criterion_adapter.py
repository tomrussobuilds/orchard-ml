"""
Classification Criterion Adapter.

Wraps :func:`orchard.trainer.setup.get_criterion` to satisfy
:class:`~orchard.core.task_protocols.TaskCriterionFactory`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from ...trainer.setup import get_criterion

if TYPE_CHECKING:  # pragma: no cover
    from ...core.config import TrainingConfig


class ClassificationCriterionAdapter:
    """Builds classification loss functions (CrossEntropy / Focal)."""

    def get_criterion(
        self,
        training: TrainingConfig,
        class_weights: torch.Tensor | None = None,
    ) -> nn.Module:
        """
        Delegate to the existing criterion factory.

        Args:
            training: Training sub-config with criterion parameters.
            class_weights: Optional per-class weights for imbalanced datasets.

        Returns:
            Loss module (CrossEntropyLoss or FocalLoss).
        """
        return get_criterion(training, class_weights=class_weights)
