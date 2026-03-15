"""
Classification Validation Metrics Adapter.

Wraps :func:`orchard.trainer.engine.validate_epoch` to satisfy
:class:`~orchard.core.task_protocols.TaskValidationMetrics`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ...trainer.engine import validate_epoch

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping


class ClassificationMetricsAdapter:
    """Computes per-epoch classification metrics (loss, accuracy, AUC, F1)."""

    def compute_validation_metrics(
        self,
        model: nn.Module,
        val_loader: DataLoader[Any],
        criterion: nn.Module,
        device: torch.device,
    ) -> Mapping[str, float]:
        """
        Delegate to the existing validation engine.

        Args:
            model: Neural network model to evaluate.
            val_loader: Validation data provider.
            criterion: Loss function.
            device: Hardware target.

        Returns:
            Immutable mapping with keys: loss, accuracy, auc, f1.
        """
        return validate_epoch(model, val_loader, criterion, device)
