"""
Detection Validation Metrics Adapter.

Computes mAP-family metrics using ``torchmetrics.detection.MeanAveragePrecision``
to satisfy :class:`~orchard.core.task_protocols.TaskValidationMetrics`.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

from ...core.paths import METRIC_LOSS, METRIC_MAP, METRIC_MAP_50, METRIC_MAP_75


# TODO(cleanup): Extract to shared _utils.py — duplicated in evaluation_adapter.py
def _to_cpu(d: dict[str, Any]) -> dict[str, Any]:
    """Move all tensor values in a dict to CPU."""
    return {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in d.items()}


class DetectionMetricsAdapter:
    """Computes mAP validation metrics for object detection."""

    def compute_validation_metrics(
        self,
        model: nn.Module,
        val_loader: DataLoader[Any],
        criterion: nn.Module,  # noqa: ARG002
        device: torch.device,
    ) -> Mapping[str, float]:
        """
        Run detection inference and compute mAP metrics.

        Iterates the validation loader, collects predictions and targets,
        then computes mean Average Precision at multiple IoU thresholds.

        Detection models do not produce a single validation loss in eval
        mode, so ``"loss"`` is returned as ``0.0``.

        Args:
            model: Detection model to evaluate.
            val_loader: Validation data provider.
            criterion: Ignored (detection models compute losses internally).
            device: Hardware target for inference.

        Returns:
            Immutable mapping with keys: ``loss``, ``map``, ``map_50``, ``map_75``.
        """
        model.eval()
        metric = MeanAveragePrecision(iou_type="bbox")

        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                predictions = model(images)
                metric.update(
                    [_to_cpu(p) for p in predictions],
                    [_to_cpu(t) for t in targets],
                )

        result = metric.compute()

        return MappingProxyType(
            {
                METRIC_LOSS: 0.0,
                METRIC_MAP: float(result["map"]),
                METRIC_MAP_50: float(result["map_50"]),
                METRIC_MAP_75: float(result["map_75"]),
            }
        )
