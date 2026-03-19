"""
Detection Criterion Adapter.

Detection models (e.g. Faster R-CNN) compute losses internally during
the forward pass — no external criterion is needed. This adapter returns
a sentinel module that raises on accidental use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:  # pragma: no cover
    from ...core.config import TrainingConfig


class _DetectionNoOpCriterion(nn.Module):
    """Sentinel criterion that fails fast if accidentally called."""

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:  # noqa: ARG002
        raise RuntimeError(
            "Detection models compute losses internally. "  # pragma: no mutate
            "The criterion should never be called in the detection pipeline."  # pragma: no mutate
        )


class DetectionCriterionAdapter:
    """Returns a no-op sentinel criterion for detection tasks."""

    def get_criterion(
        self,
        training: TrainingConfig,  # noqa: ARG002
        class_weights: torch.Tensor | None = None,  # noqa: ARG002
    ) -> nn.Module:
        """
        Return a sentinel criterion.

        Detection models compute their own losses internally (classification
        loss, box regression loss, objectness, RPN box reg). The returned
        module raises ``RuntimeError`` if its ``forward()`` is ever called,
        making misuse immediately visible.

        Args:
            training: Training sub-config (ignored for detection).
            class_weights: Per-class weights (ignored for detection).

        Returns:
            Sentinel ``nn.Module`` that raises on forward.
        """
        return _DetectionNoOpCriterion()
