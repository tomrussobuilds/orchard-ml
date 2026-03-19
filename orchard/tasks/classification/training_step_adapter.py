"""
Classification Training Step Adapter.

Wraps the standard classification forward pass (logits + criterion) to
satisfy :class:`~orchard.core.task_protocols.TaskTrainingStep`.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn


class ClassificationTrainingStepAdapter:
    """Computes classification training loss with optional MixUp blending."""

    def compute_training_loss(
        self,
        model: nn.Module,
        inputs: Any,
        targets: Any,
        criterion: nn.Module,
        mixup_fn: Callable[..., Any] | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Execute classification forward pass and compute loss.

        When ``mixup_fn`` is provided, inputs and targets are blended
        before the forward pass and the loss is computed as a convex
        combination of the two target sets.

        Args:
            model: Neural network producing logits.
            inputs: Batch of input tensors.
            targets: Batch of target tensors.
            criterion: Loss function (e.g. CrossEntropyLoss).
            mixup_fn: Optional MixUp augmentation callable.
            device: Target device for tensor placement.

        Returns:
            Scalar loss tensor for backward pass.
        """
        if device is not None:
            inputs = inputs.to(device)
            targets = targets.to(device)
        if mixup_fn is not None:
            inputs, y_a, y_b, lam = mixup_fn(inputs, targets)
            outputs = model(inputs)
            loss: torch.Tensor = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            return loss
        outputs = model(inputs)
        result: torch.Tensor = criterion(outputs, targets)
        return result
