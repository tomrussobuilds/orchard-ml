"""
Detection Training Step Adapter.

Wraps the detection forward pass (model returns loss dict) to satisfy
:class:`~orchard.core.task_protocols.TaskTrainingStep`.

Detection models like Faster R-CNN expect ``list[Tensor]`` images and
``list[dict]`` targets in training mode, and return a ``dict`` of losses
that must be summed for backpropagation.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn


class DetectionTrainingStepAdapter:
    """Computes detection training loss by summing model-internal losses."""

    def compute_training_loss(
        self,
        model: nn.Module,
        inputs: Any,
        targets: Any,
        criterion: nn.Module,  # noqa: ARG002
        mixup_fn: Callable[..., Any] | None = None,  # noqa: ARG002
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Execute detection forward pass and compute total loss.

        Moves images and target dicts to device, calls the model in
        training mode (which returns a loss dict), and sums all loss
        components into a single scalar for backpropagation.

        Args:
            model: Detection model (e.g. Faster R-CNN) in training mode.
            inputs: List of image tensors, one per image in the batch.
            targets: List of target dicts, each with ``boxes`` and ``labels``.
            criterion: Ignored (detection models compute losses internally).
            mixup_fn: Ignored (MixUp is not applicable to detection).
            device: Target device for tensor placement.

        Returns:
            Scalar loss tensor (sum of all loss components).
        """
        if device is not None:
            images = [img.to(device) for img in inputs]
            targets_on_device: list[dict[str, Any]] = [
                {k: v.to(device) for k, v in t.items()} for t in targets
            ]
        else:
            images = list(inputs)
            targets_on_device = list(targets)

        loss_dict = model(images, targets_on_device)
        total_loss = torch.stack(list(loss_dict.values())).sum()
        return total_loss
