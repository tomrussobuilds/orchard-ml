"""
Custom Loss Functions Module

This module implements advanced objective functions for computer vision tasks,
extending standard PyTorch criteria. It includes specialized losses like
Focal Loss to handle extreme class imbalances and difficult samples
often encountered in medical imaging and fine-grained classification.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# CUSTOM LOSSES
class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss for multi-class classification.

    Focal Loss reshapes the standard Cross Entropy loss such that it
    down-weights the loss assigned to well-classified (easy) examples,
    focusing the model's learning on hard, misclassified samples.

    Formula:
        Loss = -alpha * (1 - pt)^gamma * log(pt)
        where pt is the probability of the true class.

    Attributes:
        gamma (float): Focusing parameter. Higher values reduce the relative
            loss for well-classified examples (default: 2.0).
        alpha (float): Balancing parameter for class importance.
        weight (torch.Tensor, optional): A manual rescaling weight given to
            each class.
    """

    def __init__(
        self, gamma: float = 2.0, alpha: float = 1.0, weight: Optional[torch.Tensor] = None
    ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the focal loss between input logits and ground truth targets.

        Args:
            inputs (torch.Tensor): Model predictions (logits) of shape (N, C).
            targets (torch.Tensor): Ground truth labels of shape (N).

        Returns:
            torch.Tensor: The computed scalar loss.
        """
        # Calculate standard cross entropy without reduction
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.weight)

        # pt is the probability of the correct class
        pt = torch.exp(-ce_loss)

        # Compute focal loss components
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


# HELPER FUNCTIONS
def get_loss_name(criterion: nn.Module) -> str:
    """
    Utility to extract a human-readable name from a loss instance.
    """
    return criterion.__class__.__name__
