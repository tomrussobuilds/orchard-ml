"""
Optimization Setup Module.

This module provides factory functions to instantiate PyTorch optimization
components (optimizers, schedulers, and loss functions) based on the
training configuration sub-model.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from ..core import TrainingConfig
from .losses import FocalLoss


# CLASS WEIGHTS
def compute_class_weights(
    labels: np.ndarray, num_classes: int, device: torch.device
) -> torch.Tensor:
    """
    Compute balanced class weights (sklearn formula: N / (n_classes * count_c)).

    Args:
        labels: Training set labels (1D array).
        num_classes: Total number of classes.
        device: Target device for the weight tensor.

    Returns:
        1D tensor of per-class weights, shape ``(num_classes,)``.
    """
    classes, counts = np.unique(labels, return_counts=True)
    n_total = len(labels)
    weight_map = {int(c): n_total / (num_classes * cnt) for c, cnt in zip(classes, counts)}
    weights = [weight_map.get(i, 1.0) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float).to(device)


# FACTORIES
def get_criterion(training: TrainingConfig, class_weights: torch.Tensor | None = None) -> nn.Module:
    """
    Universal Vision Criterion Factory.

    Args:
        training: Training sub-config with criterion parameters.
        class_weights: Optional per-class weights for imbalanced datasets.

    Returns:
        Loss module (CrossEntropyLoss or FocalLoss).

    Raises:
        ValueError: If ``training.criterion_type`` is not recognised.
    """
    c_type = training.criterion_type.lower()
    weights = class_weights if training.weighted_loss else None

    if c_type == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=training.label_smoothing, weight=weights)

    elif c_type == "focal":
        return FocalLoss(gamma=training.focal_gamma, weight=weights)

    else:
        raise ValueError(f"Unknown criterion type: {c_type}")


def get_optimizer(model: nn.Module, training: TrainingConfig) -> optim.Optimizer:
    """
    Factory function to instantiate optimizer from config.

    Dispatches on ``training.optimizer_type``:

    - **sgd** — SGD with momentum, suited for convolutional architectures.
    - **adamw** — AdamW with decoupled weight decay, suited for transformers.

    Args:
        model: Network whose parameters will be optimised.
        training: Training sub-config with optimizer hyper-parameters.

    Returns:
        Configured optimizer instance.

    Raises:
        ValueError: If ``training.optimizer_type`` is not recognised.
    """
    opt_type = training.optimizer_type.lower()

    if opt_type == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=training.learning_rate,
            momentum=training.momentum,
            weight_decay=training.weight_decay,
        )

    elif opt_type == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=training.learning_rate,
            weight_decay=training.weight_decay,
        )

    else:
        raise ValueError(
            f"Unknown optimizer type: '{opt_type}'. Available options: ['sgd', 'adamw']"
        )


def get_scheduler(
    optimizer: optim.Optimizer, training: TrainingConfig
) -> (
    lr_scheduler.CosineAnnealingLR
    | lr_scheduler.ReduceLROnPlateau
    | lr_scheduler.StepLR
    | lr_scheduler.LambdaLR
):
    """
    Advanced Scheduler Factory.

    Supports multiple LR decay strategies based on TrainingConfig:

    - **cosine** — Smooth decay following a cosine curve.
    - **plateau** — Reduces LR when ``monitor_metric`` stops improving (``mode="max"``).
    - **step** — Periodic reduction by a fixed factor.
    - **none** — Maintains a constant learning rate.

    Args:
        optimizer: Optimizer whose learning rate will be scheduled.
        training: Training sub-config with scheduler hyper-parameters.

    Returns:
        Configured learning rate scheduler instance.

    Raises:
        ValueError: If ``training.scheduler_type`` is not recognised.
    """
    sched_type = training.scheduler_type.lower()

    if sched_type == "cosine":
        return lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=training.epochs, eta_min=training.min_lr
        )

    elif sched_type == "plateau":
        # monitor_metric is Literal["auc", "accuracy", "f1"] — all maximize
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=training.scheduler_factor,
            patience=training.scheduler_patience,
            min_lr=training.min_lr,
        )

    elif sched_type == "step":
        return lr_scheduler.StepLR(
            optimizer, step_size=training.step_size, gamma=training.scheduler_factor
        )

    elif sched_type == "none":
        # Returns a dummy scheduler that keeps LR constant
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _epoch: 1.0)

    else:
        raise ValueError(
            f"Unsupported scheduler_type: '{sched_type}'. "
            "Available options: ['cosine', 'plateau', 'step', 'none']"
        )
