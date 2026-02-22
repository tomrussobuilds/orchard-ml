"""
Optimization Setup Module

This module provides factory functions to instantiate PyTorch optimization
components (optimizers, schedulers, and loss functions) based on the
training configuration sub-model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from ..core import TrainingConfig
from .losses import FocalLoss


# FACTORIES
def get_criterion(training: TrainingConfig, class_weights: torch.Tensor | None = None) -> nn.Module:
    """
    Universal Vision Criterion Factory.

    Args:
        training: Training sub-config with criterion parameters.
        class_weights: Optional per-class weights for imbalanced datasets.
    """
    c_type = training.criterion_type.lower()
    weights = class_weights if training.weighted_loss else None

    if c_type == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=training.label_smoothing, weight=weights)

    elif c_type == "bce_logit":
        return nn.BCEWithLogitsLoss(pos_weight=weights)

    elif c_type == "focal":
        return FocalLoss(gamma=training.focal_gamma, weight=weights)

    else:
        raise ValueError(f"Unknown criterion type: {c_type}")


def get_optimizer(model: nn.Module, training: TrainingConfig) -> optim.Optimizer:
    """
    Factory function to instantiate optimizer from config.

    Dispatches on training.optimizer_type:
        - sgd: SGD with momentum, suited for convolutional architectures.
        - adamw: AdamW with decoupled weight decay, suited for transformers.
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
            f"Unknown optimizer type: '{opt_type}'. " "Available options: ['sgd', 'adamw']"
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
        - cosine: Smooth decay following a cosine curve.
        - plateau: Reduces LR when a metric (loss) stops improving.
        - step: Periodic reduction by a fixed factor.
        - none: Maintains a constant learning rate.
    """
    sched_type = training.scheduler_type.lower()

    if sched_type == "cosine":
        return lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=training.epochs, eta_min=training.min_lr
        )

    elif sched_type == "plateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
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
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

    else:
        raise ValueError(
            f"Unsupported scheduler_type: '{sched_type}'. "
            "Available options: ['cosine', 'plateau', 'step', 'none']"
        )
