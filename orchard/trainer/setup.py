"""
Optimization Setup Module

This module provides factory functions to instantiate PyTorch optimization
components (optimizers, schedulers, and loss functions) based on the
hierarchical configuration manifest.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from orchard.core import Config

from .losses import FocalLoss


# FACTORIES
def get_criterion(cfg: Config, class_weights: torch.Tensor | None = None) -> nn.Module:
    """
    Universal Vision Criterion Factory.
    """
    c_type = cfg.training.criterion_type.lower()
    weights = class_weights if cfg.training.weighted_loss else None

    if c_type == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing, weight=weights)

    elif c_type == "bce_logit":
        return nn.BCEWithLogitsLoss(pos_weight=weights)

    elif c_type == "focal":
        return FocalLoss(gamma=cfg.training.focal_gamma, weight=weights)

    else:
        raise ValueError(f"Unknown criterion type: {c_type}")


def get_optimizer(model: nn.Module, cfg: Config) -> optim.Optimizer:
    """
    Factory function to instantiate a task-specific optimizer.

    Decision Logic:
        - ResNet Variants: Uses SGD with Momentum for better generalization
          in convolutional landscapes.
        - Other (ViT/Transformers): Defaults to AdamW to handle decoupled
          weight decay and adaptive learning rates.
    """
    model_name = cfg.architecture.name.lower()

    # Strip timm/ prefix so heuristic works on the actual model identifier
    bare_name = model_name.removeprefix("timm/")

    if "resnet" in bare_name:
        return optim.SGD(
            model.parameters(),
            lr=cfg.training.learning_rate,
            momentum=cfg.training.momentum,
            weight_decay=cfg.training.weight_decay,
        )

    # Extension point: add your custom optimizer here
    # if "your_model" in bare_name:
    #     return optim.Adam(
    #         model.parameters(),
    #         lr=cfg.training.learning_rate,
    #         weight_decay=cfg.training.weight_decay,
    #     )

    # Robust default for modern attention-based or hybrid architectures
    return optim.AdamW(
        model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay
    )


def get_scheduler(
    optimizer: optim.Optimizer, cfg: Config
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
    sched_type = cfg.training.scheduler_type.lower()

    if sched_type == "cosine":
        return lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.epochs, eta_min=cfg.training.min_lr
        )

    elif sched_type == "plateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.training.scheduler_factor,
            patience=cfg.training.scheduler_patience,
            min_lr=cfg.training.min_lr,
        )

    elif sched_type == "step":
        return lr_scheduler.StepLR(
            optimizer, step_size=cfg.training.step_size, gamma=cfg.training.scheduler_factor
        )

    elif sched_type == "none":
        # Returns a dummy scheduler that keeps LR constant
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

    else:
        raise ValueError(
            f"Unsupported scheduler_type: '{sched_type}'. "
            "Available options: ['cosine', 'plateau', 'step', 'none']"
        )
