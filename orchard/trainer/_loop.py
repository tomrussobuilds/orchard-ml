"""
Shared Training Loop Kernel.

Provides the ``TrainingLoop`` class and factory functions for AMP scaler
and MixUp initialization. Both ``ModelTrainer`` and ``TrialTrainingExecutor``
compose a ``TrainingLoop`` to avoid duplicating per-epoch execution logic.

Design:

- ``run_epoch()`` executes the full train → validate → schedule cycle.
- ``run_train_step()`` executes training only (no validation, no scheduling),
  for callers that need custom validation or scheduling logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler

from ..core import TrainingConfig
from ._scheduling import step_scheduler
from .engine import mixup_data, train_one_epoch, validate_epoch

# ── Factory Functions ──────────────────────────────────────────────────────


def create_amp_scaler(training: TrainingConfig) -> torch.amp.GradScaler | None:
    """
    Create AMP GradScaler if mixed precision is enabled.

    Args:
        training: Training sub-config (reads ``use_amp``).

    Returns:
        GradScaler instance when AMP is enabled, None otherwise.
    """
    return torch.amp.GradScaler() if training.use_amp else None


def create_mixup_fn(training: TrainingConfig) -> Callable | None:
    """
    Create a seeded MixUp partial function if alpha > 0.

    Args:
        training: Training sub-config (reads ``mixup_alpha``
            and ``seed``).

    Returns:
        Partial of ``mixup_data`` with fixed alpha and seeded RNG,
        or None when MixUp is disabled.
    """
    if training.mixup_alpha > 0:
        rng = np.random.default_rng(training.seed)
        return partial(mixup_data, alpha=training.mixup_alpha, rng=rng)
    return None


# ── Loop Options ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class LoopOptions:
    """
    Scalar configuration for a :class:`TrainingLoop`.

    Groups training hyper-parameters that do not depend on PyTorch objects,
    keeping the ``TrainingLoop`` constructor lean.

    Attributes:
        grad_clip (float): Max norm for gradient clipping (0 disables).
        total_epochs (int): Total number of epochs (for tqdm progress bar).
        mixup_epochs (int): Epoch cutoff after which MixUp is disabled.
        use_tqdm (bool): Whether to show tqdm progress bar.
        monitor_metric (str): Metric key for ReduceLROnPlateau stepping
            (e.g. ``"auc"``, ``"accuracy"``).
    """

    grad_clip: float
    total_epochs: int
    mixup_epochs: int
    use_tqdm: bool
    monitor_metric: str


# ── Training Loop ──────────────────────────────────────────────────────────


class TrainingLoop:
    """
    Single-epoch execution kernel shared by ModelTrainer and TrialTrainingExecutor.

    Encapsulates the per-epoch train/validate/schedule cycle. Callers own the
    outer epoch loop and policy decisions (checkpointing, early stopping,
    Optuna pruning). This class only executes one epoch at a time.

    Attributes:
        model (nn.Module): Neural network to train.
        train_loader (DataLoader): Training data provider.
        val_loader (DataLoader): Validation data provider.
        optimizer (Optimizer): Gradient descent optimizer.
        scheduler (LRScheduler | None): Learning rate scheduler (or None).
        criterion (nn.Module): Loss function.
        device (torch.device): Hardware target (CUDA/MPS/CPU).
        scaler (GradScaler | None): AMP GradScaler (or None).
        mixup_fn (Callable | None): MixUp partial function (or None).
        options (LoopOptions): Scalar training options (see :class:`LoopOptions`).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: LRScheduler | None,
        criterion: nn.Module,
        device: torch.device,
        scaler: torch.amp.GradScaler | None,
        mixup_fn: Callable | None,
        options: LoopOptions,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.scaler = scaler
        self.mixup_fn = mixup_fn
        self.options = options

    def run_train_step(self, epoch: int) -> float:
        """
        Execute a single training epoch with MixUp cutoff.

        Applies MixUp augmentation only when ``epoch <= mixup_epochs``.
        Does **not** run validation or step the scheduler.

        Args:
            epoch: Current epoch number (1-indexed).

        Returns:
            Average training loss for the epoch.
        """
        current_mixup = self.mixup_fn if epoch <= self.options.mixup_epochs else None
        return train_one_epoch(
            model=self.model,
            loader=self.train_loader,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=self.device,
            mixup_fn=current_mixup,
            scaler=self.scaler,
            grad_clip=self.options.grad_clip,
            epoch=epoch,
            total_epochs=self.options.total_epochs,
            use_tqdm=self.options.use_tqdm,
        )

    def run_epoch(self, epoch: int) -> tuple[float, dict[str, float]]:
        """
        Execute a full train → validate → schedule cycle for one epoch.

        Args:
            epoch: Current epoch number (1-indexed).

        Returns:
            Tuple of (average training loss, validation metrics dict).
        """
        train_loss = self.run_train_step(epoch)
        val_metrics = validate_epoch(
            model=self.model,
            val_loader=self.val_loader,
            criterion=self.criterion,
            device=self.device,
        )
        step_scheduler(self.scheduler, val_metrics[self.options.monitor_metric])
        return train_loss, val_metrics
