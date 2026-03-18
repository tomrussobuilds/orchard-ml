"""
Training execution utilities for Optuna trials.

Provides TrialTrainingExecutor, which orchestrates the training and validation
loop for a single Optuna trial with built-in pruning, metric tracking, and
scheduler management. Per-epoch training is delegated to ``_loop.TrainingLoop``
(shared with ModelTrainer), while validation remains local with error-resilient
fallback metrics.

Key responsibilities:

- Execute epoch-level training/validation cycles
- Apply Optuna pruning logic with warmup period
- Track and report metrics to Optuna
- Handle scheduler stepping (plateau-aware)
- Provide error-resilient validation with fallback metrics

.. todo::
   Unify ``TrialTrainingExecutor`` and ``ModelTrainer`` into a single
   engine with pluggable epoch-end callbacks (early stopping,
   checkpointing, Optuna pruning).  Both already share the full
   training kernel (``TrainingLoop``, ``validate_epoch``,
   ``step_scheduler``, AMP scaler, Mixup); the only divergence is
   the epoch-level loop and post-validation actions.
"""

from __future__ import annotations

import logging
import math
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from ...core.task_protocols import TaskTrainingStep, TaskValidationMetrics

import optuna
import torch

from ...core import LOGGER_NAME, LogStyle, OptunaConfig, TrainingConfig
from ...core.paths import METRIC_LOSS
from ...trainer import validate_epoch
from ...trainer._loop import (
    LoopOptions,
    TrainingLoop,
    create_amp_scaler,
    create_mixup_fn,
)
from ...trainer._scheduling import step_scheduler
from .metric_extractor import MetricExtractor

logger = logging.getLogger(LOGGER_NAME)

# Module-level constants
_GENERIC_FALLBACK = MappingProxyType({METRIC_LOSS: 999.0})
_MAX_CONSECUTIVE_VAL_FAILURES: int = 3


# TRAINING EXECUTOR
class TrialTrainingExecutor:
    """
    Executes training loop with Optuna pruning integration.

    Orchestrates a complete training cycle for a single Optuna trial, including:

    - Training and validation epochs
    - Metric extraction and tracking
    - Pruning decisions with warmup period
    - Learning rate scheduling
    - Progress logging

    Pruning and warmup parameters are read from the ``optuna`` sub-config;
    training hyperparameters from ``training``.

    Attributes:
        model: PyTorch model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        criterion: Loss function.
        device: Training device (CPU/CUDA/MPS).
        metric_extractor: Handles metric extraction and best-value tracking.
        enable_pruning: Whether to enable trial pruning.
        warmup_epochs: Epochs before pruning activates.
        monitor_metric: Name of the metric driving scheduling.
        scaler (GradScaler | None): AMP gradient scaler (None when use_amp is False).
        mixup_fn (callable | None): Mixup augmentation function (None when alpha is 0).
        epochs: Total training epochs.
        log_interval: Epoch interval for progress logging.
        _loop (TrainingLoop): Shared epoch kernel for training steps (train only, no validation).

    Example:
        >>> executor = TrialTrainingExecutor(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     criterion=criterion,
        ...     training=trial_cfg.training,
        ...     optuna=trial_cfg.optuna,
        ...     log_interval=trial_cfg.telemetry.log_interval,
        ...     device=device,
        ...     metric_extractor=MetricExtractor("auc"),
        ... )
        >>> best_metric = executor.execute(trial)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader[Any],
        val_loader: torch.utils.data.DataLoader[Any],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        criterion: torch.nn.Module,
        training: TrainingConfig,
        optuna: OptunaConfig,
        log_interval: int,
        device: torch.device,
        metric_extractor: MetricExtractor,
        training_step: TaskTrainingStep | None = None,
        validation_metrics: TaskValidationMetrics | None = None,
        fallback_metrics: Mapping[str, float] | None = None,
    ) -> None:
        """
        Initialize training executor.

        Args:
            model: PyTorch model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            optimizer: Optimizer instance.
            scheduler: Learning rate scheduler.
            criterion: Loss function.
            training: Training hyperparameters sub-config.
            optuna: Optuna pruning/warmup sub-config.
            log_interval: Epoch interval for progress logging.
            device: Training device.
            metric_extractor: Metric extraction and tracking handler.
            training_step: Task-specific forward pass adapter.
                If provided, used instead of the default classification-specific
                forward logic. Obtained via ``get_task(task_type).training_step``.
            validation_metrics: Task-specific validation metrics adapter.
                If provided, used instead of the default ``validate_epoch``.
            fallback_metrics: Metrics returned on validation failure.
                Defaults to a minimal generic fallback (loss only).
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.metric_extractor = metric_extractor
        self._validation_metrics = validation_metrics
        resolved_fallback = fallback_metrics or _GENERIC_FALLBACK

        # Ensure fallback metrics include monitor_metric for scheduler stepping.
        # Task-specific fallback_metrics should always include it; the generic
        # fallback is augmented automatically with a neutral value.
        if training.monitor_metric not in resolved_fallback:
            augmented = dict(resolved_fallback)
            augmented[training.monitor_metric] = 0.0
            self._fallback_metrics: Mapping[str, float] = MappingProxyType(augmented)
        else:
            self._fallback_metrics = resolved_fallback

        # Pruning config
        self.enable_pruning = optuna.enable_pruning
        self.warmup_epochs = optuna.pruning_warmup_epochs

        # Training state
        self.scaler = create_amp_scaler(training, device=str(device))
        self.mixup_fn = create_mixup_fn(training)
        self.epochs = training.epochs
        self.monitor_metric = training.monitor_metric
        self.log_interval = log_interval
        self._consecutive_val_failures: int = 0

        # Shared epoch kernel (train step only — validation is error-resilient here)
        self._loop = TrainingLoop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            scaler=self.scaler,
            mixup_fn=self.mixup_fn,
            options=LoopOptions(
                grad_clip=training.grad_clip,
                total_epochs=self.epochs,
                mixup_epochs=training.mixup_epochs,
                use_tqdm=False,
                monitor_metric=self.monitor_metric,
            ),
            training_step=training_step,
        )

    def execute(self, trial: optuna.Trial) -> float:
        """
        Execute full training loop with pruning.

        Runs training for cfg.training.epochs, reporting metrics to Optuna
        after each epoch. Applies pruning logic after warmup period.

        Args:
            trial: Optuna trial for reporting and pruning

        Returns:
            Best validation metric achieved during training

        Raises:
            optuna.TrialPruned: If trial should terminate early
        """
        for epoch in range(1, self.epochs + 1):
            # Train (delegated to shared loop)
            epoch_loss = self._loop.run_train_step(epoch)

            # Validate
            val_metrics = self._validate_epoch()

            # Extract and track metric
            current_metric = self.metric_extractor.extract(val_metrics)
            best_metric = self.metric_extractor.update_best(current_metric)

            # Report to Optuna (skip NaN to avoid poisoning the pruner)
            if not math.isnan(current_metric):
                trial.report(current_metric, epoch)

            # Check pruning
            if self._should_prune(trial, epoch):
                logger.info(
                    "%s%s Trial %d pruned at epoch %d (%s=%.4f)",
                    LogStyle.INDENT,
                    LogStyle.ARROW,
                    trial.number,
                    epoch,
                    self.metric_extractor.metric_name,
                    current_metric,
                )
                raise optuna.TrialPruned()

            # Scheduler step (uses monitor_metric, consistent with ModelTrainer)
            step_scheduler(self.scheduler, val_metrics[self.monitor_metric])

            # Logging
            if epoch % self.log_interval == 0 or epoch == self.epochs:
                logger.info(
                    "%sT%d E%d/%d | Loss:%.4f | %s:%.4f (Best:%.4f)",
                    LogStyle.DOUBLE_INDENT,
                    trial.number,
                    epoch,
                    self.epochs,
                    epoch_loss,
                    self.metric_extractor.metric_name,
                    current_metric,
                    best_metric,
                )

        self._log_trial_complete(trial, best_metric, epoch_loss)
        return best_metric

    def _validate_epoch(self) -> Mapping[str, float]:
        """
        Validate single epoch with error handling.

        Uses the injected ``TaskValidationMetrics`` adapter when available,
        falling back to the default ``validate_epoch`` engine otherwise.

        Returns:
            Dictionary of validation metrics (loss, accuracy, auc, etc.)
            Returns fallback metrics on validation failure
        """
        try:
            if self._validation_metrics is not None:
                metrics = self._validation_metrics.compute_validation_metrics(
                    model=self.model,
                    val_loader=self.val_loader,
                    criterion=self.criterion,
                    device=self.device,
                )
            else:
                metrics = validate_epoch(
                    model=self.model,
                    val_loader=self.val_loader,
                    criterion=self.criterion,
                    device=self.device,
                )
            self._consecutive_val_failures = 0
            return metrics

        except (RuntimeError, ValueError) as e:
            self._consecutive_val_failures += 1
            logger.error(  # pragma: no mutate
                "%s%s Validation failed (x%d): %s",
                LogStyle.INDENT,  # pragma: no mutate
                LogStyle.FAILURE,  # pragma: no mutate
                self._consecutive_val_failures,  # pragma: no mutate
                e,  # pragma: no mutate
            )
            if self._consecutive_val_failures >= _MAX_CONSECUTIVE_VAL_FAILURES:
                raise RuntimeError(
                    f"Validation failed {self._consecutive_val_failures} consecutive times, "
                    "aborting trial"  # pragma: no mutate
                ) from e
            return dict(self._fallback_metrics)

    def _should_prune(self, trial: optuna.Trial, epoch: int) -> bool:
        """
        Check if trial should be pruned.

        Pruning is disabled if:

        - enable_pruning is False
        - epoch < warmup_epochs

        Args:
            trial: Optuna trial
            epoch: Current epoch number

        Returns:
            True if trial should be pruned, False otherwise
        """
        if not self.enable_pruning or epoch < self.warmup_epochs:
            return False
        return trial.should_prune()

    def _log_trial_complete(
        self,
        trial: optuna.Trial,
        best_metric: float,
        final_loss: float,
    ) -> None:
        """
        Log trial completion summary.

        Args:
            trial: Optuna trial
            best_metric: Best metric achieved
            final_loss: Final training loss
        """
        logger.info("")
        logger.info(
            "%s%s Trial %d completed",
            LogStyle.INDENT,
            LogStyle.SUCCESS,
            trial.number,
        )
        best_label = "Best %s" % self.metric_extractor.metric_name.upper()  # pragma: no mutate
        logger.info(
            "%s%s %-18s: %.6f",
            LogStyle.INDENT,
            LogStyle.ARROW,
            best_label,
            best_metric,
        )
        logger.info(
            "%s%s %-18s: %.4f",
            LogStyle.INDENT,
            LogStyle.ARROW,
            "Final Loss",
            final_loss,
        )
        logger.info("")
