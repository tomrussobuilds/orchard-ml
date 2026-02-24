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
"""

from __future__ import annotations

import logging

import optuna
import torch

from ...core import LOGGER_NAME, Config, LogStyle
from ...core.paths import METRIC_ACCURACY, METRIC_AUC, METRIC_LOSS
from ...trainer import validate_epoch
from ...trainer._loop import TrainingLoop, create_amp_scaler, create_mixup_fn
from ...trainer._scheduling import step_scheduler
from .metric_extractor import MetricExtractor

logger = logging.getLogger(LOGGER_NAME)

# Module-level constants
_FALLBACK_METRICS: dict[str, float] = {METRIC_LOSS: 999.0, METRIC_ACCURACY: 0.0, METRIC_AUC: 0.0}


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

    All pruning and warmup parameters are read from cfg.optuna to enforce
    single source of truth.

    Attributes:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        criterion: Loss function
        cfg: Complete trial configuration
        device: Training device (CPU/CUDA/MPS)
        metric_extractor: Handles metric extraction and best-value tracking
        enable_pruning: Whether to enable trial pruning (from cfg.optuna.enable_pruning)
        warmup_epochs: Epochs before pruning activates (from cfg.optuna.pruning_warmup_epochs)
        scaler (GradScaler | None): AMP gradient scaler (None when use_amp is False)
        mixup_fn (callable | None): Mixup augmentation function (None when alpha is 0)
        epochs: Total training epochs
        log_interval: Epoch interval for progress logging
        _loop (TrainingLoop): Shared epoch kernel for training steps (train only, no validation)

    Example:
        >>> executor = TrialTrainingExecutor(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     criterion=criterion,
        ...     cfg=trial_cfg,
        ...     device=device,
        ...     metric_extractor=MetricExtractor("auc"),
        ... )
        >>> best_metric = executor.execute(trial)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        cfg: Config,
        device: torch.device,
        metric_extractor: MetricExtractor,
    ) -> None:
        """
        Initialize training executor.

        Args:
            model (torch.nn.Module): PyTorch model to train
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            optimizer (torch.optim.Optimizer): Optimizer instance
            scheduler (LRScheduler | None): Learning rate scheduler
            criterion (torch.nn.Module): Loss function
            cfg (Config): Trial configuration (reads optuna.* settings)
            device (torch.device): Training device
            metric_extractor (MetricExtractor): Metric extraction and tracking handler
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.cfg = cfg
        self.device = device
        self.metric_extractor = metric_extractor

        # Read pruning config from cfg.optuna (single source of truth)
        self.enable_pruning = cfg.optuna.enable_pruning
        self.warmup_epochs = cfg.optuna.pruning_warmup_epochs

        # Training state
        self.scaler = create_amp_scaler(cfg)
        self.mixup_fn = create_mixup_fn(cfg)
        self.epochs = cfg.training.epochs
        self.log_interval = cfg.telemetry.log_interval

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
            grad_clip=cfg.training.grad_clip,
            total_epochs=self.epochs,
            mixup_epochs=cfg.training.mixup_epochs,
            use_tqdm=False,
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

            # Report to Optuna
            trial.report(current_metric, epoch)

            # Check pruning
            if self._should_prune(trial, epoch):
                logger.info(
                    f"{LogStyle.INDENT}{LogStyle.ARROW} "
                    f"Trial {trial.number} pruned at epoch {epoch} "
                    f"({self.metric_extractor.metric_name}={current_metric:.4f})"
                )
                raise optuna.TrialPruned()

            # Scheduler step
            step_scheduler(self.scheduler, val_metrics[METRIC_LOSS])

            # Logging
            if epoch % self.log_interval == 0 or epoch == self.epochs:
                logger.info(
                    f"{LogStyle.DOUBLE_INDENT}T{trial.number} "
                    f"E{epoch}/{self.epochs} | "
                    f"Loss:{epoch_loss:.4f} | "
                    f"{self.metric_extractor.metric_name}:{current_metric:.4f} "
                    f"(Best:{best_metric:.4f})"
                )

        self._log_trial_complete(trial, best_metric, epoch_loss)
        return best_metric

    def _validate_epoch(self) -> dict[str, float]:
        """
        Validate single epoch with error handling.

        Returns:
            Dictionary of validation metrics (loss, accuracy, auc, etc.)
            Returns fallback metrics on validation failure
        """
        try:
            val_metrics = validate_epoch(
                model=self.model,
                val_loader=self.val_loader,
                criterion=self.criterion,
                device=self.device,
            )

            if val_metrics is None or not isinstance(val_metrics, dict):
                logger.error(f"Invalid validation result: {val_metrics}")
                return dict(_FALLBACK_METRICS)

            return val_metrics

        except (RuntimeError, ValueError) as e:
            logger.error(f"Validation failed: {e}")
            return dict(_FALLBACK_METRICS)

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
        logger.info(f"{LogStyle.INDENT}{LogStyle.SUCCESS} Trial {trial.number} completed")
        best_label = f"Best {self.metric_extractor.metric_name.upper()}"
        logger.info(f"{LogStyle.INDENT}{LogStyle.ARROW} {best_label:<18}: {best_metric:.6f}")
        logger.info(f"{LogStyle.INDENT}{LogStyle.ARROW} {'Final Loss':<18}: {final_loss:.4f}")
        logger.info("")
