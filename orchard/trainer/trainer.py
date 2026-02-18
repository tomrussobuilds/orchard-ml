"""
Model Training & Lifecycle Orchestration.

This module encapsulates the `ModelTrainer` engine, responsible for executing
the training loop, validation phases, and learning rate scheduling. It bridges
validated configurations with execution kernels, ensuring atomic state management
through specialized checkpointing and weight restoration logic.

Key Features:
    - Automated Checkpointing: Tracks performance metrics (AUC/Accuracy) and
      persists the optimal model state.
    - Deterministic Restoration: Guarantees that the model instance in memory
      reflects the 'best' found parameters upon completion.
    - Modern Training Utilities: Native support for Mixed Precision (AMP),
      Gradient Clipping, and Mixup augmentation.
    - Lifecycle Telemetry: Unified logging of loss trajectories, metric
      evolution, and resource utilization.
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from orchard.core import LOGGER_NAME, Config, load_model_weights

if TYPE_CHECKING:  # pragma: no cover
    from orchard.tracking import TrackerProtocol

from .engine import mixup_data, train_one_epoch, validate_epoch

# Global logger instance
logger = logging.getLogger(LOGGER_NAME)


# TRAINING LOGIC
class ModelTrainer:
    """
    Encapsulates the core training, validation, and scheduling logic.

    Manages the complete training lifecycle including epoch iteration, metric tracking,
    automated checkpointing based on validation performance, and early stopping with
    patience-based criteria. Integrates modern training techniques (AMP, Mixup, gradient
    clipping) and ensures deterministic model restoration to best-performing weights.

    The trainer follows a structured execution flow:
        1. Training Phase: Forward/backward passes with optional Mixup augmentation
        2. Validation Phase: Performance evaluation on held-out data
        3. Scheduling Phase: Learning rate updates (ReduceLROnPlateau or step-based)
        4. Checkpointing: Save model when validation AUC improves
        5. Early Stopping: Halt training if no improvement for `patience` epochs

    Attributes:
        model (nn.Module): Neural network architecture to train
        train_loader (DataLoader): Training data provider
        val_loader (DataLoader): Validation data provider
        optimizer (torch.optim.Optimizer): Gradient descent optimizer
        scheduler (LRScheduler): Learning rate scheduler
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss)
        device (torch.device): Hardware target (CUDA/MPS/CPU)
        cfg (Config): Global configuration manifest (SSOT)
        output_path (Path | None): Checkpoint save location (default: ./best_model.pth)
        epochs (int): Total number of training epochs
        patience (int): Early stopping patience (epochs without improvement)
        best_acc (float): Best validation accuracy achieved
        best_auc (float): Best validation AUC achieved
        epochs_no_improve (int): Consecutive epochs without AUC improvement
        scaler (torch.amp.GradScaler): Automatic Mixed Precision scaler
        mixup_fn (callable | None): Mixup augmentation function (partial of mixup_data)
        best_path (Path): Filesystem path for best model checkpoint
        train_losses (List[float]): Training loss history per epoch
        val_metrics_history (List[dict]): Validation metrics history per epoch
        val_aucs (List[float]): Validation AUC history per epoch

    Example:
        >>> from orchard.trainer import ModelTrainer
        >>> trainer = ModelTrainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     criterion=criterion,
        ...     device=device,
        ...     cfg=cfg,
        ...     output_path=paths.models / "best_model.pth"
        ... )
        >>> checkpoint_path, losses, metrics = trainer.train()
        >>> # Model automatically restored to best weights
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: LRScheduler,
        criterion: nn.Module,
        device: torch.device,
        cfg: Config,
        output_path: Path | None = None,
        tracker: TrackerProtocol | None = None,
    ):
        """
        Initializes the ModelTrainer with all required training components.

        Args:
            model: Neural network architecture to train
            train_loader: DataLoader for training dataset
            val_loader: DataLoader for validation dataset
            optimizer: Gradient descent optimizer (e.g., SGD, Adam)
            scheduler: Learning rate scheduler for training dynamics
            criterion: Loss function for optimization (e.g., CrossEntropyLoss)
            device: Compute device (torch.device) for training
            cfg: Validated global configuration containing training hyperparameters
            output_path: Optional path for best model checkpoint (default: ./best_model.pth)
            tracker: Optional experiment tracker for MLflow metric logging
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.cfg = cfg
        self.tracker = tracker

        # Hyperparameters
        self.epochs = cfg.training.epochs
        self.patience = cfg.training.patience
        self.best_acc = -1.0
        self.best_auc = -1.0
        self.epochs_no_improve = 0

        # Modern AMP Support (PyTorch 2.x+)
        self.scaler = torch.amp.GradScaler(enabled=cfg.training.use_amp)

        # Mixup configuration (RNG seeded from training seed for reproducibility)
        self.mixup_fn = None
        if cfg.training.mixup_alpha > 0:
            mixup_rng = np.random.default_rng(cfg.training.seed)
            self.mixup_fn = partial(mixup_data, alpha=cfg.training.mixup_alpha, rng=mixup_rng)

        # Output Management
        self.best_path = output_path or Path("./best_model.pth")
        self.best_path.parent.mkdir(parents=True, exist_ok=True)

        # History tracking
        self.train_losses: List[float] = []
        self.val_metrics_history: List[dict] = []
        self.val_aucs: List[float] = []

        # Track if we saved at least one valid checkpoint during training
        self._checkpoint_saved: bool = False

        logger.info(f"Trainer initialized. Best model checkpoint: {self.best_path.name}")

    def train(self) -> Tuple[Path, List[float], List[dict]]:
        """
        Executes the main training loop with checkpointing and early stopping.

        Performs iterative training across configured epochs, executing:
            - Forward/backward passes with optional Mixup augmentation
            - Validation metric computation (loss, accuracy, AUC)
            - Learning rate scheduling (plateau-aware or step-based)
            - Automated checkpointing on validation AUC improvement
            - Early stopping with patience-based criteria

        Returns:
            Tuple containing:
                - Path: Filesystem path to best model checkpoint
                - List[float]: Training loss history per epoch
                - List[dict]: Validation metrics history (loss, accuracy, AUC per epoch)

        Notes:
            - Model weights are automatically restored to best checkpoint after training
            - Mixup augmentation is disabled after cosine_fraction × epochs
            - Early stopping triggers if no AUC improvement for `patience` epochs
        """
        for epoch in range(1, self.epochs + 1):
            logger.info(f" Epoch {epoch:02d}/{self.epochs} ".center(60, "-"))

            mixup_cutoff = int(self.cfg.training.cosine_fraction * self.epochs)
            current_mixup = self.mixup_fn if epoch <= mixup_cutoff else None

            # --- 1. Training Phase ---
            epoch_loss = train_one_epoch(
                model=self.model,
                loader=self.train_loader,
                criterion=self.criterion,
                optimizer=self.optimizer,
                device=self.device,
                mixup_fn=current_mixup,
                scaler=self.scaler,
                grad_clip=self.cfg.training.grad_clip,
                epoch=epoch,
                total_epochs=self.epochs,
            )
            self.train_losses.append(epoch_loss)

            # --- 2. Validation Phase ---
            val_metrics = validate_epoch(
                model=self.model,
                val_loader=self.val_loader,
                criterion=self.criterion,
                device=self.device,
            )
            self.val_metrics_history.append(val_metrics)

            val_acc = val_metrics["accuracy"]
            val_loss = val_metrics["loss"]
            val_auc = val_metrics.get("auc", 0.0)

            logger.info(
                f"Epoch {epoch} Validation AUC: {val_auc:.4f} "
                f"Previous Best AUC: {self.best_auc:.4f}"
            )
            self.val_aucs.append(val_auc)

            # --- 3. Scheduling Phase ---
            self._smart_step_scheduler(val_loss)

            if val_acc > self.best_acc:
                self.best_acc = val_acc

            # --- 4. Checkpoint & Early Stopping Logic ---
            if self._handle_checkpointing(val_metrics):
                logger.warning(f"Early stopping triggered at epoch {epoch}.")
                break

            # --- 5. Unified Logging ---
            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Loss: [T: {epoch_loss:.4f} | V: {val_loss:.4f}] | "
                f"Acc: {val_acc:.4f} (Best Acc: {self.best_acc:.4f}) | "
                f"AUC: {val_auc:.4f} (Best AUC: {self.best_auc:.4f}) | "
                f"LR: {current_lr:.2e} | Patience: {self.patience - self.epochs_no_improve}"
            )

            # --- 6. Experiment Tracking ---
            if self.tracker is not None:
                self.tracker.log_epoch(epoch, epoch_loss, val_metrics, current_lr)

        logger.info(
            f"Training finished. Peak Performance -> AUC: {self.best_auc:.4f} "
            f"| Acc: {self.best_acc:.4f}"
        )

        self._finalize_weights()

        return self.best_path, self.train_losses, self.val_metrics_history

    def _smart_step_scheduler(self, val_loss: float) -> None:
        """
        Updates the learning rate scheduler based on its type.

        If the scheduler is an instance of ReduceLROnPlateau, it requires
        the validation loss to determine the next step. Otherwise, it
        performs a standard step.

        Args:
            val_loss: Current epoch's validation loss
        """
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        else:
            if not isinstance(self.scaler, torch.amp.GradScaler):
                self.scheduler.step()

    def _handle_checkpointing(self, val_metrics: dict) -> bool:
        """
        Manages model checkpointing and tracks early stopping progress.

        Saves the model state if the current validation AUC exceeds
        the previous best. Increments the patience counter otherwise.

        Args:
            val_metrics: Validation metrics dictionary containing 'accuracy' and 'auc'

        Returns:
            True if early stopping criteria are met, False otherwise
        """
        val_acc = val_metrics["accuracy"]
        val_auc = val_metrics.get("auc", 0.0)

        if val_acc > self.best_acc:
            self.best_acc = val_acc

        if val_auc > self.best_auc:
            logger.info(f"New best model! Val AUC: {val_auc:.4f} ↑ Checkpoint saved.")
            self.best_auc = val_auc
            self.epochs_no_improve = 0
            torch.save(self.model.state_dict(), self.best_path)
            self._checkpoint_saved = True
        else:
            self.epochs_no_improve += 1

        return self.epochs_no_improve >= self.patience

    def _finalize_weights(self) -> None:
        """
        Decide which weights to keep after training ends.

        Handles three scenarios:
            1. Checkpoint saved during training → load best
            2. No improvement but checkpoint file exists (prior run) → load it
            3. No checkpoint at all → save current weights as fallback
        """
        if self._checkpoint_saved and self.best_path.exists():
            self.load_best_weights()
            return

        no_improve_msg = "No checkpoint was saved during training (model never improved)."
        if self.best_path.exists():
            logger.warning(f"{no_improve_msg} Loading existing checkpoint file.")
            self.load_best_weights()
        else:
            logger.warning(f"{no_improve_msg} Saving current model state as fallback.")
            torch.save(self.model.state_dict(), self.best_path)

    def load_best_weights(self) -> None:
        """
        Load the best checkpoint from disk into the model (device-aware).

        Raises:
            RuntimeError: If the state-dict is incompatible with the model.
            FileNotFoundError: If the checkpoint file does not exist.
        """
        try:
            load_model_weights(model=self.model, path=self.best_path, device=self.device)
            logger.info(
                f" » [LIFECYCLE] Success: Model state restored to best checkpoint "
                f"({self.best_path.name})"
            )
        except (RuntimeError, FileNotFoundError) as e:
            logger.error(f" » [LIFECYCLE] Critical failure during weight restoration: {e}")
            raise
