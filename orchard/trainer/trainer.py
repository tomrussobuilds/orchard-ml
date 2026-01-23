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

# Standard Imports
import logging
from functools import partial
from pathlib import Path
from typing import List, Tuple

# Third-Party Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Internal Imports
from orchard.core import LOGGER_NAME, Config, load_model_weights

from .engine import mixup_data, train_one_epoch, validate_epoch

# TRAINING LOGIC
# Global logger instance
logger = logging.getLogger(LOGGER_NAME)


class ModelTrainer:
    """
    Encapsulates the core training, validation, and scheduling logic.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        criterion: nn.Module,
        device: torch.device,
        cfg: Config,
        output_path: Path | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.cfg = cfg

        # Hyperparameters
        self.epochs = cfg.training.epochs
        self.patience = cfg.training.patience
        self.best_acc = -1.0
        self.best_auc = -1.0
        self.epochs_no_improve = 0

        # Modern AMP Support (PyTorch 2.x+)
        self.scaler = torch.amp.GradScaler(enabled=cfg.training.use_amp)

        # Mixup configuration
        self.mixup_fn = None
        if cfg.training.mixup_alpha > 0:
            self.mixup_fn = partial(mixup_data, alpha=cfg.training.mixup_alpha, device=device)

        # Output Management
        self.best_path = output_path or Path("./best_model.pth")
        self.best_path.parent.mkdir(parents=True, exist_ok=True)

        # History tracking
        self.train_losses: List[float] = []
        self.val_metrics_history: List[dict] = []
        self.val_aucs: List[float] = []

        logger.info(f"Trainer initialized. Best model checkpoint: {self.best_path.name}")

    def train(self) -> Tuple[Path, List[float], List[dict]]:
        """
        Executes the main training loop with checkpointing and early stopping.
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

        logger.info(
            f"Training finished. Peak Performance -> AUC: {self.best_auc:.4f} | Acc: {self.best_acc:.4f}"
        )

        if self.best_path.exists():
            self.load_best_weights()
        else:
            logger.warning("Forcing checkpoint save for smoke test integrity.")
            torch.save(self.model.state_dict(), self.best_path)

        return self.best_path, self.train_losses, self.val_metrics_history

    def _smart_step_scheduler(self, val_loss: float) -> None:
        """
        Updates the learning rate scheduler based on its type.

        If the scheduler is an instance of ReduceLROnPlateau, it requires
        the validation loss to determine the next step. Otherwise, it
        performs a standard step.

        Args:
            val_loss (float): Current epoch's validation loss.
        """
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        else:
            if not isinstance(self.scaler, torch.amp.GradScaler):
                self.scheduler.step()

    def _handle_checkpointing(self, val_metrics: dict) -> bool:
        """
        Manages model checkpointing and tracks early stopping progress.

        Saves the model state if the current validation accuracy exceeds
        the previous best. Increments the patience counter otherwise.

        Args:
            val_acc (float): Current epoch's validation accuracy.

        Returns:
            bool: True if early stopping criteria are met, False otherwise.
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
        else:
            self.epochs_no_improve += 1

        return self.epochs_no_improve >= self.patience

    def load_best_weights(self) -> None:
        """
        Restores the optimal parameters into the model from the best checkpoint.

        After the training loop completes or is interrupted by early stopping,
        the model instance in memory retains the weights from the final epoch.
        This method reloads the 'best' state-dict saved during the execution
        to ensure subsequent evaluations or inference tasks use the top-performing
        model iteration.

        The restoration process is device-aware, ensuring weights are mapped
        correctly to the active compute device (CUDA/MPS/CPU).
        """
        try:
            load_model_weights(model=self.model, path=self.best_path, device=self.device)
            logger.info(
                f" » [LIFECYCLE] Success: Model state restored to best checkpoint ({self.best_path.name})"
            )
        except Exception as e:
            logger.error(f" » [LIFECYCLE] Critical failure during weight restoration: {e}")
            raise e
