"""
Model Trainer Module

This module defines the central ModelTrainer class. The Trainer encapsulates 
the entire training lifecycle, including optimization, learning rate 
scheduling (Cosine Annealing followed by ReduceLROnPlateau), validation, 
checkpointing, and early stopping.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
from typing import Tuple, List
import logging
from pathlib import Path

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from scripts.core import Config
from .engine import train_one_epoch, validate_epoch

# =========================================================================== #
#                                TRAINING LOGIC                               #
# =========================================================================== #
# Global logger instance
logger = logging.getLogger("medmnist_pipeline")

class ModelTrainer:
    """
    Encapsulates the core training, validation, and scheduling logic for the model.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        cfg: Config,
        output_dir: Path | None = None,
    ):
        """
        Initializes the ModelTrainer with model, data loaders, optimizer,
        schedulers, and early stopping parameters.

        Args:
            model (nn.Module): The PyTorch model to be trained.
            train_loader (DataLoader): DataLoader for the training data.
            val_loader (DataLoader): DataLoader for the validation data.
            device (torch.device): The device (CPU/CUDA) to run the training on.
            cfg (Config): Configuration object containing hyperparameters.
            output_dir (Path | None, optional): Directory to save the best model checkpoint.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = cfg.epochs
        self.patience = cfg.patience
        self.mixup_alpha = cfg.mixup_alpha
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay
        )
        
        # Schedulers
        cosine_epochs = int(0.5 * self.epochs)
        self.cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cosine_epochs,
            eta_min=1e-4
        )
        self.plateau_scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.3,
            patience=1,
            threshold=1e-4,
            cooldown=0,
            min_lr=1e-5,
        )

        # Early Stopping and Checkpointing
        self.best_acc: float = 0.0
        self.epochs_no_improve: int = 0
        
        model_filename = f"best_model_{cfg.model_name.lower().replace(' ', '_')}.pth"
        if output_dir:
            self.best_path = output_dir / model_filename
        else:
            self.best_path = Path(model_filename)
        
        # History tracking
        self.train_losses: List[float] = []
        self.val_accuracies: List[float] = []

        logger.info(f"Trainer initialized. Best model will be saved to: {self.best_path}")
        
    def train(self) -> Tuple[Path, List[float], List[float]]:
        """
        The main training loop. Iterates through epochs, performs training and
        validation, handles learning rate scheduling, and applies early stopping.

        Returns:
            Tuple[Path, List[float], List[float]]:
                The path to the best saved model checkpoint, the list of training
                losses, and the list of validation accuracies.
        """
        for epoch in range(1, self.epochs + 1):
            logger.info(f"Epoch {epoch:02d}/{self.epochs}".center(60, "-"))
                
            # Perform training pass
            epoch_loss = train_one_epoch(
                self.model, self.train_loader, self.criterion, 
                self.optimizer, self.device, epoch, self.epochs, self.mixup_alpha
            )
            self.train_losses.append(epoch_loss)

            # Perform validation pass
            val_acc = validate_epoch(self.model, self.val_loader, self.device)
            self.val_accuracies.append(val_acc)

            # Learning Rate Scheduling: Cosine Annealing first, then Plateau
            cosine_epochs = int(0.6 * self.epochs)
            if epoch <= cosine_epochs: 
                self.cosine_scheduler.step()
            else:
                self.plateau_scheduler.step(val_acc)

            # Checkpoint & Early Stopping Logic
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.best_path)
                logger.info(f"New best model! Val Acc: {val_acc:.4f} ↑ Checkpoint saved.")
            else:
                self.epochs_no_improve += 1
            
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch:02d} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"Best: {self.best_acc:.4f} | LR: {current_lr:.6f} | "
                f"Patience Left: {self.patience - self.epochs_no_improve}"
            )

            if self.epochs_no_improve >= self.patience:
                logger.info(f"Early stopping triggered at epoch {epoch} due to lack of improvement.")
                break
            
        logger.info(f"Training finished. Best validation accuracy: {self.best_acc:.4f}")
        return self.best_path, self.train_losses, self.val_accuracies