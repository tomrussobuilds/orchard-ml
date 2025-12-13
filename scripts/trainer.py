"""
Training Utilities and Trainer Module

This module provides implementations for the MixUp data augmentation technique
and the central ModelTrainer class. The Trainer encapsulates the entire training
lifecycle, including optimization, learning rate scheduling (Cosine Annealing
followed by ReduceLROnPlateau), validation, checkpointing, and early stopping.
"""
# =========================================================================== #
#                                Standard Imports
# =========================================================================== #
from typing import Tuple, List, Final
import logging
from pathlib import Path

# =========================================================================== #
#                                Third-Party Imports
# =========================================================================== #
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
# =========================================================================== #
#                                Internal Imports
# =========================================================================== #
from .utils import Logger, Config, MODELS_DIR

# Global logger instance
logger: Final[logging.Logger] = Logger().get_logger()


# =========================================================================== #
#                               MIXUP UTILITY
# =========================================================================== #

def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Applies MixUp augmentation, generating a convex combination of two random
    samples and their corresponding targets.

    Args:
        x (torch.Tensor): Input data batch (images).
        y (torch.Tensor): Target labels batch.
        alpha (float): Beta distribution parameter (set to 0 to disable MixUp).
        device (torch.device | None): The device where tensors should be placed.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
            mixed_x: The blended input images.
            y_a: The original targets.
            y_b: The permuted targets.
            lam: The mixing coefficient $\\lambda$.
    """
    if alpha <= 0:
        return x, y, y, 1.0

    # Draw mixing coefficient $\lambda$ from Beta distribution
    lam: float = np.random.beta(alpha, alpha)
    batch_size: int = x.size(0)
    
    # Generate a random permutation of indices
    if device is not None:
        index = torch.randperm(batch_size, device=device)
    else:
        index = torch.randperm(batch_size).to(x.device)

    # Calculate the mixed input
    mixed_x: torch.Tensor = lam * x + (1 - lam) * x[index, :]
    
    # Get the corresponding targets
    y_a: torch.Tensor = y
    y_b: torch.Tensor = y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    Calculates the MixUp loss as a weighted average of the loss for the two targets.

    Args:
        criterion (nn.Module): The standard loss function (e.g., CrossEntropyLoss).
        pred (torch.Tensor): Model predictions for the mixed input.
        y_a (torch.Tensor): The original targets.
        y_b (torch.Tensor): The permuted targets.
        lam (float): The mixing coefficient $\\lambda$.

    Returns:
        torch.Tensor: The final MixUp-regularized loss value.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =========================================================================== #
#                               TRAINING LOGIC
# =========================================================================== #

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
        config: Config,
        best_path: Path | None = None,
    ):
        """
        Initializes the ModelTrainer with model, data loaders, optimizer,
        schedulers, and early stopping parameters.

        Args:
            model (nn.Module): The PyTorch model to be trained.
            train_loader (DataLoader): DataLoader for the training data.
            val_loader (DataLoader): DataLoader for the validation data.
            device (torch.device): The device (CPU/CUDA) to run the training on.
            config (Config): Configuration object containing hyperparameters.
            best_path (Path | None, optional): Path to save the best model checkpoint.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = config.epochs
        self.patience = config.patience
        self.mixup_alpha = config.mixup_alpha
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
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
        self.best_path: Path = best_path or (MODELS_DIR / "resnet18_bloodmnist_best.pth")
        
        # History tracking
        self.train_losses: List[float] = []
        self.val_accuracies: List[float] = []

        logger.info(f"Trainer initialized. Best model will be saved to: {self.best_path}")
        
    def _validate_epoch(self) -> float:
        """
        Performs a full validation cycle on the validation set.

        Returns:
            float: The computed validation accuracy.
        """
        self.model.eval()
        correct: int = 0
        total: int = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        return correct / total
    
    def _train_epoch(self, epoch: int) -> float:
        """
        Performs a single training cycle over the training set, applying MixUp
        based on the epoch number.

        Args:
            epoch (int): The current epoch number.

        Returns:
            float: The average training loss for the epoch.
        """
        self.model.train()
        running_loss: float = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Training", leave=False)
        
        # Gradually disable MixUp after 80% of epochs
        alpha = self.mixup_alpha
        if epoch > int(0.5 * self.epochs):
            alpha = 0.0
        
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            if alpha > 0:
                # Apply MixUp
                inputs, targets_a, targets_b, lam = mixup_data(
                    inputs, targets, alpha, self.device
                )
                outputs = self.model(inputs)
                loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
            else:
                # Standard training
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update running loss and progress bar
            running_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix(
                {"loss": f"{running_loss / ((progress_bar.n + 1) * inputs.size(0)):.4f}"}
            )

        # Return average loss per sample
        return running_loss / len(self.train_loader.dataset)
        
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
                
            epoch_loss = self._train_epoch(epoch)
            self.train_losses.append(epoch_loss)

            # Validation
            val_acc = self._validate_epoch()
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