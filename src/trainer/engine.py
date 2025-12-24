"""
Training Utilities and Trainer Module

This module provides implementations for the MixUp data augmentation technique
and the central ModelTrainer class. The Trainer encapsulates the entire training
lifecycle, including optimization, learning rate scheduling (Cosine Annealing
followed by ReduceLROnPlateau), validation, checkpointing, and early stopping.
"""
# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import logging
from typing import Tuple

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from src.core import Config

# =========================================================================== #
#                               MIXUP UTILITY                                 #
# =========================================================================== #
# Global logger instance
logger = logging.getLogger("medmnist_pipeline")

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
            lam: The mixing coefficient $\lambda$.
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
        lam (float): The mixing coefficient $\lambda$.

    Returns:
        torch.Tensor: The final MixUp-regularized loss value.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =========================================================================== #
#                               CORE ENGINES                                  #
# =========================================================================== #

def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    cfg: Config,
) -> float:
    """
    Performs a single training cycle over the training set, applying MixUp
    based on the epoch number.
    """
    model.train()
    running_loss: float = 0.0
    progress_bar = tqdm(train_loader, desc=f"Training", leave=False)
    
    # Determine if MixUp should be applied this epoch
    cosine_limit = int(cfg.training.cosine_fraction * cfg.training.epochs)
    current_alpha = cfg.training.mixup_alpha if epoch <= cosine_limit else 0.0
    
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)

        if current_alpha > 0:
            # Apply MixUp
            inputs, targets_a, targets_b, lam = mixup_data(
                inputs, targets, current_alpha, device
            )
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            # Standard training
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update running loss and progress bar
        running_loss += loss.item() * inputs.size(0)
        progress_bar.set_postfix(
            {"loss": f"{running_loss / ((progress_bar.n + 1) * inputs.size(0)):.4f}"}
        )

    return running_loss / len(train_loader.dataset)


def validate_epoch(
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> float:
    """
    Performs a full validation cycle on the validation set.
    """
    model.eval()
    correct: int = 0
    total: int = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    return correct / total