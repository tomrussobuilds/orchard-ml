"""
Core Training and Validation Engines.

High-performance implementation of training and validation loops with modern
PyTorch features: Automatic Mixed Precision (AMP), Gradient Clipping, and
MixUp augmentation for improved numerical stability and hardware utilization.
"""

# Standard Imports
from typing import Tuple

# Third-Party Imports
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm


# TRAINING ENGINE
def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mixup_fn=None,
    scaler=None,
    grad_clip: float = 0.0,
    epoch: int = 0,
    total_epochs: int = 1,
    use_tqdm: bool = True,
) -> float:
    """
    Performs a single full pass over the training dataset.

    Args:
        model: Neural network architecture to train
        loader: Training data provider
        criterion: Loss function
        optimizer: Gradient descent optimizer
        device: Hardware target (CUDA/MPS/CPU)
        mixup_fn: Function to apply MixUp data blending (optional)
        scaler: PyTorch GradScaler for mixed precision training (optional)
        grad_clip: Max norm for gradient clipping (0 disables)
        epoch: Current epoch index for progress bar
        total_epochs: Total number of epochs (for progress bar)
        use_tqdm: Show progress bar during training

    Returns:
        Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    # Create iterator with or without progress bar
    if use_tqdm:
        iterator = tqdm(loader, desc=f"Train Epoch {epoch}/{total_epochs}", leave=True, ncols=100)
    else:
        iterator = loader

    # Training loop - iterate directly without enumerate
    for inputs, targets in iterator:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # Apply MixUp if enabled
        if mixup_fn:
            inputs, y_a, y_b, lam = mixup_fn(inputs, targets)
            outputs = model(inputs)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # Backward pass with optional AMP and gradient clipping
        if scaler:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # Accumulate loss
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        # Update progress bar with current loss
        if use_tqdm:
            iterator.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / total_samples


# VALIDATION ENGINE
def validate_epoch(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Evaluates model performance on held-out validation set.

    Computes validation loss, accuracy, and ROC-AUC score under no_grad context.
    AUC calculated using One-vs-Rest (OvR) strategy with macro-averaging for
    robust performance estimation on potentially imbalanced MedMNIST datasets.

    Args:
        model: Neural network model to evaluate
        val_loader: Validation data provider
        criterion: Loss function (e.g., CrossEntropyLoss)
        device: Hardware target (CUDA/MPS/CPU)

    Returns:
        dict: Validation metrics
            - 'loss' (float): Average cross-entropy loss
            - 'accuracy' (float): Classification accuracy [0.0, 1.0]
            - 'auc' (float): Macro-averaged Area Under the ROC Curve
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    # Buffers for global metrics (CPU to save VRAM)
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Collect probabilities for AUC (move to CPU to save VRAM)
            probs = torch.softmax(outputs, dim=1)
            all_targets.append(targets.cpu())
            all_probs.append(probs.cpu())

            # Loss computation
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

            # Accuracy computation
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    # Global metric computation
    y_true = torch.cat(all_targets).numpy()
    y_score = torch.cat(all_probs).numpy()

    num_classes = y_score.shape[1]
    np.arange(num_classes)

    # Compute ROC-AUC
    try:
        if num_classes == 2:
            # Binary classification: use only positive class probabilities
            auc = roc_auc_score(y_true, y_score[:, 1])
        else:
            # Multi-class: use macro-averaged One-vs-Rest
            auc = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
    except (ValueError, IndexError) as e:
        # Log error for debugging
        import logging

        logger = logging.getLogger("VisionForge")
        logger.warning(f"AUC calculation failed: {e}. Setting auc=0.0")
        auc = 0.0

    # Ensure we always return a dict
    return {"loss": val_loss / total, "accuracy": correct / total, "auc": auc}


# MIXUP UTILITY
def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Applies MixUp augmentation by blending two random samples.

    MixUp generates convex combinations of training pairs to improve
    generalization and calibration. Particularly effective for small
    medical imaging datasets.

    Args:
        x: Input data batch (images)
        y: Target labels batch
        alpha: Beta distribution parameter (0 disables MixUp)
        device: Device for tensor placement (unused, kept for compatibility)

    Returns:
        Tuple containing:
            - mixed_x: Blended input images
            - y_a: Original targets
            - y_b: Permuted targets
            - lam: Mixing coefficient lambda
    """
    if alpha <= 0:
        return x, y, y, 1.0

    # Draw mixing coefficient from Beta distribution
    lam: float = np.random.beta(alpha, alpha)
    batch_size: int = x.size(0)

    # Generate random permutation (device-aware)
    index = torch.randperm(batch_size)
    if x.is_cuda:  # pragma: no cover
        index = index.to(x.device)

    # Create mixed input
    mixed_x: torch.Tensor = lam * x + (1 - lam) * x[index, :]

    # Get corresponding targets
    y_a: torch.Tensor = y
    y_b: torch.Tensor = y[index]

    return mixed_x, y_a, y_b, lam
