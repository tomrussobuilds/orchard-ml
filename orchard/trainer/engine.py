"""
Core Training and Validation Engines.

Stateless, single-epoch execution kernels consumed by ``ModelTrainer``.
Each function accepts fully-resolved objects (model, loader, criterion,
device) and returns plain Python values — no side-effects on global state.

Features:

- AMP Integration: ``torch.autocast`` + ``GradScaler`` for mixed precision,
  with automatic device-type resolution (CUDA/CPU).
- Gradient Clipping: Per-batch ``clip_grad_norm_`` applied after unscaling
  when AMP is active, preventing gradient explosions.
- MixUp Augmentation: Beta-distribution blending (``mixup_data``) with
  seeded NumPy generator for reproducible regularization.
- Divergence Guard: ``train_one_epoch`` raises ``RuntimeError`` on
  NaN/Inf loss to prevent checkpointing corrupted weights.

Key Functions:
    compute_auc: Macro-averaged ROC-AUC with graceful fallback.
    train_one_epoch: Single training pass with AMP, MixUp, and tqdm progress.
    validate_epoch: No-grad evaluation returning loss, accuracy, macro AUC, and macro F1.
    mixup_data: Convex sample blending for data augmentation.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from tqdm.auto import tqdm

from ..core import LOGGER_NAME
from ..core.paths import METRIC_ACCURACY, METRIC_AUC, METRIC_F1, METRIC_LOSS

# Module-level logger (avoid dynamic imports in exception handlers)
logger = logging.getLogger(LOGGER_NAME)


# AUC COMPUTATION (shared by trainer.engine and evaluation.metrics)
def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute macro-averaged ROC-AUC with graceful fallback.

    Handles binary (positive class probability) and multiclass (OvR)
    cases. Returns 0.0 on failure or NaN.

    Args:
        y_true: Ground truth class indices, shape ``(N,)``.
        y_score: Probability distributions, shape ``(N, C)`` (softmax output).

    Returns:
        ROC-AUC score, or 0.0 if computation fails.
    """
    try:
        n_classes = y_score.shape[1] if y_score.ndim == 2 else 1
        if n_classes <= 2:
            auc = roc_auc_score(y_true, y_score[:, 1] if y_score.ndim == 2 else y_score)
        else:
            auc = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
    except (ValueError, TypeError, IndexError) as e:
        logger.warning(f"ROC-AUC calculation failed: {e}. Defaulting to 0.0")
        return 0.0

    if np.isnan(auc):
        return 0.0
    return float(auc)


# TRAINING ENGINE
def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mixup_fn: Callable | None = None,
    scaler: torch.amp.grad_scaler.GradScaler | None = None,
    grad_clip: float | None = 0.0,
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

    # Resolve autocast device type for AMP
    amp_enabled = scaler is not None
    amp_device_type = device.type if amp_enabled else "cpu"

    # Training loop - iterate directly without enumerate
    for inputs, targets in iterator:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # Forward pass with optional AMP autocast
        with torch.autocast(device_type=amp_device_type, enabled=amp_enabled):
            # Apply MixUp if enabled
            if mixup_fn:
                inputs, y_a, y_b, lam = mixup_fn(inputs, targets)
                outputs = model(inputs)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

        # Guard: halt on diverged loss to prevent saving corrupted weights
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(
                f"Training diverged: loss={loss.item()} at epoch {epoch}. "
                "Check learning rate, data preprocessing, or enable gradient clipping."
            )

        # Backward pass with optional AMP and gradient clipping
        _backward_step(loss, optimizer, model, scaler, grad_clip)

        # Accumulate loss
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        # Update progress bar with current loss
        if use_tqdm:
            iterator.set_postfix({"loss": f"{loss.item():.4f}"})

    # Handle empty training set (defensive guard)
    if total_samples == 0:
        logger.warning("Empty training set: no samples processed. Returning zero loss.")
        return 0.0

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
    robust performance estimation on potentially imbalanced datasets.

    Args:
        model: Neural network model to evaluate
        val_loader: Validation data provider
        criterion: Loss function (e.g., CrossEntropyLoss)
        device: Hardware target (CUDA/MPS/CPU)

    Returns:
        Validation metrics dict with keys:

        - ``loss``: Average cross-entropy loss
        - ``accuracy``: Classification accuracy [0.0, 1.0]
        - ``auc``: Macro-averaged Area Under the ROC Curve
        - ``f1``: Macro-averaged F1 score
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

    # Handle empty validation set (defensive guard)
    if total == 0 or len(all_targets) == 0:
        logger.warning("Empty validation set: no samples processed. Returning zero metrics.")
        return {METRIC_LOSS: 0.0, METRIC_ACCURACY: 0.0, METRIC_AUC: 0.0, METRIC_F1: 0.0}

    # Global metric computation
    y_true = torch.cat(all_targets).numpy()
    y_score = torch.cat(all_probs).numpy()
    y_pred = y_score.argmax(axis=1)

    auc = compute_auc(y_true, y_score)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))

    return {
        METRIC_LOSS: val_loss / total,
        METRIC_ACCURACY: correct / total,
        METRIC_AUC: auc,
        METRIC_F1: macro_f1,
    }


# MIXUP UTILITY
def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Applies MixUp augmentation by blending two random samples.

    MixUp generates convex combinations of training pairs to improve
    generalization and calibration.

    Args:
        x: Input data batch (images)
        y: Target labels batch
        alpha: Beta distribution parameter (0 disables MixUp)
        rng: NumPy random generator for reproducibility (seeded from config)

    Returns:
        4-tuple of (mixed_x, y_a, y_b, lam).
    """
    if alpha <= 0:
        return x, y, y, 1.0

    if rng is None:
        # Defensive fallback — production path always provides a seeded rng
        # via ModelTrainer (seeded from cfg.training.seed).
        rng = np.random.default_rng(seed=42)

    # Draw mixing coefficient from Beta distribution
    lam: float = float(rng.beta(alpha, alpha))
    batch_size: int = x.size(0)

    # Generate random permutation (device-aware: CUDA and MPS)
    # GPU branch excluded from coverage — CI runs CPU-only, covered in local GPU testing
    index = torch.randperm(batch_size)
    if x.is_cuda or x.is_mps:  # pragma: no cover
        index = index.to(x.device)

    # Create mixed input
    mixed_x: torch.Tensor = lam * x + (1 - lam) * x[index]

    # Get corresponding targets
    y_a: torch.Tensor = y
    y_b: torch.Tensor = y[index]

    return mixed_x, y_a, y_b, lam


# INTERNAL HELPERS
def _backward_step(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    scaler: torch.amp.grad_scaler.GradScaler | None,
    grad_clip: float | None,
) -> None:
    """
    Perform a backward pass with optional gradient scaling and clipping.

    Handles mixed precision training via a GradScaler if provided and
    applies gradient clipping when ``grad_clip`` is greater than zero.

    Args:
        loss: Computed loss for the current batch.
        optimizer: Optimizer used to update model parameters.
        model: Neural network whose parameters will be updated.
        scaler: AMP scaler. If ``None``, standard precision backward pass is used.
        grad_clip: Maximum norm for gradient clipping.
            If ``None`` or ``<= 0``, no clipping is applied.
    """
    if scaler:
        # Mixed precision backward pass
        scaler.scale(loss).backward()
        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        # Standard backward pass
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
