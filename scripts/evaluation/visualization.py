"""
Visualization Utilities Module

This module provides functions for generating visual reports of the model's 
performance, including training loss/accuracy curves, normalized confusion 
matrices, and sample prediction grids with true vs. predicted labels.
"""

# =========================================================================== #
#                                Standard Imports
# =========================================================================== #
from typing import Sequence, List
from pathlib import Path
import logging

# =========================================================================== #
#                                Third-Party Imports
# =========================================================================== #
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for plotting
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =========================================================================== #
#                                Internal Imports
# =========================================================================== #
from scripts.core import Config

# =========================================================================== #
#                                CONFIGURATION
# =========================================================================== #
# Set a professional style for all plots
try:
    plt.style.use('seaborn-v0_8-muted')
except OSError:
    plt.style.use('ggplot') # Fallback style

# Global logger instance
logger = logging.getLogger("medmnist_pipeline")

# =========================================================================== #
#                               VISUALIZATION FUNCTIONS
# =========================================================================== #

def show_predictions(images: np.ndarray,
                     true_labels: np.ndarray,
                     preds: np.ndarray,
                     classes: List[str],
                     n: int = 12,
                     save_path: Path | None = None,
                     cfg: Config | None = None
) -> None:
    """
    Displays a grid of randomly selected test images with their true and
    predicted labels, highlighting correct vs. incorrect predictions.

    Args:
        images (np.ndarray): The array of test images.
        true_labels (np.ndarray): The array of true labels for the test set.
        preds (np.ndarray): The array of model predictions for the test set.
        classes (List[str]): List of class names for labeling.
        n (int): The number of samples to display (must be multiple of 4).
        save_path (Path | None): Path to save the figure. If None, the plot is shown.
        cfg (Config | None): Configuration object for title metadata.
    """
    if images is None or len(images) == 0:
        logger.warning(
            "No images available for show_predictions. Skipping plot."
        )
        return
    
    if n > len(images):
        n = len(images)
    
    rows = int(np.ceil(n / 4))
    cols = 4

    plt.figure(figsize=(12, 3 * rows))
    indices = np.arange(n)

    for i, idx in enumerate(indices):
        # Initialize img variable from the input array
        img = images[idx].copy()

        # Handle Denormalization if Config is provided
        if cfg and hasattr(cfg, 'dataset'):
            mean = np.array(cfg.dataset.mean)
            std = np.array(cfg.dataset.std)

            if img.shape[-1] == 3:
                mean = mean.reshape(1, 1, -1)
                std = std.reshape(1, 1, -1)
            else:
                mean = mean.reshape(-1, 1, 1)
                std = std.reshape(-1, 1, 1)

            img = (img * std) + mean
            img = np.clip(img, 0, 1)

        # Transpose from (C, H, W) to (H, W, C) for matplotlib compatibility
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))

        true_label = int(true_labels[idx])
        pred_label = int(preds[idx])

        plt.subplot(rows, cols, i+1)
        
        # Support for grayscale (H, W) or RGB (H, W, C)
        if img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1):
            plt.imshow(img.squeeze(), cmap='gray')
        else:
            plt.imshow(img)
            
        color = "green" if true_label == pred_label else "red"
        
        plt.title(
            f"T:{classes[true_label]}\nP:{classes[pred_label]}",
            color=color, fontsize=10
        )
        plt.axis("off")
    
    model_title = cfg.model_name if cfg else "Model"
    plt.suptitle(f"Test Predictions — {model_title}", fontsize=16)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
            facecolor="white"
        )
        logger.info(f"Sample predictions saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_training_curves(
        train_losses: Sequence[float],
        val_accuracies: Sequence[float],
        out_path: Path | None = None,
        save_npz: bool = True,
        cfg: Config | None = None,
        out_dir: Path | None = None
) -> None:
    """
    Plots the training loss and validation accuracy curves on a dual-axis plot
    and optionally saves the raw numerical data.

    Args:
        train_losses (Sequence[float]): List of training losses per epoch.
        val_accuracies (Sequence[float]): List of validation accuracies per epoch.
        out_path (Path): Path to save the generated plot.
        save_npz (bool): Whether to save raw data to a .npz file.
    """
    if out_dir and not out_path:
        out_path = Path(out_dir) / "training_curves.png"
    elif not out_path:
        raise ValueError("Either out_path or out_dir must be provided.")
    
    fig, ax1 = plt.subplots(figsize=(9, 6))

    # Plot Training Loss on the left axis (ax1)
    ax1.plot(train_losses, color='#e74c3c', lw=2, label="Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color='#e74c3c', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#e74c3c')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # Plot Validation Accuracy on the right axis (ax2)
    ax2 = ax1.twinx()
    ax2.plot(val_accuracies, color='#3498db', lw=2, label="Validation Accuracy")
    ax2.set_ylabel("Accuracy", color='#3498db', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#3498db')

    plt.title("Training Loss & Validation Accuracy", fontsize=14, pad=15)
    fig.tight_layout()
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    logger.info(f"Training curves saved → {out_path}")

    if save_npz:
        npz_path = out_path.with_suffix('.npz')
        np.savez(
            npz_path,
            train_losses=train_losses,
            val_accuracies=val_accuracies
        )
        logger.info(f"Raw training data saved → {npz_path}")

    plt.close()

def plot_confusion_matrix(
        all_labels: np.ndarray,
        all_preds: np.ndarray,
        classes: List[str],
        out_path: Path,
        cfg: Config | None = None
) -> None:
    """
    Generates and saves a normalized confusion matrix plot.

    Args:
        all_labels (np.ndarray): Array of true labels.
        all_preds (np.ndarray): Array of predicted labels.
        classes (List[str]): List of class names for labeling.
        out_path (Path): Path to save the generated plot.
        cfg (Config | None): Configuration object for title metadata.
    """
    # Calculate the normalized confusion matrix (rows sum to 1)
    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=np.arange(len(classes)),
        normalize='true',
    )

    cm = np.nan_to_num(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes,
    )

    fig, ax = plt.subplots(figsize=(11, 9))
    disp.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=False,
        values_format='.3f'
    )

    model_name = cfg.model_name if cfg else "Model"
    plt.title(f"Confusion Matrix – {model_name}", fontsize=14, pad=20)
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved → {out_path}")