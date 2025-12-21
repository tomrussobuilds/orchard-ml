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
#                               VISUALIZATION FUNCTIONS
# =========================================================================== #
# Global logger instance
logger = logging.getLogger("medmnist_pipeline")

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
    # Ensure n is a multiple of 4 for a clean 3x4 grid or similar
    if n > len(images):
        n = len(images)
    
    rows = int(np.ceil(n / 4))
    cols = 4

    plt.figure(figsize=(12, 3 * rows))
    # Randomly select N indices from the test set
    indices = np.random.choice(len(images), n, replace=False)

    for i, idx in enumerate(indices):
        img = images[idx]
        true_label = int(true_labels[idx])
        pred_label = int(preds[idx])

        plt.subplot(rows, cols, i+1)
        plt.imshow(img)
        color = "green" if true_label == pred_label else "red"
        
        plt.title(
            f"T:{classes[true_label]}\nP:{classes[pred_label]}",
            color=color, fontsize=10
        )
        plt.axis("off")
    
    model_title = cfg.model_name if cfg else "Model"
    dataset_title = cfg.dataset_name if (cfg and hasattr(cfg, 'dataset_name')) else "Dataset"

    plt.suptitle(f"Test Predictions — {model_title} on {dataset_title}", fontsize=16)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        logger.info(f"Sample predictions saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_training_curves(
        train_losses: Sequence[float],
        val_accuracies: Sequence[float],
        out_path: Path,
        cfg: Config | None = None
) -> None:
    """
    Plots the training loss and validation accuracy curves on a dual-axis plot.

    Args:
        train_losses (Sequence[float]): List of training losses per epoch.
        val_accuracies (Sequence[float]): List of validation accuracies per epoch.
        out_path (Path): Path to save the generated plot.
    """
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot Training Loss on the left axis (ax1)
    ax1.plot(train_losses, 'r-', label="Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot Validation Accuracy on the right axis (ax2)
    ax2 = ax1.twinx()
    ax2.plot(val_accuracies, 'b-', label="Validation Accuracy")
    ax2.set_ylabel("Accuracy", color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    plt.title("Training Loss & Validation Accuracy", fontsize=14)
    fig.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
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
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes,
    )

    fig, ax = plt.subplots(figsize=(11, 9))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False, values_format='.3f')

    plt.title(f"Confusion Matrix – {cfg.model_name} on {cfg.dataset_name}", fontsize=14, pad=20)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved → {out_path}")

def save_training_curves(
        train_losses: Sequence[float],
        val_accuracies: Sequence[float],
        out_dir: Path,
        cfg: Config | None = None
    ) -> None:
    """Plots and saves training curves and their raw data to disk."""
    plot_training_curves(train_losses, val_accuracies, out_dir / "training_curves.png")

    # Save raw data for later analysis
    np.savez(
        out_dir / "training_curves.npz",
        train_losses=train_losses,
        val_accuracies=val_accuracies,
    )
    logger.info(f"Training curves data saved → {out_dir / 'training_curves.npz'}")

def save_sample_predictions(
        images: np.ndarray,
        true_labels: np.ndarray,
        classes: List[str],
        all_preds: np.ndarray,
        out_path: Path,
        cfg: Config | None = None
    ) -> None:
    """Generates and saves a figure showing sample predictions."""
    show_predictions(
        images=images,
        true_labels=true_labels,
        classes=classes,
        all_preds=all_preds,
        n=12,
        save_path=out_path,
        cfg=cfg
    )
    logger.info(f"Sample predictions figure saved → {out_path}")