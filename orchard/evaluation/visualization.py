"""
Visualization Utilities Module

This module provides functions for generating visual reports of the model's
performance, including training loss/accuracy curves, normalized confusion
matrices, and sample prediction grids. It is fully integrated with the
Pydantic Configuration Engine for aesthetic and technical consistency.
"""

import logging
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader

from orchard.core import LOGGER_NAME, Config

# Global logger instance
logger = logging.getLogger(LOGGER_NAME)


# PUBLIC INTERFACE
def show_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    classes: List[str],
    save_path: Path | None = None,
    cfg: Config | None = None,
    n: int | None = None,
) -> None:
    """
    Orchestrates the visualization of model predictions on a sample batch.

    This function coordinates data extraction, model inference, grid layout
    generation, and image post-processing. Highlights correct (green)
    vs. incorrect (red) predictions.
    """
    model.eval()

    # 1. Parameter Resolution & Batch Inference
    num_samples = n or (cfg.evaluation.n_samples if cfg else 12)
    images, labels, preds = _get_predictions_batch(model, loader, device, num_samples)

    # 2. Grid & Figure Setup
    grid_cols = cfg.evaluation.grid_cols if cfg else 4
    _, axes = _setup_prediction_grid(len(images), grid_cols, cfg)

    # 3. Plotting Loop
    for i, ax in enumerate(axes):
        if i < len(images):
            img = _denormalize_image(images[i], cfg) if cfg else images[i]
            display_img = _prepare_for_plt(img)

            ax.imshow(
                display_img,
                cmap="gray" if display_img.ndim == 2 else None,
            )

            is_correct = labels[i] == preds[i]
            ax.set_title(
                f"T:{classes[labels[i]]}\nP:{classes[preds[i]]}",
                color="green" if is_correct else "red",
                fontsize=9,
            )

        ax.axis("off")

    tta_info = f" | TTA: {'ON' if cfg.training.use_tta else 'OFF'}" if cfg else ""

    domain_info = ""
    if cfg:
        if cfg.dataset.metadata.is_texture_based:
            domain_info = " | Mode: Texture"
        elif cfg.dataset.metadata.is_anatomical:
            domain_info = " | Mode: Anatomical"
        else:
            domain_info = " | Mode: Standard"

    if cfg:
        plt.suptitle(
            f"Sample Predictions — {cfg.model.name} | "
            f"Resolution: {cfg.dataset.resolution}"
            f"{domain_info}{tta_info}",
            fontsize=14,
        )

    # 4. Export and Cleanup
    _finalize_figure(plt, save_path, cfg)


def plot_training_curves(
    train_losses: Sequence[float], val_accuracies: Sequence[float], out_path: Path, cfg: Config
) -> None:
    """
    Plots training loss and validation accuracy curves on a dual-axis plot.
    Automatically saves raw numerical data to .npz for reproducibility.
    """
    fig, ax1 = plt.subplots(figsize=(9, 6))

    # Left Axis: Training Loss
    ax1.plot(train_losses, color="#e74c3c", lw=2, label="Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="#e74c3c", fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#e74c3c")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Right Axis: Validation Accuracy
    ax2 = ax1.twinx()
    ax2.plot(val_accuracies, color="#3498db", lw=2, label="Validation Accuracy")
    ax2.set_ylabel("Accuracy", color="#3498db", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#3498db")

    fig.suptitle(
        f"Training Metrics — {cfg.model.name} | Resolution — {cfg.dataset.resolution}",
        fontsize=14,
        y=1.02,
    )

    fig.tight_layout()

    plt.savefig(out_path, dpi=cfg.evaluation.fig_dpi, bbox_inches="tight")
    logger.info(f"Training curves saved → {out_path.name}")

    # Export raw data for post-run analysis
    npz_path = out_path.with_suffix(".npz")
    np.savez(npz_path, train_losses=train_losses, val_accuracies=val_accuracies)
    plt.close()


def plot_confusion_matrix(
    all_labels: np.ndarray, all_preds: np.ndarray, classes: List[str], out_path: Path, cfg: Config
) -> None:
    """
    Generates and saves a normalized confusion matrix plot.
    """
    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(len(classes)), normalize="true")
    cm = np.nan_to_num(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(11, 9))

    disp.plot(ax=ax, cmap=cfg.evaluation.cmap_confusion, xticks_rotation=45, values_format=".3f")
    plt.title(
        f"Confusion Matrix — {cfg.model.name} | Resolution — {cfg.dataset.resolution}",
        fontsize=12,
        pad=20,
    )

    plt.tight_layout()

    fig.savefig(out_path, dpi=cfg.evaluation.fig_dpi, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved → {out_path.name}")


def _get_predictions_batch(model: nn.Module, loader: DataLoader, device: torch.device, n: int):
    """Handles data extraction and model forward pass for a small sample."""
    batch = next(iter(loader))
    images_tensor = batch[0][:n].to(device)
    labels_tensor = batch[1][:n]

    with torch.no_grad():
        outputs = model(images_tensor)
        preds = outputs.argmax(dim=1).cpu().numpy()

    return images_tensor.cpu().numpy(), labels_tensor.numpy().flatten(), preds


def _setup_prediction_grid(num_samples: int, cols: int, cfg: Config | None):
    """Calculates grid dimensions and initializes matplotlib subplots."""
    rows = int(np.ceil(num_samples / cols))
    base_w, base_h = cfg.evaluation.fig_size_predictions if cfg else (12, 8)

    fig, axes = plt.subplots(
        rows, cols, figsize=(base_w, (base_h / 3) * rows), constrained_layout=True
    )
    # Ensure axes is always an array even for 1x1 grids
    return fig, np.atleast_1d(axes).flatten()


def _finalize_figure(plt_obj, save_path: Path | None, cfg: Config | None):
    """Handles saving to disk and cleaning up memory."""
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        dpi = cfg.evaluation.fig_dpi if cfg else 200
        plt_obj.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        logger.info(f"Predictions grid saved → {save_path.name}")
    else:
        plt_obj.show()
        logger.debug("Displaying figure interactive mode")

    plt_obj.close()


def _denormalize_image(img: np.ndarray, cfg: Config) -> np.ndarray:
    """Reverses the normalization transform using dataset-specific stats."""
    mean = np.array(cfg.dataset.mean).reshape(-1, 1, 1)
    std = np.array(cfg.dataset.std).reshape(-1, 1, 1)
    img = (img * std) + mean
    return np.clip(img, 0, 1)


def _prepare_for_plt(img: np.ndarray) -> np.ndarray:
    """Converts a deep learning tensor (C, H, W) to (H, W, C) for Matplotlib."""
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))

    if img.ndim == 3 and img.shape[-1] == 1:
        img = img.squeeze(-1)

    return img
