"""Visualization utilities for model evaluation.

Provides formatted visual reports including training loss/accuracy curves,
normalized confusion matrices, and sample prediction grids. Integrated with
the Pydantic configuration engine for aesthetic and technical consistency.
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

from ..core import LOGGER_NAME, Config

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
    """Visualize model predictions on a sample batch.

    Coordinates data extraction, model inference, grid layout generation,
    and image post-processing. Highlights correct (green) vs. incorrect
    (red) predictions.

    Args:
        model: Trained model to evaluate.
        loader: DataLoader providing evaluation samples.
        device: Target device for inference.
        classes: Human-readable class label names.
        save_path: Output file path. If None, displays interactively.
        cfg: Configuration object for layout and normalization settings.
        n: Number of samples to display. Defaults to ``cfg.evaluation.n_samples``.
    """
    model.eval()
    style = cfg.evaluation.plot_style if cfg else "seaborn-v0_8-muted"

    with plt.style.context(style):
        # 1. Parameter Resolution & Batch Inference
        num_samples = n or (cfg.evaluation.n_samples if cfg else 12)
        images, labels, preds = _get_predictions_batch(model, loader, device, num_samples)

        # 2. Grid & Figure Setup
        grid_cols = cfg.evaluation.grid_cols if cfg else 4
        _, axes = _setup_prediction_grid(len(images), grid_cols, cfg)

        # 3. Plotting Loop
        for i, ax in enumerate(axes):
            if i < len(images):
                _plot_single_prediction(ax, images[i], labels[i], preds[i], classes, cfg)
            ax.axis("off")

        # 4. Suptitle
        if cfg:
            plt.suptitle(_build_suptitle(cfg), fontsize=14)

        # 5. Export and Cleanup
        _finalize_figure(plt, save_path, cfg)


def plot_training_curves(
    train_losses: Sequence[float], val_accuracies: Sequence[float], out_path: Path, cfg: Config
) -> None:
    """Plot training loss and validation accuracy on a dual-axis chart.

    Saves the figure to disk and exports raw numerical data as ``.npz``
    for reproducibility.

    Args:
        train_losses: Per-epoch training loss values.
        val_accuracies: Per-epoch validation accuracy values.
        out_path: Destination file path for the saved figure.
        cfg: Configuration with architecture and evaluation settings.
    """
    with plt.style.context(cfg.evaluation.plot_style):
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
            f"Training Metrics — {cfg.architecture.name} | Resolution — {cfg.dataset.resolution}",
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
    """Generate and save a row-normalized confusion matrix plot.

    Args:
        all_labels: Ground-truth label array.
        all_preds: Predicted label array.
        classes: Human-readable class label names.
        out_path: Destination file path for the saved figure.
        cfg: Configuration with architecture and evaluation settings.
    """
    with plt.style.context(cfg.evaluation.plot_style):
        cm = confusion_matrix(
            all_labels, all_preds, labels=np.arange(len(classes)), normalize="true"
        )
        cm = np.nan_to_num(cm)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        fig, ax = plt.subplots(figsize=(11, 9))

        disp.plot(
            ax=ax, cmap=cfg.evaluation.cmap_confusion, xticks_rotation=45, values_format=".3f"
        )
        plt.title(
            f"Confusion Matrix — {cfg.architecture.name} | Resolution — {cfg.dataset.resolution}",
            fontsize=12,
            pad=20,
        )

        plt.tight_layout()

        fig.savefig(out_path, dpi=cfg.evaluation.fig_dpi, bbox_inches="tight")
        plt.close()
        logger.info(f"Confusion matrix saved → {out_path.name}")


def _plot_single_prediction(
    ax,
    image: np.ndarray,
    label: int,
    pred: int,
    classes: List[str],
    cfg: "Config | None",
) -> None:
    """Render a single prediction cell with color-coded correctness title.

    Args:
        ax: Matplotlib axes for this cell.
        image: Raw image array in ``(C, H, W)`` format.
        label: Ground-truth class index.
        pred: Predicted class index.
        classes: Human-readable class label names.
        cfg: Configuration for denormalization. If None, skips denorm.
    """
    img = _denormalize_image(image, cfg) if cfg else image
    display_img = _prepare_for_plt(img)

    ax.imshow(display_img, cmap="gray" if display_img.ndim == 2 else None)

    is_correct = label == pred
    ax.set_title(
        f"T:{classes[label]}\nP:{classes[pred]}",
        color="green" if is_correct else "red",
        fontsize=9,
    )


def _build_suptitle(cfg: "Config") -> str:
    """Build the figure suptitle with architecture, resolution, domain, and TTA info.

    Args:
        cfg: Configuration object providing architecture, dataset, and training fields.

    Returns:
        Formatted suptitle string.
    """
    tta_info = f" | TTA: {'ON' if cfg.training.use_tta else 'OFF'}"

    if cfg.dataset.metadata.is_texture_based:
        domain_info = " | Mode: Texture"
    elif cfg.dataset.metadata.is_anatomical:
        domain_info = " | Mode: Anatomical"
    else:
        domain_info = " | Mode: Standard"

    return (
        f"Sample Predictions — {cfg.architecture.name} | "
        f"Resolution: {cfg.dataset.resolution}"
        f"{domain_info}{tta_info}"
    )


def _get_predictions_batch(model: nn.Module, loader: DataLoader, device: torch.device, n: int):
    """Extract a sample batch and run model inference.

    Args:
        model: Trained model in eval mode.
        loader: DataLoader to draw the batch from.
        device: Target device for forward pass.
        n: Number of samples to extract.

    Returns:
        Tuple of ``(images, labels, preds)`` as numpy arrays.
    """
    batch = next(iter(loader))
    images_tensor = batch[0][:n].to(device)
    labels_tensor = batch[1][:n]

    with torch.no_grad():
        outputs = model(images_tensor)
        preds = outputs.argmax(dim=1).cpu().numpy()

    return images_tensor.cpu().numpy(), labels_tensor.numpy().flatten(), preds


def _setup_prediction_grid(num_samples: int, cols: int, cfg: Config | None):
    """Calculate grid dimensions and initialize matplotlib subplots.

    Args:
        num_samples: Total number of images to display.
        cols: Number of columns in the grid.
        cfg: Configuration for figure size. Falls back to ``(12, 8)`` if None.

    Returns:
        Tuple of ``(fig, axes)`` where axes is a flat 1-D array.
    """
    rows = int(np.ceil(num_samples / cols))
    base_w, base_h = cfg.evaluation.fig_size_predictions if cfg else (12, 8)

    fig, axes = plt.subplots(
        rows, cols, figsize=(base_w, (base_h / 3) * rows), constrained_layout=True
    )
    # Ensure axes is always an array even for 1x1 grids
    return fig, np.atleast_1d(axes).flatten()


def _finalize_figure(plt_obj, save_path: Path | None, cfg: Config | None):
    """Save the current figure to disk or display interactively, then close.

    Args:
        plt_obj: The ``matplotlib.pyplot`` module reference.
        save_path: Output file path. If None, calls ``plt.show()`` instead.
        cfg: Configuration for DPI. Falls back to 200 if None.
    """
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        dpi = cfg.evaluation.fig_dpi if cfg else 200
        plt_obj.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        logger.info(f"Predictions grid saved → {save_path.name}")
    else:
        plt_obj.show()
        logger.debug("Displaying figure interactive mode")  # visible at log_level: DEBUG

    plt_obj.close()


def _denormalize_image(img: np.ndarray, cfg: Config) -> np.ndarray:
    """Reverse channel-wise normalization using dataset-specific statistics.

    Args:
        img: Normalized image array in ``(C, H, W)`` format.
        cfg: Configuration providing ``dataset.mean`` and ``dataset.std``.

    Returns:
        Denormalized image clipped to ``[0, 1]``.
    """
    mean = np.array(cfg.dataset.mean).reshape(-1, 1, 1)
    std = np.array(cfg.dataset.std).reshape(-1, 1, 1)
    img = (img * std) + mean
    return np.clip(img, 0, 1)


def _prepare_for_plt(img: np.ndarray) -> np.ndarray:
    """Convert a deep-learning tensor layout to matplotlib-compatible format.

    Transposes ``(C, H, W)`` to ``(H, W, C)`` and squeezes single-channel
    images to 2-D for correct grayscale rendering.

    Args:
        img: Image array, either ``(C, H, W)`` or already ``(H, W)``.

    Returns:
        Image array in ``(H, W, C)`` or ``(H, W)`` format.
    """
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))

    if img.ndim == 3 and img.shape[-1] == 1:
        img = img.squeeze(-1)

    return img
