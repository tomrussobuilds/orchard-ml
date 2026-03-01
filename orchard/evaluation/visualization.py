"""
Visualization utilities for model evaluation.

Provides formatted visual reports including training loss/accuracy curves,
normalized confusion matrices, and sample prediction grids. Integrated with
the PlotContext DTO for aesthetic and technical consistency.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader

from ..core import LOGGER_NAME, LogStyle
from .plot_context import PlotContext

# Global logger instance
logger = logging.getLogger(LOGGER_NAME)


# PUBLIC INTERFACE
def show_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    classes: list[str],
    save_path: Path | None = None,
    ctx: PlotContext | None = None,
    n: int | None = None,
) -> None:
    """
    Visualize model predictions on a sample batch.

    Coordinates data extraction, model inference, grid layout generation,
    and image post-processing. Highlights correct (green) vs. incorrect
    (red) predictions.

    Args:
        model: Trained model to evaluate.
        loader: DataLoader providing evaluation samples.
        device: Target device for inference.
        classes: Human-readable class label names.
        save_path: Output file path. If None, displays interactively.
        ctx: PlotContext with layout and normalization settings.
        n: Number of samples to display. Defaults to ``ctx.n_samples``.
    """
    model.eval()
    style = ctx.plot_style if ctx else "seaborn-v0_8-muted"  # pragma: no mutant

    with plt.style.context(style):  # pragma: no mutant
        # 1. Parameter Resolution & Batch Inference
        num_samples = n or (ctx.n_samples if ctx else 12)
        images, labels, preds = _get_predictions_batch(model, loader, device, num_samples)

        # 2. Grid & Figure Setup
        grid_cols = ctx.grid_cols if ctx else 4
        _, axes = _setup_prediction_grid(len(images), grid_cols, ctx)

        # 3. Plotting Loop
        for i, ax in enumerate(axes):
            if i < len(images):
                _plot_single_prediction(ax, images[i], labels[i], preds[i], classes, ctx)
            ax.axis("off")  # pragma: no mutant

        # 4. Suptitle
        if ctx:
            plt.suptitle(_build_suptitle(ctx), fontsize=14)  # pragma: no mutant

        # 5. Export and Cleanup
        _finalize_figure(plt, save_path, ctx)


def plot_training_curves(
    train_losses: Sequence[float], val_accuracies: Sequence[float], out_path: Path, ctx: PlotContext
) -> None:
    """
    Plot training loss and validation accuracy on a dual-axis chart.

    Saves the figure to disk and exports raw numerical data as ``.npz``
    for reproducibility.

    Args:
        train_losses: Per-epoch training loss values.
        val_accuracies: Per-epoch validation accuracy values.
        out_path: Destination file path for the saved figure.
        ctx: PlotContext with architecture and evaluation settings.
    """
    with plt.style.context(ctx.plot_style):  # pragma: no mutant
        fig, ax1 = plt.subplots(figsize=(9, 6))  # pragma: no mutant

        # Left Axis: Training Loss
        ax1.plot(train_losses, color="#e74c3c", lw=2, label="Training Loss")  # pragma: no mutant
        ax1.set_xlabel("Epoch")  # pragma: no mutant
        ax1.set_ylabel("Loss", color="#e74c3c", fontweight="bold")  # pragma: no mutant
        ax1.tick_params(axis="y", labelcolor="#e74c3c")  # pragma: no mutant
        ax1.grid(True, linestyle="--", alpha=0.4)  # pragma: no mutant

        # Right Axis: Validation Accuracy
        ax2 = ax1.twinx()  # pragma: no mutant
        ax2.plot(
            val_accuracies, color="#3498db", lw=2, label="Validation Accuracy"
        )  # pragma: no mutant
        ax2.set_ylabel("Accuracy", color="#3498db", fontweight="bold")  # pragma: no mutant
        ax2.tick_params(axis="y", labelcolor="#3498db")  # pragma: no mutant

        fig.suptitle(  # pragma: no mutant
            f"Training Metrics — {ctx.arch_name} | Resolution — {ctx.resolution}",
            fontsize=14,  # pragma: no mutant
            y=1.02,  # pragma: no mutant
        )

        fig.tight_layout()  # pragma: no mutant

        plt.savefig(out_path, dpi=ctx.fig_dpi, bbox_inches="tight")  # pragma: no mutant
        logger.info(  # pragma: no mutant
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Training Curves':<18}: {out_path.name}"
        )

        # Export raw data for post-run analysis
        npz_path = out_path.with_suffix(".npz")
        np.savez(npz_path, train_losses=train_losses, val_accuracies=val_accuracies)
        plt.close()


def plot_confusion_matrix(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    classes: list[str],
    out_path: Path,
    ctx: PlotContext,
) -> None:
    """
    Generate and save a row-normalized confusion matrix plot.

    Args:
        all_labels: Ground-truth label array.
        all_preds: Predicted label array.
        classes: Human-readable class label names.
        out_path: Destination file path for the saved figure.
        ctx: PlotContext with architecture and evaluation settings.
    """
    with plt.style.context(ctx.plot_style):  # pragma: no mutant
        cm = confusion_matrix(
            all_labels, all_preds, labels=np.arange(len(classes)), normalize="true"
        )
        cm = np.nan_to_num(cm)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        fig, ax = plt.subplots(figsize=(11, 9))  # pragma: no mutant

        disp.plot(
            ax=ax, cmap=ctx.cmap_confusion, xticks_rotation=45, values_format=".3f"
        )  # pragma: no mutant
        plt.title(  # pragma: no mutant
            f"Confusion Matrix — {ctx.arch_name} | Resolution — {ctx.resolution}",
            fontsize=12,  # pragma: no mutant
            pad=20,  # pragma: no mutant
        )

        plt.tight_layout()  # pragma: no mutant

        fig.savefig(out_path, dpi=ctx.fig_dpi, bbox_inches="tight")  # pragma: no mutant
        plt.close()
        logger.info(  # pragma: no mutant
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Confusion Matrix':<18}: {out_path.name}"
        )


def _plot_single_prediction(
    ax,
    image: np.ndarray,
    label: int,
    pred: int,
    classes: list[str],
    ctx: PlotContext | None,
) -> None:
    """
    Render a single prediction cell with color-coded correctness title.

    Args:
        ax: Matplotlib axes for this cell.
        image: Raw image array in ``(C, H, W)`` format.
        label: Ground-truth class index.
        pred: Predicted class index.
        classes: Human-readable class label names.
        ctx: PlotContext for denormalization. If None, skips denorm.
    """
    img = _denormalize_image(image, ctx) if ctx else image
    display_img = _prepare_for_plt(img)

    ax.imshow(display_img, cmap="gray" if display_img.ndim == 2 else None)  # pragma: no mutant

    is_correct = label == pred
    ax.set_title(  # pragma: no mutant
        f"T:{classes[label]}\nP:{classes[pred]}",
        color="green" if is_correct else "red",
        fontsize=9,  # pragma: no mutant
    )


def _build_suptitle(ctx: PlotContext) -> str:
    """
    Build the figure suptitle with architecture, resolution, domain, and TTA info.

    Args:
        ctx: PlotContext providing architecture, dataset, and training fields.

    Returns:
        Formatted suptitle string.
    """
    tta_info = f" | TTA: {'ON' if ctx.use_tta else 'OFF'}"  # pragma: no mutant

    if ctx.is_texture_based:
        domain_info = " | Mode: Texture"  # pragma: no mutant
    elif ctx.is_anatomical:
        domain_info = " | Mode: Anatomical"  # pragma: no mutant
    else:
        domain_info = " | Mode: Standard"  # pragma: no mutant

    return (
        f"Sample Predictions — {ctx.arch_name} | "  # pragma: no mutant
        f"Resolution: {ctx.resolution}"
        f"{domain_info}{tta_info}"
    )


def _get_predictions_batch(
    model: nn.Module, loader: DataLoader, device: torch.device, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract a sample batch and run model inference.

    Args:
        model: Trained model in eval mode.
        loader: DataLoader to draw the batch from.
        device: Target device for forward pass.
        n: Number of samples to extract.

    Returns:
        tuple of ``(images, labels, preds)`` as numpy arrays.
    """
    batch = next(iter(loader))
    images_tensor = batch[0][:n].to(device)
    labels_tensor = batch[1][:n]

    with torch.no_grad():
        outputs = model(images_tensor)
        preds = outputs.argmax(dim=1).cpu().numpy()

    return images_tensor.cpu().numpy(), labels_tensor.numpy().flatten(), preds


def _setup_prediction_grid(
    num_samples: int, cols: int, ctx: PlotContext | None
) -> tuple[plt.Figure, np.ndarray]:
    """
    Calculate grid dimensions and initialize matplotlib subplots.

    Args:
        num_samples: Total number of images to display.
        cols: Number of columns in the grid.
        ctx: PlotContext for figure size. Falls back to ``(12, 8)`` if None.

    Returns:
        tuple of ``(fig, axes)`` where axes is a flat 1-D array.
    """
    rows = int(np.ceil(num_samples / cols))
    base_w, base_h = ctx.fig_size_predictions if ctx else (12, 8)  # pragma: no mutant

    fig, axes = plt.subplots(  # pragma: no mutant
        rows, cols, figsize=(base_w, (base_h / 3) * rows), constrained_layout=True
    )
    # Ensure axes is always an array even for 1x1 grids
    return fig, np.atleast_1d(axes).flatten()


def _finalize_figure(plt_obj, save_path: Path | None, ctx: PlotContext | None) -> None:
    """
    Save the current figure to disk or display interactively, then close.

    Args:
        plt_obj: The ``matplotlib.pyplot`` module reference.
        save_path: Output file path. If None, calls ``plt.show()`` instead.
        ctx: PlotContext for DPI. Falls back to 200 if None.
    """
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        dpi = ctx.fig_dpi if ctx else 200
        plt_obj.savefig(
            save_path, dpi=dpi, bbox_inches="tight", facecolor="white"
        )  # pragma: no mutant
        logger.info(  # pragma: no mutant
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Predictions Grid':<18}: {save_path.name}"
        )
    else:
        plt_obj.show()
        logger.debug("Displaying figure interactive mode")  # pragma: no mutant

    plt_obj.close()


def _denormalize_image(img: np.ndarray, ctx: PlotContext) -> np.ndarray:
    """
    Reverse channel-wise normalization using dataset-specific statistics.

    Args:
        img: Normalized image array in ``(C, H, W)`` format.
        ctx: PlotContext providing ``mean`` and ``std``.

    Returns:
        Denormalized image clipped to ``[0, 1]``.
    """
    mean = np.array(ctx.mean).reshape(-1, 1, 1)
    std = np.array(ctx.std).reshape(-1, 1, 1)
    img = (img * std) + mean
    return np.clip(img, 0, 1)  # type: ignore[no-any-return]


def _prepare_for_plt(img: np.ndarray) -> np.ndarray:
    """
    Convert a deep-learning tensor layout to matplotlib-compatible format.

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
