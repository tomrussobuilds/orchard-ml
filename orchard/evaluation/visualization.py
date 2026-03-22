"""
Visualization utilities for model evaluation.

Provides formatted visual reports including training loss/accuracy curves,
normalized confusion matrices, and sample prediction grids. Integrated with
the PlotContext DTO for aesthetic and technical consistency.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader

from ..core import LOGGER_NAME, LogStyle
from .plot_context import PlotContext

# Global logger instance
logger = logging.getLogger(LOGGER_NAME)


# PUBLIC INTERFACE
def show_predictions(
    model: nn.Module,
    loader: DataLoader[Any],
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
    # cosmetic fallback
    style = ctx.plot_style if ctx else "seaborn-v0_8-muted"  # pragma: no mutate

    with plt.style.context(style):  # pragma: no mutate
        # 1. Parameter Resolution & Batch Inference
        # cosmetic fallback
        num_samples = n or (ctx.n_samples if ctx else 12)  # pragma: no mutate
        images, labels, preds = _get_predictions_batch(
            model, loader, device, num_samples  # pragma: no mutate
        )

        # 2. Grid & Figure Setup
        # cosmetic fallback
        grid_cols = ctx.grid_cols if ctx else 4  # pragma: no mutate
        _, axes = _setup_prediction_grid(len(images), grid_cols, ctx)  # pragma: no mutate

        # 3. Plotting Loop
        for i, ax in enumerate(axes):
            # guard for extra grid cells beyond actual images
            if i < len(images):  # pragma: no mutate
                _plot_single_prediction(
                    ax, images[i], labels[i], preds[i], classes, ctx  # pragma: no mutate
                )
            ax.axis("off")  # pragma: no mutate

        # 4. Suptitle
        if ctx:
            plt.suptitle(_build_suptitle(ctx), fontsize=14)  # pragma: no mutate

        # 5. Export and Cleanup
        # forwarding; tested in _finalize_figure
        _finalize_figure(plt, save_path, ctx)  # pragma: no mutate


def plot_training_curves(
    train_losses: Sequence[float],
    val_metric_values: Sequence[float],
    out_path: Path,
    ctx: PlotContext,
    *,
    val_label: str,
) -> None:
    """
    Plot training loss and a validation metric on a dual-axis chart.

    Saves the figure to disk and exports raw numerical data as ``.npz``
    for reproducibility.

    Args:
        train_losses: Per-epoch training loss values.
        val_metric_values: Per-epoch validation metric values.
        out_path: Destination file path for the saved figure.
        ctx: PlotContext with architecture and evaluation settings.
        val_label: Label for the right y-axis and legend entry.
    """
    # matplotlib cosmetic — colors, fonts, sizes, layout
    with plt.style.context(ctx.plot_style):  # pragma: no mutate
        fig, ax1 = plt.subplots(figsize=(9, 6))  # pragma: no mutate

        # Left Axis: Training Loss
        ax1.plot(train_losses, color="#e74c3c", lw=2, label="Training Loss")  # pragma: no mutate
        ax1.set_xlabel("Epoch")  # pragma: no mutate
        ax1.set_ylabel("Loss", color="#e74c3c", fontweight="bold")  # pragma: no mutate
        ax1.tick_params(axis="y", labelcolor="#e74c3c")  # pragma: no mutate
        ax1.grid(True, linestyle="--", alpha=0.4)  # pragma: no mutate

        # Right Axis: Validation Metric
        ax2 = ax1.twinx()  # pragma: no mutate
        ax2.plot(  # pragma: no mutate
            val_metric_values, color="#3498db", lw=2, label=val_label  # pragma: no mutate
        )  # pragma: no mutate
        ax2.set_ylabel(val_label, color="#3498db", fontweight="bold")  # pragma: no mutate
        ax2.tick_params(axis="y", labelcolor="#3498db")  # pragma: no mutate

        fig.suptitle(  # pragma: no mutate
            f"Training Metrics — {ctx.arch_name} | Resolution — {ctx.resolution}",
            fontsize=14,  # pragma: no mutate
            y=1.02,  # pragma: no mutate
        )

        fig.tight_layout()  # pragma: no mutate

        plt.savefig(out_path, dpi=ctx.fig_dpi, bbox_inches="tight")  # pragma: no mutate
        logger.info(
            "%s%s %-18s: %s", LogStyle.INDENT, LogStyle.ARROW, "Training Curves", out_path.name
        )

        # Export raw data for post-run analysis
        npz_path = out_path.with_suffix(".npz")
        np.savez(npz_path, train_losses=train_losses, val_metric_values=val_metric_values)
        plt.close()


def plot_confusion_matrix(
    all_labels: npt.NDArray[Any],
    all_preds: npt.NDArray[Any],
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
    # matplotlib cosmetic — confusion matrix rendering and styling
    with plt.style.context(ctx.plot_style):  # pragma: no mutate
        cm = confusion_matrix(  # pragma: no mutate
            all_labels,
            all_preds,
            labels=np.arange(len(classes)),
            normalize="true",  # pragma: no mutate
        )
        cm = np.nan_to_num(cm)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=classes  # pragma: no mutate
        )
        fig, ax = plt.subplots(figsize=(11, 9))  # pragma: no mutate

        disp.plot(  # pragma: no mutate
            ax=ax,
            cmap=ctx.cmap_confusion,
            xticks_rotation=45,  # pragma: no mutate
            values_format=".3f",  # pragma: no mutate
        )  # pragma: no mutate
        plt.title(  # pragma: no mutate
            f"Confusion Matrix — {ctx.arch_name} | Resolution — {ctx.resolution}",
            fontsize=12,  # pragma: no mutate
            pad=20,  # pragma: no mutate
        )

        plt.tight_layout()  # pragma: no mutate

        fig.savefig(out_path, dpi=ctx.fig_dpi, bbox_inches="tight")  # pragma: no mutate
        plt.close()
        logger.info(
            "%s%s %-18s: %s", LogStyle.INDENT, LogStyle.ARROW, "Confusion Matrix", out_path.name
        )


def _plot_single_prediction(
    ax: Any,
    image: npt.NDArray[Any],
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

    # cosmetic — imshow cmap and title styling
    ax.imshow(display_img, cmap="gray" if display_img.ndim == 2 else None)  # pragma: no mutate

    is_correct = label == pred
    ax.set_title(  # pragma: no mutate
        f"T:{classes[label]}\nP:{classes[pred]}",  # pragma: no mutate
        color="green" if is_correct else "red",  # pragma: no mutate
        fontsize=9,  # pragma: no mutate
    )


def _build_suptitle(ctx: PlotContext) -> str:
    """
    Build the figure suptitle with architecture, resolution, domain, and TTA info.

    Args:
        ctx: PlotContext providing architecture, dataset, and training fields.

    Returns:
        Formatted suptitle string.
    """
    tta_info = f" | TTA: {'ON' if ctx.use_tta else 'OFF'}"  # pragma: no mutate

    if ctx.is_texture_based:
        domain_info = " | Mode: Texture"  # pragma: no mutate
    elif ctx.is_anatomical:
        domain_info = " | Mode: Anatomical"  # pragma: no mutate
    else:
        domain_info = " | Mode: Standard"  # pragma: no mutate

    return (
        f"Sample Predictions — {ctx.arch_name} | "  # pragma: no mutate
        f"Resolution: {ctx.resolution}"
        f"{domain_info}{tta_info}"
    )


def _get_predictions_batch(
    model: nn.Module, loader: DataLoader[Any], device: torch.device, n: int
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
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
    images_tensor = batch[0][:n].to(device)  # pragma: no mutate
    labels_tensor = batch[1][:n]

    with torch.no_grad():
        outputs = model(images_tensor)
        preds = outputs.argmax(dim=1).cpu().numpy()

    return images_tensor.cpu().numpy(), labels_tensor.numpy().flatten(), preds


def _setup_prediction_grid(
    num_samples: int, cols: int, ctx: PlotContext | None
) -> tuple[Figure, npt.NDArray[Any]]:
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
    # cosmetic fallback and layout
    base_w, base_h = ctx.fig_size_predictions if ctx else (12, 8)  # pragma: no mutate

    fig, axes = plt.subplots(  # pragma: no mutate
        rows,
        cols,
        figsize=(base_w, (base_h / 3) * rows),  # pragma: no mutate
        constrained_layout=True,  # pragma: no mutate
    )
    # Ensure axes is always an array even for 1x1 grids
    return fig, np.atleast_1d(axes).flatten()


def _finalize_figure(plt_obj: Any, save_path: Path | None, ctx: PlotContext | None) -> None:
    """
    Save the current figure to disk or display interactively, then close.

    Args:
        plt_obj: The ``matplotlib.pyplot`` module reference.
        save_path: Output file path. If None, calls ``plt.show()`` instead.
        ctx: PlotContext for DPI. Falls back to 200 if None.
    """
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # cosmetic — dpi fallback and savefig styling
        dpi = ctx.fig_dpi if ctx else 200  # pragma: no mutate
        plt_obj.savefig(  # pragma: no mutate
            save_path, dpi=dpi, bbox_inches="tight", facecolor="white"  # pragma: no mutate
        )
        logger.info(
            "%s%s %-18s: %s", LogStyle.INDENT, LogStyle.ARROW, "Predictions Grid", save_path.name
        )
    else:
        plt_obj.show()
        logger.debug("Displaying figure interactive mode")

    plt_obj.close()


def _denormalize_image(img: npt.NDArray[Any], ctx: PlotContext) -> npt.NDArray[Any]:
    """
    Reverse channel-wise normalization using dataset-specific statistics.

    Args:
        img: Normalized image array in ``(C, H, W)`` format.
        ctx: PlotContext providing ``mean`` and ``std``.

    Returns:
        Denormalized image clipped to ``[0, 1]``.
    """
    mean = np.array(ctx.mean).reshape(-1, 1, 1)  # pragma: no mutate
    std = np.array(ctx.std).reshape(-1, 1, 1)  # pragma: no mutate
    img = (img * std) + mean
    result: npt.NDArray[Any] = np.clip(img, 0, 1)
    return result


def _prepare_for_plt(img: npt.NDArray[Any]) -> npt.NDArray[Any]:
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
        img = img.squeeze(-1)  # pragma: no mutate

    return img
