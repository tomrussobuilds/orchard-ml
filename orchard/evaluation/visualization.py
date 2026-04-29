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


# Cosmetic constants — colours, line widths, font sizes, layout, defaults
# pragma: no mutate start
_LOSS_COLOR = "#e74c3c"
_METRIC_COLOR = "#3498db"
_LINE_WIDTH = 2
_GRID_LINESTYLE = "--"
_GRID_ALPHA = 0.4
_TRAINING_FIGSIZE = (9, 6)
_SUPTITLE_FONTSIZE = 14
_TRAINING_TITLE_Y = 1.02
_CM_FIGSIZE = (11, 9)
_CM_TICKS_ROTATION = 45
_CM_VALUES_FORMAT = ".3f"
_CM_TITLE_FONTSIZE = 12
_CM_TITLE_PAD = 20
_GRAY_CMAP = "gray"
_CORRECT_COLOR = "green"
_INCORRECT_COLOR = "red"
_CELL_TITLE_FONTSIZE = 9
_SAVEFIG_BBOX = "tight"
_SAVEFIG_FACECOLOR = "white"
_DEFAULT_STYLE = "seaborn-v0_8-muted"
_DEFAULT_NUM_SAMPLES = 12
_DEFAULT_GRID_COLS = 4
_DEFAULT_FIGSIZE = (12, 8)
_FIGSIZE_HEIGHT_FACTOR = 3
_DEFAULT_DPI = 200
# pragma: no mutate end


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
    style = ctx.plot_style if ctx else _DEFAULT_STYLE

    with plt.style.context(style):
        # 1. Parameter Resolution & Batch Inference
        num_samples = n or (ctx.n_samples if ctx else _DEFAULT_NUM_SAMPLES)
        images, labels, preds = _get_predictions_batch(model, loader, device, num_samples)

        # 2. Grid & Figure Setup
        grid_cols = ctx.grid_cols if ctx else _DEFAULT_GRID_COLS
        _, axes = _setup_prediction_grid(len(images), grid_cols, ctx)

        # 3. Plotting Loop
        for i, ax in enumerate(axes):
            # guard for extra grid cells beyond actual images
            if i < len(images):
                _plot_single_prediction(ax, images[i], labels[i], preds[i], classes, ctx)
            ax.axis("off")

        # 4. Suptitle
        if ctx:
            plt.suptitle(_build_suptitle(ctx), fontsize=_SUPTITLE_FONTSIZE)

        # 5. Export and Cleanup
        _finalize_figure(plt, save_path, ctx)


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
    with plt.style.context(ctx.plot_style):
        fig, ax1 = plt.subplots(figsize=_TRAINING_FIGSIZE)

        # Left Axis: Training Loss
        ax1.plot(train_losses, color=_LOSS_COLOR, lw=_LINE_WIDTH, label="Training Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color=_LOSS_COLOR, fontweight="bold")
        ax1.tick_params(axis="y", labelcolor=_LOSS_COLOR)
        ax1.grid(True, linestyle=_GRID_LINESTYLE, alpha=_GRID_ALPHA)

        # Right Axis: Validation Metric
        ax2 = ax1.twinx()
        ax2.plot(val_metric_values, color=_METRIC_COLOR, lw=_LINE_WIDTH, label=val_label)
        ax2.set_ylabel(val_label, color=_METRIC_COLOR, fontweight="bold")
        ax2.tick_params(axis="y", labelcolor=_METRIC_COLOR)

        fig.suptitle(
            f"Training Metrics — {ctx.arch_name} | Resolution — {ctx.resolution}",
            fontsize=_SUPTITLE_FONTSIZE,
            y=_TRAINING_TITLE_Y,
        )

        fig.tight_layout()

        plt.savefig(out_path, dpi=ctx.fig_dpi, bbox_inches=_SAVEFIG_BBOX)
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
    with plt.style.context(ctx.plot_style):
        cm = confusion_matrix(
            all_labels,
            all_preds,
            labels=np.arange(len(classes)),
            normalize="true",
        )
        cm = np.nan_to_num(cm)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        fig, ax = plt.subplots(figsize=_CM_FIGSIZE)

        disp.plot(
            ax=ax,
            cmap=ctx.cmap_confusion,
            xticks_rotation=_CM_TICKS_ROTATION,
            values_format=_CM_VALUES_FORMAT,
        )
        plt.title(
            f"Confusion Matrix — {ctx.arch_name} | Resolution — {ctx.resolution}",
            fontsize=_CM_TITLE_FONTSIZE,
            pad=_CM_TITLE_PAD,
        )

        plt.tight_layout()

        fig.savefig(out_path, dpi=ctx.fig_dpi, bbox_inches=_SAVEFIG_BBOX)
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

    ax.imshow(display_img, cmap=_GRAY_CMAP if display_img.ndim == 2 else None)

    is_correct = label == pred
    ax.set_title(
        f"T:{classes[label]}\nP:{classes[pred]}",
        color=_CORRECT_COLOR if is_correct else _INCORRECT_COLOR,
        fontsize=_CELL_TITLE_FONTSIZE,
    )


def _build_suptitle(ctx: PlotContext) -> str:
    """
    Build the figure suptitle with architecture, resolution, domain, and TTA info.

    Args:
        ctx: PlotContext providing architecture, dataset, and training fields.

    Returns:
        Formatted suptitle string.
    """
    tta_info = f" | TTA: {'ON' if ctx.use_tta else 'OFF'}"

    if ctx.is_texture_based:
        domain_info = " | Mode: Texture"
    elif ctx.is_anatomical:
        domain_info = " | Mode: Anatomical"
    else:
        domain_info = " | Mode: Standard"

    return (
        f"Sample Predictions — {ctx.arch_name} | "
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
    images_tensor = batch[0][:n].to(device)
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
    base_w, base_h = ctx.fig_size_predictions if ctx else _DEFAULT_FIGSIZE

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(base_w, (base_h / _FIGSIZE_HEIGHT_FACTOR) * rows),
        constrained_layout=True,
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
        dpi = ctx.fig_dpi if ctx else _DEFAULT_DPI
        plt_obj.savefig(save_path, dpi=dpi, bbox_inches=_SAVEFIG_BBOX, facecolor=_SAVEFIG_FACECOLOR)
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
    mean = np.array(ctx.mean).reshape(-1, 1, 1)
    std = np.array(ctx.std).reshape(-1, 1, 1)
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
        img = img.squeeze(-1)

    return img
