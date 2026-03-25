"""
Detection visualization utilities.

Renders bounding-box overlays on sample images from the test set,
showing ground-truth boxes (green) and predicted boxes (red/blue)
with confidence scores. Follows the same PlotContext / _finalize_figure
pattern used by the classification visualization module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader

from ..core import LOGGER_NAME
from .plot_context import PlotContext
from .visualization import _denormalize_image, _finalize_figure, _prepare_for_plt

logger = logging.getLogger(LOGGER_NAME)

# Cosmetic constants — colours, linewidths, font sizes
_GT_COLOR = "#2ecc71"  # pragma: no mutate
_PRED_COLOR = "#e74c3c"  # pragma: no mutate
_GT_LW = 2.0  # pragma: no mutate
_PRED_LW = 2.0  # pragma: no mutate
_SCORE_FONTSIZE = 7  # pragma: no mutate
_DEFAULT_CONFIDENCE = 0.5  # pragma: no mutate


def show_detections(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    classes: list[str],
    save_path: Path | None = None,
    ctx: PlotContext | None = None,
    n: int | None = None,
    confidence_threshold: float = _DEFAULT_CONFIDENCE,
) -> None:
    """
    Visualize detection predictions with bounding-box overlays.

    Renders a grid of sample images from the test loader with
    ground-truth boxes (green) and predicted boxes (red) above
    a configurable confidence threshold.

    Args:
        model: Trained detection model in eval mode.
        loader: DataLoader yielding ``(list[Tensor], list[dict])`` batches.
        device: Target device for inference.
        classes: Human-readable class label names.
        save_path: Output file path. If None, displays interactively.
        ctx: PlotContext with layout and normalization settings.
        n: Number of samples to display. Defaults to ``ctx.n_samples``.
        confidence_threshold: Minimum score to display a predicted box.
    """
    model.eval()
    style = ctx.plot_style if ctx else "seaborn-v0_8-muted"  # pragma: no mutate

    with plt.style.context(style):
        num_samples = n or (ctx.n_samples if ctx else 12)  # pragma: no mutate
        images, targets, predictions = _get_detection_batch(
            model, loader, device, num_samples  # pragma: no mutate
        )

        grid_cols = ctx.grid_cols if ctx else 4  # pragma: no mutate
        rows = int(np.ceil(len(images) / grid_cols))  # pragma: no mutate
        base_w, base_h = ctx.fig_size_predictions if ctx else (12, 8)  # pragma: no mutate

        fig, axes = plt.subplots(
            rows,
            grid_cols,
            figsize=(base_w, (base_h / 3) * rows),
            constrained_layout=True,
        )
        axes_flat: npt.NDArray[Any] = np.atleast_1d(axes).flatten()

        for i, ax in enumerate(axes_flat):
            if i < len(images):
                _plot_single_detection(
                    ax,
                    images[i],
                    targets[i],
                    predictions[i],
                    classes,
                    ctx,
                    confidence_threshold,
                )
            ax.axis("off")

        if ctx:
            plt.suptitle(
                f"Detection Samples — {ctx.arch_name} | Resolution: {ctx.resolution}",
                fontsize=14,
            )

        _finalize_figure(plt, save_path, ctx)  # pragma: no mutate


def _get_detection_batch(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    n: int,
) -> tuple[
    list[npt.NDArray[Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    Extract sample images, ground-truth targets, and model predictions.

    Args:
        model: Detection model in eval mode.
        loader: Detection DataLoader.
        device: Target device.
        n: Maximum number of samples to collect.

    Returns:
        Tuple of (images_np, targets_cpu, predictions_cpu).
    """
    collected_images: list[npt.NDArray[Any]] = []
    collected_targets: list[dict[str, Any]] = []
    collected_preds: list[dict[str, Any]] = []

    with torch.no_grad():
        for images, targets in loader:
            images_on_device = [img.to(device) for img in images]  # pragma: no mutate
            preds = model(images_on_device)

            for img, tgt, pred in zip(images, targets, preds):
                collected_images.append(img.cpu().numpy())
                collected_targets.append(
                    {
                        k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                        for k, v in tgt.items()
                    }
                )
                collected_preds.append(
                    {
                        k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                        for k, v in pred.items()
                    }
                )
                if len(collected_images) >= n:
                    return collected_images, collected_targets, collected_preds

    return collected_images, collected_targets, collected_preds


def _plot_single_detection(
    ax: Any,
    image: npt.NDArray[Any],
    target: dict[str, Any],
    prediction: dict[str, Any],
    classes: list[str],
    ctx: PlotContext | None,
    confidence_threshold: float,
) -> None:
    """
    Render one image with ground-truth and predicted bounding boxes.

    Args:
        ax: Matplotlib axes for this cell.
        image: Raw image array in ``(C, H, W)`` format.
        target: Ground-truth dict with ``boxes`` and ``labels`` arrays.
        prediction: Prediction dict with ``boxes``, ``scores``, ``labels``.
        classes: Human-readable class label names.
        ctx: PlotContext for denormalization.
        confidence_threshold: Minimum score to render a predicted box.
    """
    img = _denormalize_image(image, ctx) if ctx and ctx.mean else image
    display_img = _prepare_for_plt(img)

    ax.imshow(display_img, cmap="gray" if display_img.ndim == 2 else None)

    # Ground-truth boxes (green, dashed)
    gt_boxes = target.get("boxes", np.empty((0, 4)))  # pragma: no mutate
    gt_labels = target.get("labels", np.empty(0, dtype=int))  # pragma: no mutate
    for box, label in zip(gt_boxes, gt_labels):
        _draw_box(ax, box, _GT_COLOR, _GT_LW, linestyle="--")
        _draw_label(ax, box, _format_gt_label(classes, int(label)), _GT_COLOR)

    # Predicted boxes (red, solid) — filtered by confidence
    pred_boxes = prediction.get("boxes", np.empty((0, 4)))  # pragma: no mutate
    pred_scores = prediction.get("scores", np.empty(0))  # pragma: no mutate
    pred_labels = prediction.get("labels", np.empty(0, dtype=int))  # pragma: no mutate

    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        if float(score) < confidence_threshold:
            continue
        _draw_box(ax, box, _PRED_COLOR, _PRED_LW, linestyle="-")
        _draw_label(
            ax,
            box,
            _format_pred_label(classes, int(label), float(score)),
            _PRED_COLOR,
            bottom=True,
        )


def _draw_box(
    ax: Any,
    box: npt.NDArray[Any],
    color: str,
    lw: float,
    *,
    linestyle: str = "-",  # pragma: no mutate
) -> None:
    """
    Draw a single bounding box rectangle on the axes.

    Args:
        ax: Matplotlib axes.
        box: Array of ``[x1, y1, x2, y2]``.
        color: Edge colour.
        lw: Line width.
        linestyle: Matplotlib line style string.
    """
    x1, y1, x2, y2 = box
    rect = Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        linewidth=lw,
        edgecolor=color,
        facecolor="none",
        linestyle=linestyle,
    )
    ax.add_patch(rect)


def _draw_label(
    ax: Any,
    box: npt.NDArray[Any],
    text: str,
    color: str,
    *,
    bottom: bool = False,  # pragma: no mutate
) -> None:
    """
    Place a text label near a bounding box.

    Args:
        ax: Matplotlib axes.
        box: Array of ``[x1, y1, x2, y2]``.
        text: Label string.
        color: Text and background colour.
        bottom: If True, place below box; otherwise above.
    """
    x1, y1, _, y2 = box
    y_pos = y2 + 1 if bottom else y1 - 1  # pragma: no mutate
    va = "top" if bottom else "bottom"  # pragma: no mutate
    ax.text(
        x1,
        y_pos,
        text,
        color="white",
        fontsize=_SCORE_FONTSIZE,
        va=va,
        ha="left",
        bbox={"facecolor": color, "alpha": 0.6, "pad": 1, "edgecolor": "none"},
    )


def _format_gt_label(classes: list[str], label: int) -> str:
    """
    Format ground-truth label text.

    Args:
        classes: Class name list.
        label: Class index.

    Returns:
        Formatted label like ``"GT: person"``.
    """
    name = classes[label] if label < len(classes) else str(label)  # pragma: no mutate
    return f"GT: {name}"  # pragma: no mutate


def _format_pred_label(classes: list[str], label: int, score: float) -> str:
    """
    Format predicted-box label text with confidence score.

    Args:
        classes: Class name list.
        label: Predicted class index.
        score: Confidence score.

    Returns:
        Formatted label like ``"person 0.92"``.
    """
    name = classes[label] if label < len(classes) else str(label)  # pragma: no mutate
    return f"{name} {score:.2f}"  # pragma: no mutate
