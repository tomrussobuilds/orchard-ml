"""
Tests for Detection Visualization Module.

Validates bounding-box overlay rendering, grid layout, confidence
filtering, and integration with the PlotContext / _finalize_figure
pattern.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from orchard.evaluation.detection_visualization import (
    _draw_box,
    _draw_label,
    _format_gt_label,
    _format_pred_label,
    _get_detection_batch,
    _plot_single_detection,
    show_detections,
)
from orchard.evaluation.plot_context import PlotContext

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def det_ctx() -> PlotContext:
    """PlotContext for detection visualization tests."""
    return PlotContext(
        arch_name="fasterrcnn",
        resolution=224,
        fig_dpi=100,
        plot_style="default",
        cmap_confusion="Blues",
        grid_cols=2,
        n_samples=4,
        fig_size_predictions=(10, 8),
    )


def _make_det_model(
    num_preds: int = 2,
) -> MagicMock:
    """Mock detection model returning predictions per image."""

    def forward(images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        return [
            {
                "boxes": torch.tensor(
                    [[10.0, 10.0, 50.0, 50.0], [20.0, 20.0, 60.0, 60.0]][:num_preds]
                ),
                "scores": torch.tensor([0.9, 0.3][:num_preds]),
                "labels": torch.tensor([0, 1][:num_preds]),
            }
            for _ in images
        ]

    model = MagicMock(spec=nn.Module)
    model.side_effect = forward
    return model


def _make_det_loader(
    batch_size: int = 2,
    img_size: int = 64,
) -> MagicMock:
    """Mock detection DataLoader yielding one batch."""
    images = [torch.randn(3, img_size, img_size) for _ in range(batch_size)]
    targets = [
        {
            "boxes": torch.tensor([[5.0, 5.0, 40.0, 40.0]]),
            "labels": torch.tensor([0]),
        }
        for _ in range(batch_size)
    ]
    loader = MagicMock()
    loader.__iter__ = MagicMock(return_value=iter([(images, targets)]))
    return loader


# ── _format_gt_label / _format_pred_label ────────────────────────────────────


@pytest.mark.unit
def test_format_gt_label_valid_index() -> None:
    """GT label uses class name when index is valid."""
    assert _format_gt_label(["cat", "dog"], 0) == "GT: cat"
    assert _format_gt_label(["cat", "dog"], 1) == "GT: dog"


@pytest.mark.unit
def test_format_gt_label_out_of_range() -> None:
    """GT label falls back to str(index) for out-of-range labels."""
    assert _format_gt_label(["cat"], 5) == "GT: 5"


@pytest.mark.unit
def test_format_pred_label_valid() -> None:
    """Pred label includes class name and score."""
    result = _format_pred_label(["person", "car"], 1, 0.87)
    assert result == "car 0.87"


@pytest.mark.unit
def test_format_pred_label_out_of_range() -> None:
    """Pred label falls back to str(index) for out-of-range labels."""
    result = _format_pred_label(["cat"], 99, 0.5)
    assert result == "99 0.50"


@pytest.mark.unit
def test_format_gt_label_at_boundary_uses_fallback() -> None:
    """At the exact boundary ``label == len(classes)``, GT label uses str(label).

    Kills the ``<`` → ``<=`` mutation: with ``<=``, ``classes[label]`` would
    raise IndexError (out-of-range access)."""
    assert _format_gt_label(["cat"], 1) == "GT: 1"


@pytest.mark.unit
def test_format_pred_label_at_boundary_uses_fallback() -> None:
    """At the exact boundary ``label == len(classes)``, pred label uses str(label).

    Kills the ``<`` → ``<=`` mutation: with ``<=``, ``classes[label]`` would
    raise IndexError."""
    assert _format_pred_label(["cat"], 1, 0.7) == "1 0.70"


# ── _draw_box ────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_draw_box_adds_patch() -> None:
    """_draw_box adds a Rectangle patch to the axes."""
    ax = MagicMock()
    box = np.array([10.0, 20.0, 50.0, 60.0])
    _draw_box(ax, box, "#ff0000", 2.0)
    ax.add_patch.assert_called_once()

    rect = ax.add_patch.call_args[0][0]
    assert rect.get_xy() == (10.0, 20.0)
    assert rect.get_width() == pytest.approx(40.0)
    assert rect.get_height() == pytest.approx(40.0)


@pytest.mark.unit
def test_draw_box_dashed() -> None:
    """_draw_box supports dashed linestyle."""
    ax = MagicMock()
    box = np.array([0.0, 0.0, 10.0, 10.0])
    _draw_box(ax, box, "#00ff00", 1.5, linestyle="--")
    rect = ax.add_patch.call_args[0][0]
    assert rect.get_linestyle() == "--"


# ── _draw_label ──────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_draw_label_places_text_above() -> None:
    """_draw_label places text above box by default."""
    ax = MagicMock()
    box = np.array([10.0, 20.0, 50.0, 60.0])
    _draw_label(ax, box, "GT: cat", "#00ff00")
    ax.text.assert_called_once()
    _, y_pos = ax.text.call_args[0][:2]
    assert y_pos < 20.0  # above y1


@pytest.mark.unit
def test_draw_label_places_text_below() -> None:
    """_draw_label places text below box when bottom=True."""
    ax = MagicMock()
    box = np.array([10.0, 20.0, 50.0, 60.0])
    _draw_label(ax, box, "person 0.9", "#ff0000", bottom=True)
    ax.text.assert_called_once()
    _, y_pos = ax.text.call_args[0][:2]
    assert y_pos > 60.0  # below y2


# ── _get_detection_batch ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_get_detection_batch_returns_correct_count() -> None:
    """_get_detection_batch collects exactly n samples."""
    model = _make_det_model()
    loader = _make_det_loader(batch_size=4)
    images, targets, preds = _get_detection_batch(model, loader, torch.device("cpu"), n=2)

    assert len(images) == 2
    assert len(targets) == 2
    assert len(preds) == 2


@pytest.mark.unit
def test_get_detection_batch_returns_numpy() -> None:
    """_get_detection_batch returns numpy arrays for images and target values."""
    model = _make_det_model()
    loader = _make_det_loader(batch_size=2)
    images, targets, preds = _get_detection_batch(model, loader, torch.device("cpu"), n=2)

    assert isinstance(images[0], np.ndarray)
    assert isinstance(targets[0]["boxes"], np.ndarray)
    assert isinstance(preds[0]["scores"], np.ndarray)


@pytest.mark.unit
def test_get_detection_batch_fewer_than_n() -> None:
    """_get_detection_batch returns all available when loader has fewer than n."""
    model = _make_det_model()
    loader = _make_det_loader(batch_size=2)
    images, targets, preds = _get_detection_batch(model, loader, torch.device("cpu"), n=10)

    # Only 2 available, requested 10 — returns 2
    assert len(images) == 2
    assert len(targets) == 2
    assert len(preds) == 2


@pytest.mark.unit
def test_get_detection_batch_multiple_batches() -> None:
    """_get_detection_batch spans multiple batches if needed."""
    images_b1 = [torch.randn(3, 32, 32)]
    targets_b1 = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([0])}]
    images_b2 = [torch.randn(3, 32, 32)]
    targets_b2 = [{"boxes": torch.tensor([[5.0, 5.0, 15.0, 15.0]]), "labels": torch.tensor([1])}]

    loader = MagicMock()
    loader.__iter__ = MagicMock(
        return_value=iter([(images_b1, targets_b1), (images_b2, targets_b2)])
    )

    model = _make_det_model(num_preds=1)
    imgs, tgts, prs = _get_detection_batch(model, loader, torch.device("cpu"), n=2)
    assert len(imgs) == 2


# ── _plot_single_detection ───────────────────────────────────────────────────


@pytest.mark.unit
def test_plot_single_detection_draws_gt_and_pred() -> None:
    """_plot_single_detection renders GT and predicted boxes."""
    ax = MagicMock()
    image = np.random.rand(3, 64, 64).astype(np.float32)
    target: dict[str, Any] = {
        "boxes": np.array([[5.0, 5.0, 30.0, 30.0]]),
        "labels": np.array([0]),
    }
    prediction: dict[str, Any] = {
        "boxes": np.array([[10.0, 10.0, 40.0, 40.0]]),
        "scores": np.array([0.8]),
        "labels": np.array([0]),
    }

    _plot_single_detection(ax, image, target, prediction, ["cat"], None, 0.5)

    # 2 boxes: 1 GT + 1 pred
    assert ax.add_patch.call_count == 2
    # 2 text labels: GT label + pred label
    assert ax.text.call_count == 2


@pytest.mark.unit
def test_plot_single_detection_filters_low_confidence() -> None:
    """Predicted boxes below threshold are not drawn."""
    ax = MagicMock()
    image = np.random.rand(3, 64, 64).astype(np.float32)
    target: dict[str, Any] = {
        "boxes": np.array([[5.0, 5.0, 30.0, 30.0]]),
        "labels": np.array([0]),
    }
    prediction: dict[str, Any] = {
        "boxes": np.array([[10.0, 10.0, 40.0, 40.0]]),
        "scores": np.array([0.2]),  # below threshold
        "labels": np.array([0]),
    }

    _plot_single_detection(ax, image, target, prediction, ["cat"], None, 0.5)

    # Only GT box drawn, pred filtered out
    assert ax.add_patch.call_count == 1
    assert ax.text.call_count == 1


@pytest.mark.unit
def test_plot_single_detection_empty_predictions() -> None:
    """Handles images with no predictions gracefully."""
    ax = MagicMock()
    image = np.random.rand(3, 64, 64).astype(np.float32)
    target: dict[str, Any] = {
        "boxes": np.array([[5.0, 5.0, 30.0, 30.0]]),
        "labels": np.array([0]),
    }
    prediction: dict[str, Any] = {
        "boxes": np.empty((0, 4)),
        "scores": np.empty(0),
        "labels": np.empty(0, dtype=int),
    }

    _plot_single_detection(ax, image, target, prediction, ["cat"], None, 0.5)

    # Only GT box
    assert ax.add_patch.call_count == 1


@pytest.mark.unit
def test_plot_single_detection_empty_targets() -> None:
    """Handles images with no ground truth gracefully."""
    ax = MagicMock()
    image = np.random.rand(3, 64, 64).astype(np.float32)
    target: dict[str, Any] = {
        "boxes": np.empty((0, 4)),
        "labels": np.empty(0, dtype=int),
    }
    prediction: dict[str, Any] = {
        "boxes": np.array([[10.0, 10.0, 40.0, 40.0]]),
        "scores": np.array([0.9]),
        "labels": np.array([0]),
    }

    _plot_single_detection(ax, image, target, prediction, ["cat"], None, 0.5)

    # Only pred box
    assert ax.add_patch.call_count == 1


@pytest.mark.unit
def test_plot_single_detection_with_ctx_denormalizes(det_ctx: PlotContext) -> None:
    """When ctx has mean/std, image is denormalized."""
    ctx_with_stats = PlotContext(
        arch_name=det_ctx.arch_name,
        resolution=det_ctx.resolution,
        fig_dpi=det_ctx.fig_dpi,
        plot_style=det_ctx.plot_style,
        cmap_confusion=det_ctx.cmap_confusion,
        grid_cols=det_ctx.grid_cols,
        n_samples=det_ctx.n_samples,
        fig_size_predictions=det_ctx.fig_size_predictions,
        mean=(0.5, 0.5, 0.5),
        std=(0.2, 0.2, 0.2),
    )

    ax = MagicMock()
    image = np.random.rand(3, 64, 64).astype(np.float32)
    target: dict[str, Any] = {"boxes": np.empty((0, 4)), "labels": np.empty(0, dtype=int)}
    prediction: dict[str, Any] = {
        "boxes": np.empty((0, 4)),
        "scores": np.empty(0),
        "labels": np.empty(0, dtype=int),
    }

    # Should not raise — denormalization path exercised
    _plot_single_detection(ax, image, target, prediction, [], ctx_with_stats, 0.5)
    ax.imshow.assert_called_once()


# ── show_detections (integration) ────────────────────────────────────────────


@pytest.mark.unit
@patch("orchard.evaluation.detection_visualization._finalize_figure")
def test_show_detections_saves_figure(
    mock_finalize: MagicMock,
    tmp_path: Path,
    det_ctx: PlotContext,
) -> None:
    """show_detections creates a grid, plots detections, and finalizes."""
    import matplotlib

    matplotlib.use("Agg")

    model = _make_det_model()
    loader = _make_det_loader(batch_size=4)
    save_path = tmp_path / "det_grid.png"

    show_detections(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        classes=["person"],
        save_path=save_path,
        ctx=det_ctx,
        n=4,
    )

    model.eval.assert_called_once()
    mock_finalize.assert_called_once()


@pytest.mark.unit
@patch("orchard.evaluation.detection_visualization._finalize_figure")
@patch("orchard.evaluation.detection_visualization._get_detection_batch")
def test_show_detections_forwards_device(
    mock_batch: MagicMock,
    mock_finalize: MagicMock,
    det_ctx: PlotContext,
) -> None:
    """show_detections passes the device argument through to _get_detection_batch."""
    import matplotlib

    matplotlib.use("Agg")

    mock_batch.return_value = (
        [np.random.rand(3, 32, 32).astype(np.float32)],
        [{"boxes": np.empty((0, 4)), "labels": np.empty(0, dtype=int)}],
        [{"boxes": np.empty((0, 4)), "scores": np.empty(0), "labels": np.empty(0, dtype=int)}],
    )
    device = torch.device("cpu")

    show_detections(
        model=_make_det_model(),
        loader=_make_det_loader(),
        device=device,
        classes=["a"],
        ctx=det_ctx,
        n=1,
    )

    assert mock_batch.call_args[0][2] is device


@pytest.mark.unit
@patch("orchard.evaluation.detection_visualization._finalize_figure")
@patch("orchard.evaluation.detection_visualization._plot_single_detection")
def test_show_detections_forwards_ctx(
    mock_plot: MagicMock,
    mock_finalize: MagicMock,
    det_ctx: PlotContext,
) -> None:
    """show_detections passes ctx through to _plot_single_detection."""
    import matplotlib

    matplotlib.use("Agg")

    model = _make_det_model()
    loader = _make_det_loader(batch_size=1)

    show_detections(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        classes=["a"],
        ctx=det_ctx,
        n=1,
    )

    assert mock_plot.call_count >= 1
    assert mock_plot.call_args[0][5] is det_ctx


@pytest.mark.unit
@patch("orchard.evaluation.detection_visualization._finalize_figure")
def test_show_detections_no_ctx(mock_finalize: MagicMock, tmp_path: Path) -> None:
    """show_detections works without PlotContext (cosmetic fallbacks)."""
    import matplotlib

    matplotlib.use("Agg")

    model = _make_det_model()
    loader = _make_det_loader(batch_size=2)

    show_detections(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        classes=["obj"],
        save_path=tmp_path / "grid.png",
        n=2,
    )

    mock_finalize.assert_called_once()


@pytest.mark.unit
def test_plot_single_detection_exact_threshold_excluded() -> None:
    """Score exactly equal to threshold is NOT drawn (strict < comparison)."""
    ax = MagicMock()
    image = np.random.rand(3, 64, 64).astype(np.float32)
    target: dict[str, Any] = {"boxes": np.empty((0, 4)), "labels": np.empty(0, dtype=int)}
    prediction: dict[str, Any] = {
        "boxes": np.array([[10.0, 10.0, 40.0, 40.0]]),
        "scores": np.array([0.5]),  # exactly at threshold
        "labels": np.array([0]),
    }

    _plot_single_detection(ax, image, target, prediction, ["cat"], None, 0.5)

    # Score 0.5 < 0.5 is False, so box IS drawn
    assert ax.add_patch.call_count == 1


@pytest.mark.unit
def test_plot_single_detection_continue_not_break() -> None:
    """Low-confidence pred skips but does NOT stop iteration (continue, not break)."""
    ax = MagicMock()
    image = np.random.rand(3, 64, 64).astype(np.float32)
    target: dict[str, Any] = {"boxes": np.empty((0, 4)), "labels": np.empty(0, dtype=int)}
    prediction: dict[str, Any] = {
        "boxes": np.array([[10.0, 10.0, 40.0, 40.0], [20.0, 20.0, 50.0, 50.0]]),
        "scores": np.array([0.1, 0.9]),  # first below, second above
        "labels": np.array([0, 0]),
    }

    _plot_single_detection(ax, image, target, prediction, ["cat"], None, 0.5)

    # Only the second pred (score=0.9) should be drawn — break would miss it
    assert ax.add_patch.call_count == 1


@pytest.mark.unit
def test_plot_single_detection_pred_label_below_box() -> None:
    """Predicted labels are placed BELOW the box (bottom=True)."""
    ax = MagicMock()
    image = np.random.rand(3, 64, 64).astype(np.float32)
    target: dict[str, Any] = {"boxes": np.empty((0, 4)), "labels": np.empty(0, dtype=int)}
    prediction: dict[str, Any] = {
        "boxes": np.array([[10.0, 20.0, 50.0, 60.0]]),
        "scores": np.array([0.9]),
        "labels": np.array([0]),
    }

    _plot_single_detection(ax, image, target, prediction, ["cat"], None, 0.5)

    # Pred label should be placed below box (y > y2=60)
    text_call = ax.text.call_args
    y_pos = text_call[0][1]  # second positional arg
    assert y_pos > 60.0, f"Pred label y={y_pos} should be below y2=60"


@pytest.mark.unit
def test_show_detections_confidence_threshold_via_plot_single() -> None:
    """High confidence threshold filters out low-score predictions (unit test)."""
    ax = MagicMock()
    image = np.random.rand(3, 64, 64).astype(np.float32)
    target: dict[str, Any] = {
        "boxes": np.array([[5.0, 5.0, 30.0, 30.0]]),
        "labels": np.array([0]),
    }
    prediction: dict[str, Any] = {
        "boxes": np.array([[10.0, 10.0, 40.0, 40.0], [15.0, 15.0, 45.0, 45.0]]),
        "scores": np.array([0.9, 0.3]),  # one above, one below threshold
        "labels": np.array([0, 1]),
    }

    _plot_single_detection(ax, image, target, prediction, ["a", "b"], None, 0.5)

    # GT box (1) + high-confidence pred box (1) = 2 patches
    assert ax.add_patch.call_count == 2
    # GT label + pred label = 2 text calls
    assert ax.text.call_count == 2
