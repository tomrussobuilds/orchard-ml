"""
Smoke Tests for Visualization Module.

Minimal tests to validate visualization utilities for training curves,
confusion matrices, and prediction grids.
These are essential smoke tests to boost coverage from 0% to ~30%.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from orchard.evaluation.plot_context import PlotContext
from orchard.evaluation.visualization import (
    _denormalize_image,
    _prepare_for_plt,
    plot_confusion_matrix,
    plot_training_curves,
    show_predictions,
)


# FIXTURES
@pytest.fixture
def ctx_rgb() -> None:
    """PlotContext for RGB 28x28 datasets."""
    return PlotContext(  # type: ignore
        arch_name="resnet18",
        resolution=28,
        fig_dpi=200,
        plot_style="seaborn-v0_8-muted",
        cmap_confusion="Blues",
        grid_cols=4,
        n_samples=12,
        fig_size_predictions=(12, 9),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        use_tta=False,
        is_anatomical=True,
        is_texture_based=False,
    )


@pytest.fixture
def ctx_gray() -> None:
    """PlotContext for grayscale 28x28 datasets."""
    return PlotContext(  # type: ignore
        arch_name="model",
        resolution=28,
        fig_dpi=150,
        plot_style="seaborn-v0_8-muted",
        cmap_confusion="viridis",
        grid_cols=3,
        n_samples=6,
        fig_size_predictions=(9, 6),
        mean=(0.5,),
        std=(0.5,),
        use_tta=False,
        is_anatomical=False,
        is_texture_based=False,
    )


# PLOT TRAINING CURVES
@pytest.mark.unit
@patch("orchard.evaluation.visualization.plt")
@patch("orchard.evaluation.visualization.np.savez")
def test_plot_training_curves_basic(  # type: ignore
    mock_savez: MagicMock, mock_plt: MagicMock, tmp_path: Path, ctx_rgb
) -> None:
    """Test plot_training_curves creates and saves figure."""
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    train_losses = [0.8, 0.6, 0.4, 0.2]
    val_metric_values = [0.6, 0.7, 0.8, 0.9]
    out_path = tmp_path / "curves.png"

    plot_training_curves(
        train_losses, val_metric_values, out_path, ctx_rgb, val_label="Validation Accuracy"
    )

    assert mock_plt.subplots.called
    assert mock_plt.savefig.called
    assert mock_plt.close.called
    mock_savez.assert_called_once()


@pytest.mark.unit
@patch("orchard.evaluation.visualization.plt")
@patch("orchard.evaluation.visualization.np.savez")
def test_plot_training_curves_empty_lists(  # type: ignore
    mock_savez: MagicMock, mock_plt: MagicMock, tmp_path: Path, ctx_gray
) -> None:
    """Test plot_training_curves handles empty metric lists."""
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    out_path = tmp_path / "curves.png"

    plot_training_curves([], [], out_path, ctx_gray, val_label="Validation Accuracy")

    assert mock_plt.subplots.called
    assert mock_plt.savefig.called
    assert mock_plt.close.called


# PLOT CONFUSION MATRIX
@pytest.mark.unit
@patch("orchard.evaluation.visualization.ConfusionMatrixDisplay")
@patch("orchard.evaluation.visualization.plt")
@patch("orchard.evaluation.visualization.confusion_matrix")
def test_plot_confusion_matrix_basic(
    mock_cm: MagicMock, mock_plt: MagicMock, mock_cmd_cls: MagicMock, tmp_path: Path
) -> None:
    """Test plot_confusion_matrix creates and saves figure."""
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    all_labels = np.array([0, 1, 2, 0, 1, 2])
    all_preds = np.array([0, 1, 1, 0, 2, 2])
    classes = ["class0", "class1", "class2"]
    out_path = tmp_path / "confusion.png"

    ctx = PlotContext(
        arch_name="efficientnet",
        resolution=224,
        fig_dpi=200,
        plot_style="seaborn-v0_8-muted",
        cmap_confusion="Blues",
        grid_cols=4,
        n_samples=12,
        fig_size_predictions=(12, 9),
    )

    mock_cm.return_value = np.eye(3)

    plot_confusion_matrix(all_labels, all_preds, classes, out_path, ctx)

    mock_cm.assert_called_once()
    assert mock_plt.subplots.called
    assert mock_plt.savefig.called or mock_plt.close.called

    # Verify display_labels forwarded (not None, not removed)
    cmd_kwargs = mock_cmd_cls.call_args[1]
    assert cmd_kwargs["display_labels"] is classes


@pytest.mark.unit
@patch("orchard.evaluation.visualization.plt")
@patch("orchard.evaluation.visualization.confusion_matrix")
def test_plot_confusion_matrix_with_nan(  # type: ignore
    mock_cm: MagicMock, mock_plt: MagicMock, tmp_path: Path, ctx_gray
) -> None:
    """Test plot_confusion_matrix handles NaN values in matrix."""
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    all_labels = np.array([0, 1, 0])
    all_preds = np.array([0, 1, 0])
    classes = ["class0", "class1", "class2"]
    out_path = tmp_path / "confusion.png"

    mock_cm.return_value = np.array([[1.0, 0.0, np.nan], [0.0, 1.0, np.nan], [0.0, 0.0, 0.0]])

    plot_confusion_matrix(all_labels, all_preds, classes, out_path, ctx_gray)


# SHOW PREDICTIONS
@pytest.mark.unit
@patch("orchard.evaluation.visualization.plt")
@patch("orchard.evaluation.visualization._get_predictions_batch")
def test_show_predictions_basic(  # type: ignore
    mock_get_batch: MagicMock, mock_plt: MagicMock, tmp_path: Path, ctx_rgb
) -> None:
    """Test show_predictions creates prediction grid."""
    mock_fig = MagicMock()
    mock_axes = [MagicMock() for _ in range(12)]
    mock_plt.subplots.return_value = (mock_fig, np.array(mock_axes))

    rng = np.random.default_rng(seed=42)
    images = rng.random(size=(12, 3, 28, 28))
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    preds = np.array([0, 1, 1, 0, 2, 2, 0, 1, 2, 0, 1, 2])
    mock_get_batch.return_value = (images, labels, preds)

    mock_model = MagicMock()
    mock_loader = MagicMock()
    device = torch.device("cpu")
    classes = ["class0", "class1", "class2"]
    save_path = tmp_path / "predictions.png"

    show_predictions(mock_model, mock_loader, device, classes, save_path, ctx_rgb)

    mock_model.eval.assert_called_once()
    assert mock_plt.subplots.called
    assert mock_plt.savefig.called

    # Verify forwarding: _get_predictions_batch receives real args (not None)
    batch_args = mock_get_batch.call_args[0]
    assert batch_args[0] is mock_model
    assert batch_args[1] is mock_loader
    assert batch_args[2] is device


@pytest.mark.unit
def test_show_predictions_forwards_ctx(tmp_path: Path, ctx_rgb: Any) -> None:
    """Test show_predictions forwards ctx to _plot_single_prediction (not None)."""
    rng = np.random.default_rng(seed=42)
    images = rng.random(size=(12, 3, 28, 28))
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    preds = np.array([0, 1, 1, 0, 2, 2, 0, 1, 2, 0, 1, 2])

    mock_model = MagicMock()
    mock_loader = MagicMock()
    device = torch.device("cpu")
    classes = ["class0", "class1", "class2"]
    save_path = tmp_path / "predictions.png"

    with (
        patch("orchard.evaluation.visualization._get_predictions_batch") as mock_get_batch,
        patch("orchard.evaluation.visualization.plt") as mock_plt,
        patch("orchard.evaluation.visualization._plot_single_prediction") as mock_plot,
    ):
        mock_fig = MagicMock()
        mock_axes = np.empty(12, dtype=object)
        for idx in range(12):
            mock_axes[idx] = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        mock_get_batch.return_value = (images, labels, preds)

        show_predictions(mock_model, mock_loader, device, classes, save_path, ctx_rgb)

    assert mock_plot.call_count == 12
    # Every call should receive ctx_rgb as last arg, not None
    for call in mock_plot.call_args_list:
        assert call[0][5] is ctx_rgb


@pytest.mark.unit
@patch("orchard.evaluation.visualization.plt")
@patch("orchard.evaluation.visualization._get_predictions_batch")
def test_show_predictions_without_ctx(mock_get_batch: MagicMock, mock_plt: MagicMock) -> None:
    """Test show_predictions works without PlotContext (uses defaults)."""
    mock_fig = MagicMock()
    mock_axes = np.empty((3, 4), dtype=object)
    for i in range(3):
        for j in range(4):
            mock_axes[i, j] = MagicMock()

    mock_plt.subplots.return_value = (mock_fig, mock_axes)

    rng = np.random.default_rng(seed=42)
    images = rng.random(size=(12, 3, 28, 28))
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    preds = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    mock_get_batch.return_value = (images, labels, preds)

    mock_model = MagicMock()
    mock_loader = MagicMock()
    device = torch.device("cpu")
    classes = ["class0", "class1"]

    show_predictions(mock_model, mock_loader, device, classes, save_path=None, ctx=None)

    mock_model.eval.assert_called_once()
    assert mock_plt.subplots.called


@pytest.mark.unit
@patch("orchard.evaluation.visualization.plt")
@patch("orchard.evaluation.visualization._get_predictions_batch")
def test_show_predictions_without_save_path(  # type: ignore
    mock_get_batch: MagicMock, mock_plt: MagicMock, ctx_rgb
) -> None:
    """Test show_predictions with save_path=None (interactive mode)."""
    mock_fig = MagicMock()
    mock_axes = [MagicMock() for _ in range(12)]
    mock_plt.subplots.return_value = (mock_fig, np.array(mock_axes))

    rng = np.random.default_rng(seed=42)
    images = rng.random(size=(12, 3, 28, 28))
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    preds = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    mock_get_batch.return_value = (images, labels, preds)

    mock_model = MagicMock()
    mock_loader = MagicMock()
    device = torch.device("cpu")
    classes = ["class0", "class1"]

    show_predictions(mock_model, mock_loader, device, classes, save_path=None, ctx=ctx_rgb)

    mock_plt.show.assert_called_once()


@pytest.mark.unit
@patch("orchard.evaluation.visualization.plt")
@patch("orchard.evaluation.visualization._get_predictions_batch")
def test_show_predictions_standard_mode(  # type: ignore
    mock_get_batch: MagicMock, mock_plt: MagicMock, tmp_path: Path, ctx_gray
) -> None:
    """Test show_predictions with standard mode (neither texture nor anatomical)."""
    mock_fig = MagicMock()
    mock_axes = [MagicMock() for _ in range(6)]
    mock_plt.subplots.return_value = (mock_fig, np.array(mock_axes))

    rng = np.random.default_rng(seed=42)
    images = rng.random(size=(6, 1, 28, 28))
    labels = np.array([0, 1, 0, 1, 0, 1])
    preds = np.array([0, 1, 0, 1, 0, 1])
    mock_get_batch.return_value = (images, labels, preds)

    mock_model = MagicMock()
    mock_loader = MagicMock()
    device = torch.device("cpu")
    classes = ["class0", "class1"]
    save_path = tmp_path / "predictions.png"

    show_predictions(mock_model, mock_loader, device, classes, save_path, ctx_gray, n=6)

    assert mock_plt.subplots.called
    assert mock_plt.savefig.called


@pytest.mark.unit
@patch("orchard.evaluation.visualization.plt")
@patch("orchard.evaluation.visualization._get_predictions_batch")
def test_show_predictions_with_custom_n(
    mock_get_batch: MagicMock, mock_plt: MagicMock, tmp_path: Path
) -> None:
    """Test show_predictions respects custom n parameter."""
    mock_fig = MagicMock()
    mock_axes = [MagicMock() for _ in range(6)]
    mock_plt.subplots.return_value = (mock_fig, np.array(mock_axes))

    rng = np.random.default_rng(seed=42)
    images = rng.random((6, 3, 28, 28))
    labels = np.array([0, 1, 2, 0, 1, 2])
    preds = np.array([0, 1, 1, 0, 2, 2])
    mock_get_batch.return_value = (images, labels, preds)

    mock_model = MagicMock()
    mock_loader = MagicMock()
    device = torch.device("cpu")
    classes = ["class0", "class1", "class2"]
    save_path = tmp_path / "predictions.png"

    ctx = PlotContext(
        arch_name="vit",
        resolution=224,
        fig_dpi=150,
        plot_style="seaborn-v0_8-muted",
        cmap_confusion="Blues",
        grid_cols=3,
        n_samples=12,
        fig_size_predictions=(9, 6),
        is_texture_based=True,
        is_anatomical=False,
        use_tta=True,
    )

    show_predictions(mock_model, mock_loader, device, classes, save_path, ctx, n=6)

    mock_get_batch.assert_called_once()
    assert mock_get_batch.call_args[0][3] == 6


# HELPER FUNCTIONS - DIRECT TESTS
@pytest.mark.unit
def test_get_predictions_batch_directly() -> None:
    """Test _get_predictions_batch passes device and images correctly to model."""
    from orchard.evaluation.visualization import _get_predictions_batch

    mock_model = MagicMock()
    mock_model.return_value = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])

    images = torch.randn(5, 3, 28, 28)
    labels = torch.tensor([0, 1, 0, 1, 0])
    mock_loader = MagicMock()
    mock_loader.__iter__ = MagicMock(return_value=iter([(images, labels)]))

    device = torch.device("cpu")

    img_arr, label_arr, pred_arr = _get_predictions_batch(mock_model, mock_loader, device, n=3)

    assert img_arr.shape == (3, 3, 28, 28)
    assert label_arr.shape == (3,)
    assert pred_arr.shape == (3,)
    assert isinstance(img_arr, np.ndarray)
    assert isinstance(label_arr, np.ndarray)
    assert isinstance(pred_arr, np.ndarray)

    # Verify model received actual images (not None)
    model_input = mock_model.call_args[0][0]
    assert isinstance(model_input, torch.Tensor)
    assert model_input.shape == (3, 3, 28, 28)


@pytest.mark.unit
def test_setup_prediction_grid_directly(ctx_rgb: Any) -> None:
    """Test _setup_prediction_grid function directly."""
    from orchard.evaluation.visualization import _setup_prediction_grid

    with patch("orchard.evaluation.visualization.plt.subplots") as mock_subplots:
        mock_fig = MagicMock()
        mock_axes = np.empty((3, 4), dtype=object)
        for i in range(3):
            for j in range(4):
                mock_axes[i, j] = MagicMock()

        mock_subplots.return_value = (mock_fig, mock_axes)

        _, axes = _setup_prediction_grid(12, 4, ctx_rgb)

        mock_subplots.assert_called_once()
        assert len(axes) == 12


@pytest.mark.unit
def test_finalize_figure_with_save(tmp_path: Path, ctx_rgb: Any) -> None:
    """Test _finalize_figure with save_path."""
    from orchard.evaluation.visualization import _finalize_figure

    mock_plt = MagicMock()
    save_path = tmp_path / "test.png"

    _finalize_figure(mock_plt, save_path, ctx_rgb)

    mock_plt.savefig.assert_called_once()
    mock_plt.show.assert_not_called()
    mock_plt.close.assert_called_once()


@pytest.mark.unit
def test_finalize_figure_creates_nested_dirs(tmp_path: Path, ctx_rgb: Any) -> None:
    """Test _finalize_figure creates nested parent directories."""
    from orchard.evaluation.visualization import _finalize_figure

    mock_plt = MagicMock()
    save_path = tmp_path / "deep" / "nested" / "dir" / "fig.png"

    _finalize_figure(mock_plt, save_path, ctx_rgb)

    assert save_path.parent.exists()
    mock_plt.savefig.assert_called_once()


@pytest.mark.unit
def test_finalize_figure_without_save() -> None:
    """Test _finalize_figure without save_path (interactive mode)."""
    from orchard.evaluation.visualization import _finalize_figure

    mock_plt = MagicMock()

    ctx = PlotContext(
        arch_name="model",
        resolution=28,
        fig_dpi=150,
        plot_style="seaborn-v0_8-muted",
        cmap_confusion="Blues",
        grid_cols=4,
        n_samples=12,
        fig_size_predictions=(12, 9),
    )

    _finalize_figure(mock_plt, save_path=None, ctx=ctx)

    mock_plt.show.assert_called_once()
    mock_plt.savefig.assert_not_called()
    mock_plt.close.assert_called_once()


# HELPER FUNCTIONS - DENORMALIZE & PREPARE
@pytest.mark.unit
def test_denormalize_image() -> None:
    """Test _denormalize_image reverses normalization."""
    img = np.array([[[0.0, 0.0], [0.0, 0.0]]])

    ctx = PlotContext(
        arch_name="m",
        resolution=28,
        fig_dpi=200,
        plot_style="seaborn-v0_8-muted",
        cmap_confusion="Blues",
        grid_cols=4,
        n_samples=12,
        fig_size_predictions=(12, 9),
        mean=(0.5,),
        std=(0.5,),
    )

    result = _denormalize_image(img, ctx)

    expected = np.array([[[0.5, 0.5], [0.5, 0.5]]])
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.unit
def test_denormalize_image_rgb() -> None:
    """Test _denormalize_image reverses normalization with non-zero values."""
    # Use non-zero image so * vs / produces different results
    img = np.ones((3, 2, 2)) * 2.0

    ctx = PlotContext(
        arch_name="m",
        resolution=28,
        fig_dpi=200,
        plot_style="seaborn-v0_8-muted",
        cmap_confusion="Blues",
        grid_cols=4,
        n_samples=12,
        fig_size_predictions=(12, 9),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    result = _denormalize_image(img, ctx)

    assert result.shape == (3, 2, 2)
    # Verify denormalization formula: (img * std) + mean, clipped to [0,1]
    # Channel 0: (2.0 * 0.229) + 0.485 = 0.943
    expected_ch0 = (2.0 * 0.229) + 0.485
    assert result[0, 0, 0] == pytest.approx(expected_ch0, abs=1e-5)


@pytest.mark.unit
def test_denormalize_image_clips_values() -> None:
    """Test _denormalize_image clips values to [0, 1]."""
    img = np.full((1, 2, 2), 10.0)

    ctx = PlotContext(
        arch_name="m",
        resolution=28,
        fig_dpi=200,
        plot_style="seaborn-v0_8-muted",
        cmap_confusion="Blues",
        grid_cols=4,
        n_samples=12,
        fig_size_predictions=(12, 9),
        mean=(0.5,),
        std=(0.5,),
    )

    result = _denormalize_image(img, ctx)

    assert result.max() == pytest.approx(1.0)


@pytest.mark.unit
def test_prepare_for_plt_chw_to_hwc() -> None:
    """Test _prepare_for_plt converts (C, H, W) to (H, W, C) with correct axes."""
    rng = np.random.default_rng(seed=42)
    img = rng.random(size=(3, 28, 28))

    result = _prepare_for_plt(img)

    assert result.shape == (28, 28, 3)
    # Verify pixel values match transposed layout (not just shape)
    assert result[0, 0, 0] == pytest.approx(img[0, 0, 0])
    assert result[0, 0, 1] == pytest.approx(img[1, 0, 0])
    assert result[0, 0, 2] == pytest.approx(img[2, 0, 0])


@pytest.mark.unit
def test_prepare_for_plt_grayscale_squeeze() -> None:
    """Test _prepare_for_plt squeezes single-channel to 2D and preserves values."""
    rng = np.random.default_rng(seed=42)
    img = rng.random(size=(1, 28, 28))

    result = _prepare_for_plt(img)

    assert result.shape == (28, 28)
    # squeeze(-1) on axis=-1: verify pixel value preserved
    assert result[0, 0] == pytest.approx(img[0, 0, 0])


@pytest.mark.unit
def test_prepare_for_plt_already_2d() -> None:
    """Test _prepare_for_plt handles already 2D images."""
    rng = np.random.default_rng(seed=42)
    img = rng.random(size=(28, 28))

    result = _prepare_for_plt(img)

    assert result.shape == (28, 28)


@pytest.mark.unit
def test_plot_context_from_config() -> None:
    """Test PlotContext.from_config builds from a mock Config."""
    mock_cfg = MagicMock()
    mock_cfg.architecture.name = "resnet18"
    mock_cfg.dataset.resolution = 28
    mock_cfg.dataset.mean = (0.5,)
    mock_cfg.dataset.std = (0.5,)
    mock_cfg.evaluation.fig_dpi = 200
    mock_cfg.evaluation.plot_style = "seaborn-v0_8-muted"
    mock_cfg.evaluation.cmap_confusion = "Blues"
    mock_cfg.evaluation.grid_cols = 4
    mock_cfg.evaluation.n_samples = 12
    mock_cfg.evaluation.fig_size_predictions = (12, 9)
    mock_cfg.training.use_tta = False
    mock_cfg.dataset.metadata.is_anatomical = True
    mock_cfg.dataset.metadata.is_texture_based = False
    mock_cfg.dataset.effective_is_anatomical = True
    mock_cfg.dataset.effective_is_texture_based = False

    ctx = PlotContext.from_config(mock_cfg)

    assert ctx.arch_name == "resnet18"
    assert ctx.resolution == 28
    assert ctx.fig_dpi == 200
    assert ctx.is_anatomical is True
    assert ctx.is_texture_based is False


@pytest.mark.unit
def test_plot_context_from_config_no_metadata() -> None:
    """Test PlotContext.from_config with metadata=None."""
    mock_cfg = MagicMock()
    mock_cfg.architecture.name = "model"
    mock_cfg.dataset.resolution = 224
    mock_cfg.dataset.mean = (0.5, 0.5, 0.5)
    mock_cfg.dataset.std = (0.5, 0.5, 0.5)
    mock_cfg.evaluation.fig_dpi = 150
    mock_cfg.evaluation.plot_style = "seaborn-v0_8-muted"
    mock_cfg.evaluation.cmap_confusion = "viridis"
    mock_cfg.evaluation.grid_cols = 3
    mock_cfg.evaluation.n_samples = 6
    mock_cfg.evaluation.fig_size_predictions = (9, 6)
    mock_cfg.training.use_tta = True
    mock_cfg.dataset.metadata = None
    mock_cfg.dataset.effective_is_anatomical = True
    mock_cfg.dataset.effective_is_texture_based = True

    ctx = PlotContext.from_config(mock_cfg)

    assert ctx.is_anatomical is True  # conservative default when metadata absent
    assert ctx.is_texture_based is True  # conservative default when metadata absent
    assert ctx.use_tta is True


@pytest.mark.unit
def test_plot_single_prediction_calls_denormalize(ctx_rgb: Any) -> None:
    """Test _plot_single_prediction forwards image and ctx to _denormalize_image."""
    from orchard.evaluation.visualization import _plot_single_prediction

    ax = MagicMock()
    image = np.ones((3, 4, 4)) * 2.0
    classes = ["cat", "dog"]

    with patch("orchard.evaluation.visualization._denormalize_image") as mock_denorm:
        mock_denorm.return_value = np.ones((3, 4, 4))
        _plot_single_prediction(ax, image, label=0, pred=1, classes=classes, ctx=ctx_rgb)

        mock_denorm.assert_called_once()
        call_img, call_ctx = mock_denorm.call_args[0]
        np.testing.assert_array_equal(call_img, image)
        assert call_ctx is ctx_rgb


@pytest.mark.unit
def test_plot_single_prediction_skips_denormalize_without_ctx() -> None:
    """Test _plot_single_prediction skips denormalization when ctx is None."""
    from orchard.evaluation.visualization import _plot_single_prediction

    ax = MagicMock()
    image = np.ones((3, 4, 4))
    classes = ["cat", "dog"]

    with patch("orchard.evaluation.visualization._denormalize_image") as mock_denorm:
        _plot_single_prediction(ax, image, label=0, pred=0, classes=classes, ctx=None)
        mock_denorm.assert_not_called()


@pytest.mark.unit
def test_plot_single_prediction_correct_vs_incorrect() -> None:
    """Test _plot_single_prediction uses green for correct, red for incorrect."""
    from orchard.evaluation.visualization import _plot_single_prediction

    classes = ["cat", "dog"]
    image = np.ones((3, 4, 4))

    # Correct prediction
    ax_correct = MagicMock()
    _plot_single_prediction(ax_correct, image, label=0, pred=0, classes=classes, ctx=None)
    title_kwargs = ax_correct.set_title.call_args
    assert title_kwargs[1]["color"] == "green"

    # Incorrect prediction
    ax_wrong = MagicMock()
    _plot_single_prediction(ax_wrong, image, label=0, pred=1, classes=classes, ctx=None)
    title_kwargs = ax_wrong.set_title.call_args
    assert title_kwargs[1]["color"] == "red"


@pytest.mark.unit
def test_plot_single_prediction_calls_prepare_for_plt() -> None:
    """Test _plot_single_prediction passes denormalized image to _prepare_for_plt."""
    from orchard.evaluation.visualization import _plot_single_prediction

    ax = MagicMock()
    image = np.ones((3, 4, 4))
    classes = ["cat", "dog"]

    with patch("orchard.evaluation.visualization._prepare_for_plt") as mock_prep:
        mock_prep.return_value = np.ones((4, 4, 3))
        _plot_single_prediction(ax, image, label=0, pred=0, classes=classes, ctx=None)
        mock_prep.assert_called_once()


@pytest.mark.unit
def test_get_predictions_batch_passes_device() -> None:
    """Test _get_predictions_batch calls .to() with actual device."""
    from orchard.evaluation.visualization import _get_predictions_batch

    mock_model = MagicMock()
    mock_model.return_value = torch.tensor([[0.1, 0.9], [0.8, 0.2]])

    images = torch.randn(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 0, 1])
    mock_loader = MagicMock()
    mock_loader.__iter__ = MagicMock(return_value=iter([(images, labels)]))

    device = torch.device("cpu")
    _get_predictions_batch(mock_model, mock_loader, device, n=2)

    # The tensor passed to model should be on the correct device
    model_input = mock_model.call_args[0][0]
    assert model_input.device == device


@pytest.mark.unit
def test_setup_prediction_grid_row_calculation() -> None:
    """Test _setup_prediction_grid computes rows = ceil(n / cols)."""
    from orchard.evaluation.visualization import _setup_prediction_grid

    with patch("orchard.evaluation.visualization.plt.subplots") as mock_sub:
        mock_fig = MagicMock()
        mock_axes = np.array([MagicMock() for _ in range(8)])
        mock_sub.return_value = (mock_fig, mock_axes)

        _setup_prediction_grid(7, 4, None)

        call_args = mock_sub.call_args
        assert call_args[0][0] == 2  # ceil(7/4) = 2 rows
        assert call_args[0][1] == 4  # 4 cols


@pytest.mark.unit
@patch("orchard.evaluation.visualization.plt")
@patch("orchard.evaluation.visualization.np.savez")
def test_plot_training_curves_npz_data(  # type: ignore
    mock_savez: MagicMock, mock_plt: MagicMock, tmp_path: Path, ctx_rgb
) -> None:
    """Test plot_training_curves saves correct data in npz file."""
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    train_losses = [0.8, 0.6, 0.4]
    val_metric_values = [0.6, 0.7, 0.8]
    out_path = tmp_path / "curves.png"

    plot_training_curves(
        train_losses, val_metric_values, out_path, ctx_rgb, val_label="Validation Accuracy"
    )

    # Verify npz called with the correct path and data
    call_args = mock_savez.call_args
    npz_path = call_args[0][0]
    assert str(npz_path).endswith(".npz")
    assert call_args[1]["train_losses"] == train_losses
    assert call_args[1]["val_metric_values"] == val_metric_values


@pytest.mark.unit
def test_denormalize_image_reshape_channels() -> None:
    """Test _denormalize_image reshapes mean/std per-channel (not -2)."""
    # 3-channel image: if reshape used -2 instead of -1, shape would differ
    img = np.zeros((3, 4, 4))
    ctx = PlotContext(
        arch_name="m",
        resolution=28,
        fig_dpi=200,
        plot_style="seaborn-v0_8-muted",
        cmap_confusion="Blues",
        grid_cols=4,
        n_samples=12,
        fig_size_predictions=(12, 9),
        mean=(0.1, 0.2, 0.3),
        std=(0.4, 0.5, 0.6),
    )
    result = _denormalize_image(img, ctx)
    assert result.shape == (3, 4, 4)
    # Each channel should equal its mean (img=0, so 0*std + mean = mean)
    np.testing.assert_array_almost_equal(result[0], 0.1)
    np.testing.assert_array_almost_equal(result[1], 0.2)
    np.testing.assert_array_almost_equal(result[2], 0.3)


@pytest.mark.unit
def test_denormalize_image_clip_lower_bound() -> None:
    """Test _denormalize_image clips negative values to 0."""
    img = np.full((1, 2, 2), -10.0)
    ctx = PlotContext(
        arch_name="m",
        resolution=28,
        fig_dpi=200,
        plot_style="seaborn-v0_8-muted",
        cmap_confusion="Blues",
        grid_cols=4,
        n_samples=12,
        fig_size_predictions=(12, 9),
        mean=(0.5,),
        std=(0.5,),
    )
    result = _denormalize_image(img, ctx)
    assert result.min() == pytest.approx(0.0)


@pytest.mark.unit
def test_prepare_for_plt_transpose_axes() -> None:
    """Test _prepare_for_plt transposes (C,H,W) to (H,W,C) not arbitrary."""
    img = np.zeros((3, 4, 5))  # C=3, H=4, W=5
    img[0, :, :] = 1.0  # channel 0 = 1
    img[1, :, :] = 2.0  # channel 1 = 2

    result = _prepare_for_plt(img)
    assert result.shape == (4, 5, 3)  # H, W, C
    assert result[0, 0, 0] == 1.0  # channel 0 preserved
    assert result[0, 0, 1] == 2.0  # channel 1 preserved


@pytest.mark.unit
def test_prepare_for_plt_squeeze_axis() -> None:
    """Test _prepare_for_plt squeezes axis -1 specifically (not None)."""
    # squeeze(None) would also remove other size-1 dims;
    # the function only handles ndim==3, so test with ndim==3
    img_3d = np.zeros((1, 4, 4))  # C=1, H=4, W=4
    img_3d[0, 0, 0] = 42.0

    result = _prepare_for_plt(img_3d)
    # After transpose: (4,4,1), then squeeze(-1) -> (4,4)
    assert result.shape == (4, 4)
    assert result[0, 0] == 42.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
