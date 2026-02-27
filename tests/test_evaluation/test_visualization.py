"""
Smoke Tests for Visualization Module.

Minimal tests to validate visualization utilities for training curves,
confusion matrices, and prediction grids.
These are essential smoke tests to boost coverage from 0% to ~30%.
"""

from __future__ import annotations

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
def ctx_rgb():
    """PlotContext for RGB 28x28 datasets."""
    return PlotContext(
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
def ctx_gray():
    """PlotContext for grayscale 28x28 datasets."""
    return PlotContext(
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
def test_plot_training_curves_basic(mock_savez, mock_plt, tmp_path, ctx_rgb):
    """Test plot_training_curves creates and saves figure."""
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    train_losses = [0.8, 0.6, 0.4, 0.2]
    val_accuracies = [0.6, 0.7, 0.8, 0.9]
    out_path = tmp_path / "curves.png"

    plot_training_curves(train_losses, val_accuracies, out_path, ctx_rgb)

    assert mock_plt.subplots.called
    assert mock_plt.savefig.called
    assert mock_plt.close.called
    mock_savez.assert_called_once()


@pytest.mark.unit
@patch("orchard.evaluation.visualization.plt")
@patch("orchard.evaluation.visualization.np.savez")
def test_plot_training_curves_empty_lists(mock_savez, mock_plt, tmp_path, ctx_gray):
    """Test plot_training_curves handles empty metric lists."""
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    out_path = tmp_path / "curves.png"

    plot_training_curves([], [], out_path, ctx_gray)


# PLOT CONFUSION MATRIX
@pytest.mark.unit
@patch("orchard.evaluation.visualization.plt")
@patch("orchard.evaluation.visualization.confusion_matrix")
def test_plot_confusion_matrix_basic(mock_cm, mock_plt, tmp_path):
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


@pytest.mark.unit
@patch("orchard.evaluation.visualization.plt")
@patch("orchard.evaluation.visualization.confusion_matrix")
def test_plot_confusion_matrix_with_nan(mock_cm, mock_plt, tmp_path, ctx_gray):
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
def test_show_predictions_basic(mock_get_batch, mock_plt, tmp_path, ctx_rgb):
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


@pytest.mark.unit
@patch("orchard.evaluation.visualization.plt")
@patch("orchard.evaluation.visualization._get_predictions_batch")
def test_show_predictions_without_ctx(mock_get_batch, mock_plt):
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
def test_show_predictions_without_save_path(mock_get_batch, mock_plt, ctx_rgb):
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
def test_show_predictions_standard_mode(mock_get_batch, mock_plt, tmp_path, ctx_gray):
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
def test_show_predictions_with_custom_n(mock_get_batch, mock_plt, tmp_path):
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
def test_get_predictions_batch_directly():
    """Test _get_predictions_batch function directly."""
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


@pytest.mark.unit
def test_setup_prediction_grid_directly(ctx_rgb):
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
def test_finalize_figure_with_save(tmp_path, ctx_rgb):
    """Test _finalize_figure with save_path."""
    from orchard.evaluation.visualization import _finalize_figure

    mock_plt = MagicMock()
    save_path = tmp_path / "test.png"

    _finalize_figure(mock_plt, save_path, ctx_rgb)

    mock_plt.savefig.assert_called_once()
    mock_plt.show.assert_not_called()
    mock_plt.close.assert_called_once()


@pytest.mark.unit
def test_finalize_figure_without_save():
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
def test_denormalize_image():
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
def test_denormalize_image_rgb():
    """Test _denormalize_image handles RGB images."""
    img = np.zeros((3, 2, 2))

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
    assert 0.0 <= result.min() <= 1.0
    assert 0.0 <= result.max() <= 1.0


@pytest.mark.unit
def test_denormalize_image_clips_values():
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
def test_prepare_for_plt_chw_to_hwc():
    """Test _prepare_for_plt converts (C, H, W) to (H, W, C)."""
    rng = np.random.default_rng(seed=42)
    img = rng.random(size=(3, 28, 28))

    result = _prepare_for_plt(img)

    assert result.shape == (28, 28, 3)


@pytest.mark.unit
def test_prepare_for_plt_grayscale_squeeze():
    """Test _prepare_for_plt squeezes single-channel dimension."""
    rng = np.random.default_rng(seed=42)
    img = rng.random(size=(1, 28, 28))

    result = _prepare_for_plt(img)

    assert result.shape == (28, 28)


@pytest.mark.unit
def test_prepare_for_plt_already_2d():
    """Test _prepare_for_plt handles already 2D images."""
    rng = np.random.default_rng(seed=42)
    img = rng.random(size=(28, 28))

    result = _prepare_for_plt(img)

    assert result.shape == (28, 28)


@pytest.mark.unit
def test_plot_context_from_config():
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

    ctx = PlotContext.from_config(mock_cfg)

    assert ctx.arch_name == "resnet18"
    assert ctx.resolution == 28
    assert ctx.fig_dpi == 200
    assert ctx.is_anatomical is True
    assert ctx.is_texture_based is False


@pytest.mark.unit
def test_plot_context_from_config_no_metadata():
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

    ctx = PlotContext.from_config(mock_cfg)

    assert ctx.is_anatomical is False
    assert ctx.is_texture_based is False
    assert ctx.use_tta is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
