"""
Smoke Tests for Visualization Module.

Minimal tests to validate visualization utilities for training curves,
confusion matrices, and prediction grids.
These are essential smoke tests to boost coverage from 0% to ~30%.
"""

# Standard Imports
from unittest.mock import MagicMock, patch

# Third-Party Imports
import numpy as np
import pytest
import torch

# Internal Imports
from orchard.evaluation.visualization import (
    _denormalize_image,
    _prepare_for_plt,
    plot_confusion_matrix,
    plot_training_curves,
    show_predictions,
)


# PLOT TRAINING CURVES
@pytest.mark.unit
@patch("orchard.evaluation.visualization.plt")
@patch("orchard.evaluation.visualization.np.savez")
def test_plot_training_curves_basic(mock_savez, mock_plt, tmp_path):
    """Test plot_training_curves creates and saves figure."""
    # Configure mock to return (fig, ax) tuple
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    train_losses = [0.8, 0.6, 0.4, 0.2]
    val_accuracies = [0.6, 0.7, 0.8, 0.9]
    out_path = tmp_path / "curves.png"

    mock_cfg = MagicMock()
    mock_cfg.model.name = "resnet18"
    mock_cfg.dataset.resolution = 28
    mock_cfg.evaluation.fig_dpi = 200

    plot_training_curves(train_losses, val_accuracies, out_path, mock_cfg)

    # Verify plot methods were called
    assert mock_plt.subplots.called
    assert mock_plt.savefig.called
    assert mock_plt.close.called
    # Verify npz export
    mock_savez.assert_called_once()


@pytest.mark.unit
@patch("orchard.evaluation.visualization.plt")
@patch("orchard.evaluation.visualization.np.savez")
def test_plot_training_curves_empty_lists(mock_savez, mock_plt, tmp_path):
    """Test plot_training_curves handles empty metric lists."""
    # Configure mock to return (fig, ax) tuple
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    train_losses = []
    val_accuracies = []
    out_path = tmp_path / "curves.png"

    mock_cfg = MagicMock()
    mock_cfg.model.name = "model"
    mock_cfg.dataset.resolution = 224
    mock_cfg.evaluation.fig_dpi = 150

    # Should not crash
    plot_training_curves(train_losses, val_accuracies, out_path, mock_cfg)


# PLOT CONFUSION MATRIX
@pytest.mark.unit
@patch("orchard.evaluation.visualization.plt")
@patch("orchard.evaluation.visualization.confusion_matrix")
def test_plot_confusion_matrix_basic(mock_cm, mock_plt, tmp_path):
    """Test plot_confusion_matrix creates and saves figure."""
    # Configure mock to return (fig, ax) tuple
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    all_labels = np.array([0, 1, 2, 0, 1, 2])
    all_preds = np.array([0, 1, 1, 0, 2, 2])
    classes = ["class0", "class1", "class2"]
    out_path = tmp_path / "confusion.png"

    mock_cfg = MagicMock()
    mock_cfg.model.name = "efficientnet"
    mock_cfg.dataset.resolution = 224
    mock_cfg.evaluation.fig_dpi = 200
    mock_cfg.evaluation.cmap_confusion = "Blues"

    # Mock confusion_matrix to return valid matrix
    mock_cm.return_value = np.eye(3)

    plot_confusion_matrix(all_labels, all_preds, classes, out_path, mock_cfg)

    # Verify confusion_matrix was called with correct args
    mock_cm.assert_called_once()
    assert mock_plt.subplots.called
    assert mock_plt.savefig.called or mock_plt.close.called


@pytest.mark.unit
@patch("orchard.evaluation.visualization.plt")
@patch("orchard.evaluation.visualization.confusion_matrix")
def test_plot_confusion_matrix_with_nan(mock_cm, mock_plt, tmp_path):
    """Test plot_confusion_matrix handles NaN values in matrix."""
    # Configure mock to return (fig, ax) tuple
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    all_labels = np.array([0, 1, 0])
    all_preds = np.array([0, 1, 0])
    classes = ["class0", "class1", "class2"]
    out_path = tmp_path / "confusion.png"

    mock_cfg = MagicMock()
    mock_cfg.model.name = "model"
    mock_cfg.dataset.resolution = 28
    mock_cfg.evaluation.fig_dpi = 150
    mock_cfg.evaluation.cmap_confusion = "viridis"

    # Return matrix with NaN (e.g., class2 never seen)
    mock_cm.return_value = np.array([[1.0, 0.0, np.nan], [0.0, 1.0, np.nan], [0.0, 0.0, 0.0]])

    # Should handle NaN gracefully with np.nan_to_num
    plot_confusion_matrix(all_labels, all_preds, classes, out_path, mock_cfg)


# SHOW PREDICTIONS
@pytest.mark.unit
@patch("orchard.evaluation.visualization.plt")
@patch("orchard.evaluation.visualization._get_predictions_batch")
def test_show_predictions_basic(mock_get_batch, mock_plt, tmp_path):
    """Test show_predictions creates prediction grid."""
    # Configure mock plt.subplots to return (fig, axes)
    mock_fig = MagicMock()
    mock_axes = [MagicMock() for _ in range(12)]
    mock_plt.subplots.return_value = (mock_fig, np.array(mock_axes))

    # Mock batch results
    images = np.random.rand(12, 3, 28, 28)
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    preds = np.array([0, 1, 1, 0, 2, 2, 0, 1, 2, 0, 1, 2])
    mock_get_batch.return_value = (images, labels, preds)

    mock_model = MagicMock()
    mock_loader = MagicMock()
    device = torch.device("cpu")
    classes = ["class0", "class1", "class2"]
    save_path = tmp_path / "predictions.png"

    mock_cfg = MagicMock()
    mock_cfg.evaluation.n_samples = 12
    mock_cfg.evaluation.grid_cols = 4
    mock_cfg.evaluation.fig_dpi = 200
    mock_cfg.evaluation.fig_size_predictions = (12, 9)
    mock_cfg.model.name = "resnet18"
    mock_cfg.dataset.resolution = 28
    mock_cfg.dataset.mean = [0.5, 0.5, 0.5]
    mock_cfg.dataset.std = [0.5, 0.5, 0.5]
    mock_cfg.dataset.metadata.is_texture_based = False
    mock_cfg.dataset.metadata.is_anatomical = True
    mock_cfg.training.use_tta = False

    show_predictions(mock_model, mock_loader, device, classes, save_path, mock_cfg)

    # Verify model.eval was called
    mock_model.eval.assert_called_once()
    # Verify plot methods
    assert mock_plt.subplots.called
    assert mock_plt.savefig.called


@pytest.mark.unit
@patch("orchard.evaluation.visualization.plt")
@patch("orchard.evaluation.visualization._get_predictions_batch")
def test_show_predictions_without_config(mock_get_batch, mock_plt):
    """Test show_predictions works without config (uses defaults) - lines 68-84."""
    # Configure mock plt.subplots to return (fig, axes)
    mock_fig = MagicMock()
    # Create axes as a list, not trying to reshape with numpy
    mock_axes = np.empty((3, 4), dtype=object)
    for i in range(3):
        for j in range(4):
            mock_axes[i, j] = MagicMock()

    mock_plt.subplots.return_value = (mock_fig, mock_axes)

    images = np.random.rand(12, 3, 28, 28)
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    preds = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    mock_get_batch.return_value = (images, labels, preds)

    mock_model = MagicMock()
    mock_loader = MagicMock()
    device = torch.device("cpu")
    classes = ["class0", "class1"]

    # Need to provide a minimal cfg because _setup_prediction_grid requires it
    # This tests the branches where cfg fields are checked with "if cfg"
    mock_cfg = MagicMock()
    mock_cfg.evaluation.n_samples = 12
    mock_cfg.evaluation.grid_cols = 4
    mock_cfg.evaluation.fig_size_predictions = (12, 9)
    mock_cfg.evaluation.fig_dpi = 150
    mock_cfg.model.name = "model"
    mock_cfg.dataset.resolution = 28
    mock_cfg.dataset.mean = [0.5, 0.5, 0.5]  # Valid mean for RGB
    mock_cfg.dataset.std = [0.5, 0.5, 0.5]  # Valid std for RGB
    mock_cfg.training.use_tta = False

    # Mock dataset.metadata to not exist (AttributeError) to test else branch
    type(mock_cfg.dataset).metadata = property(lambda self: (_ for _ in ()).throw(AttributeError()))

    # Call with cfg but trigger the "if cfg" branches
    show_predictions(mock_model, mock_loader, device, classes, save_path=None, cfg=mock_cfg)

    mock_model.eval.assert_called_once()
    assert mock_plt.subplots.called


@pytest.mark.unit
@patch("orchard.evaluation.visualization.plt")
@patch("orchard.evaluation.visualization._get_predictions_batch")
def test_show_predictions_without_save_path(mock_get_batch, mock_plt, tmp_path):
    """Test show_predictions with save_path=None (interactive mode) - line 214-215."""
    mock_fig = MagicMock()
    mock_axes = [MagicMock() for _ in range(12)]
    mock_plt.subplots.return_value = (mock_fig, np.array(mock_axes))

    images = np.random.rand(12, 3, 28, 28)
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    preds = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    mock_get_batch.return_value = (images, labels, preds)

    mock_model = MagicMock()
    mock_loader = MagicMock()
    device = torch.device("cpu")
    classes = ["class0", "class1"]

    mock_cfg = MagicMock()
    mock_cfg.evaluation.n_samples = 12
    mock_cfg.evaluation.grid_cols = 4
    mock_cfg.evaluation.fig_dpi = 200
    mock_cfg.evaluation.fig_size_predictions = (12, 9)
    mock_cfg.model.name = "resnet18"
    mock_cfg.dataset.resolution = 28
    mock_cfg.dataset.mean = [0.5, 0.5, 0.5]
    mock_cfg.dataset.std = [0.5, 0.5, 0.5]
    mock_cfg.dataset.metadata.is_texture_based = False
    mock_cfg.dataset.metadata.is_anatomical = False
    mock_cfg.training.use_tta = False

    # Call with save_path=None (should call plt.show() instead of savefig)
    show_predictions(mock_model, mock_loader, device, classes, save_path=None, cfg=mock_cfg)

    # Should call show() not savefig()
    mock_plt.show.assert_called_once()


@pytest.mark.unit
@patch("orchard.evaluation.visualization.plt")
@patch("orchard.evaluation.visualization._get_predictions_batch")
def test_show_predictions_standard_mode(mock_get_batch, mock_plt, tmp_path):
    """Test show_predictions with standard mode (neither texture nor anatomical) - line 95."""
    mock_fig = MagicMock()
    mock_axes = [MagicMock() for _ in range(6)]
    mock_plt.subplots.return_value = (mock_fig, np.array(mock_axes))

    images = np.random.rand(6, 1, 28, 28)
    labels = np.array([0, 1, 0, 1, 0, 1])
    preds = np.array([0, 1, 0, 1, 0, 1])
    mock_get_batch.return_value = (images, labels, preds)

    mock_model = MagicMock()
    mock_loader = MagicMock()
    device = torch.device("cpu")
    classes = ["class0", "class1"]
    save_path = tmp_path / "predictions.png"

    mock_cfg = MagicMock()
    mock_cfg.evaluation.n_samples = 6
    mock_cfg.evaluation.grid_cols = 3
    mock_cfg.evaluation.fig_dpi = 150
    mock_cfg.evaluation.fig_size_predictions = (9, 6)
    mock_cfg.model.name = "model"
    mock_cfg.dataset.resolution = 28
    mock_cfg.dataset.mean = [0.5]
    mock_cfg.dataset.std = [0.5]
    mock_cfg.dataset.metadata.is_texture_based = False  # Not texture
    mock_cfg.dataset.metadata.is_anatomical = False  # Not anatomical -> Standard
    mock_cfg.training.use_tta = False

    show_predictions(mock_model, mock_loader, device, classes, save_path, mock_cfg, n=6)

    assert mock_plt.subplots.called
    assert mock_plt.savefig.called


@pytest.mark.unit
@patch("orchard.evaluation.visualization.plt")
@patch("orchard.evaluation.visualization._get_predictions_batch")
def test_show_predictions_with_custom_n(mock_get_batch, mock_plt, tmp_path):
    """Test show_predictions respects custom n parameter."""
    # Configure mock plt.subplots to return (fig, axes)
    mock_fig = MagicMock()
    mock_axes = [MagicMock() for _ in range(6)]
    mock_plt.subplots.return_value = (mock_fig, np.array(mock_axes))

    images = np.random.rand(6, 3, 28, 28)
    labels = np.array([0, 1, 2, 0, 1, 2])
    preds = np.array([0, 1, 1, 0, 2, 2])
    mock_get_batch.return_value = (images, labels, preds)

    mock_model = MagicMock()
    mock_loader = MagicMock()
    device = torch.device("cpu")
    classes = ["class0", "class1", "class2"]
    save_path = tmp_path / "predictions.png"

    mock_cfg = MagicMock()
    mock_cfg.evaluation.grid_cols = 3
    mock_cfg.evaluation.fig_dpi = 150
    mock_cfg.evaluation.fig_size_predictions = (9, 6)
    mock_cfg.model.name = "vit"
    mock_cfg.dataset.resolution = 224
    mock_cfg.dataset.metadata.is_texture_based = True
    mock_cfg.dataset.metadata.is_anatomical = False
    mock_cfg.training.use_tta = True

    show_predictions(mock_model, mock_loader, device, classes, save_path, mock_cfg, n=6)

    # Verify n=6 was passed to _get_predictions_batch
    mock_get_batch.assert_called_once()
    assert mock_get_batch.call_args[0][3] == 6


# HELPER FUNCTIONS - DIRECT TESTS
@pytest.mark.unit
def test_get_predictions_batch_directly():
    """Test _get_predictions_batch function directly (lines 183-191)."""
    from orchard.evaluation.visualization import _get_predictions_batch

    # Create a simple mock model
    mock_model = MagicMock()
    mock_model.return_value = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])

    # Create a mock loader
    images = torch.randn(5, 3, 28, 28)
    labels = torch.tensor([0, 1, 0, 1, 0])
    mock_loader = MagicMock()
    mock_loader.__iter__ = MagicMock(return_value=iter([(images, labels)]))

    device = torch.device("cpu")

    # Call the function
    img_arr, label_arr, pred_arr = _get_predictions_batch(mock_model, mock_loader, device, n=3)

    # Verify outputs
    assert img_arr.shape == (3, 3, 28, 28)  # First 3 images
    assert label_arr.shape == (3,)
    assert pred_arr.shape == (3,)
    assert isinstance(img_arr, np.ndarray)
    assert isinstance(label_arr, np.ndarray)
    assert isinstance(pred_arr, np.ndarray)


@pytest.mark.unit
def test_setup_prediction_grid_directly():
    """Test _setup_prediction_grid function directly."""
    from orchard.evaluation.visualization import _setup_prediction_grid

    mock_cfg = MagicMock()
    mock_cfg.evaluation.fig_size_predictions = (12, 9)

    with patch("orchard.evaluation.visualization.plt.subplots") as mock_subplots:
        mock_fig = MagicMock()
        # Create a 3x4 grid of mock axes using object dtype
        mock_axes = np.empty((3, 4), dtype=object)
        for i in range(3):
            for j in range(4):
                mock_axes[i, j] = MagicMock()

        mock_subplots.return_value = (mock_fig, mock_axes)

        fig, axes = _setup_prediction_grid(12, 4, mock_cfg)

        # Should calculate 3 rows (12 samples / 4 cols)
        mock_subplots.assert_called_once()
        # Axes should be flattened to 1D array
        assert len(axes) == 12


@pytest.mark.unit
def test_finalize_figure_with_save(tmp_path):
    """Test _finalize_figure with save_path."""
    from orchard.evaluation.visualization import _finalize_figure

    mock_plt = MagicMock()
    save_path = tmp_path / "test.png"

    mock_cfg = MagicMock()
    mock_cfg.evaluation.fig_dpi = 200

    _finalize_figure(mock_plt, save_path, mock_cfg)

    # Should call savefig, not show
    mock_plt.savefig.assert_called_once()
    mock_plt.show.assert_not_called()
    mock_plt.close.assert_called_once()


@pytest.mark.unit
def test_finalize_figure_without_save():
    """Test _finalize_figure without save_path (interactive mode) - lines 214-215."""
    from orchard.evaluation.visualization import _finalize_figure

    mock_plt = MagicMock()

    mock_cfg = MagicMock()
    mock_cfg.evaluation.fig_dpi = 150

    _finalize_figure(mock_plt, save_path=None, cfg=mock_cfg)

    # Should call show(), not savefig()
    mock_plt.show.assert_called_once()
    mock_plt.savefig.assert_not_called()
    mock_plt.close.assert_called_once()


#                    HELPER FUNCTIONS - DENORMALIZE & PREPARE                 #
@pytest.mark.unit
def test_denormalize_image():
    """Test _denormalize_image reverses normalization."""
    # Normalized image
    img = np.array([[[0.0, 0.0], [0.0, 0.0]]])  # Shape: (1, 2, 2)

    mock_cfg = MagicMock()
    mock_cfg.dataset.mean = [0.5]
    mock_cfg.dataset.std = [0.5]

    result = _denormalize_image(img, mock_cfg)

    # 0.0 * 0.5 + 0.5 = 0.5
    expected = np.array([[[0.5, 0.5], [0.5, 0.5]]])
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.unit
def test_denormalize_image_rgb():
    """Test _denormalize_image handles RGB images."""
    img = np.zeros((3, 2, 2))  # RGB

    mock_cfg = MagicMock()
    mock_cfg.dataset.mean = [0.485, 0.456, 0.406]
    mock_cfg.dataset.std = [0.229, 0.224, 0.225]

    result = _denormalize_image(img, mock_cfg)

    # Should return mean values (since input is zeros)
    assert result.shape == (3, 2, 2)
    assert 0.0 <= result.min() <= 1.0
    assert 0.0 <= result.max() <= 1.0


@pytest.mark.unit
def test_denormalize_image_clips_values():
    """Test _denormalize_image clips values to [0, 1]."""
    # Create image that would denormalize outside [0, 1]
    img = np.full((1, 2, 2), 10.0)

    mock_cfg = MagicMock()
    mock_cfg.dataset.mean = [0.5]
    mock_cfg.dataset.std = [0.5]

    result = _denormalize_image(img, mock_cfg)

    # Should be clipped to 1.0
    assert result.max() == 1.0


@pytest.mark.unit
def test_prepare_for_plt_chw_to_hwc():
    """Test _prepare_for_plt converts (C, H, W) to (H, W, C)."""
    img = np.random.rand(3, 28, 28)  # CHW

    result = _prepare_for_plt(img)

    assert result.shape == (28, 28, 3)  # HWC


@pytest.mark.unit
def test_prepare_for_plt_grayscale_squeeze():
    """Test _prepare_for_plt squeezes single-channel dimension."""
    img = np.random.rand(1, 28, 28)  # CHW with C=1

    result = _prepare_for_plt(img)

    # After transpose: (28, 28, 1) -> squeeze -> (28, 28)
    assert result.shape == (28, 28)


@pytest.mark.unit
def test_prepare_for_plt_already_2d():
    """Test _prepare_for_plt handles already 2D images."""
    img = np.random.rand(28, 28)  # Already 2D

    result = _prepare_for_plt(img)

    assert result.shape == (28, 28)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
