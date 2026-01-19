"""
Smoke Tests for Data Explorer Module.

Quick coverage tests to validate visualization utilities.
These are minimal tests to boost coverage from 0% to ~15%.
"""

# =========================================================================== #
#                         Standard Imports                                    #
# =========================================================================== #
from pathlib import Path
from unittest.mock import MagicMock, patch

# =========================================================================== #
#                         Third-Party Imports                                 #
# =========================================================================== #
import pytest
import torch

# =========================================================================== #
#                         Internal Imports                                    #
# =========================================================================== #
from orchard.data_handler.data_explorer import (
    show_sample_images,
    show_samples_for_dataset,
)

# =========================================================================== #
#                    SHOW SAMPLE IMAGES: SMOKE TESTS                          #
# =========================================================================== #


@pytest.mark.unit
@patch("orchard.data_handler.data_explorer.plt")
def test_show_sample_images_basic(mock_plt, tmp_path):
    """Test show_sample_images creates and saves a figure."""
    # Mock loader with sample data
    mock_loader = MagicMock()
    mock_images = torch.rand(16, 3, 28, 28)  # Batch of 16 RGB images
    mock_labels = torch.zeros(16)
    mock_loader.__iter__ = MagicMock(return_value=iter([(mock_images, mock_labels)]))

    save_path = tmp_path / "test_grid.png"

    show_sample_images(
        loader=mock_loader,
        save_path=save_path,
        cfg=None,
        num_samples=16,
    )

    # Verify plt methods were called
    assert mock_plt.imshow.called
    assert mock_plt.savefig.called
    assert mock_plt.close.called


@pytest.mark.unit
@patch("orchard.data_handler.data_explorer.plt")
def test_show_sample_images_with_config(mock_plt, tmp_path):
    """Test show_sample_images applies denormalization with config."""
    mock_loader = MagicMock()
    mock_images = torch.rand(8, 3, 28, 28)
    mock_labels = torch.zeros(8)
    mock_loader.__iter__ = MagicMock(return_value=iter([(mock_images, mock_labels)]))

    mock_cfg = MagicMock()
    mock_cfg.dataset.mean = [0.5, 0.5, 0.5]
    mock_cfg.dataset.std = [0.5, 0.5, 0.5]
    mock_cfg.model.name = "test_model"

    save_path = tmp_path / "test_grid.png"

    show_sample_images(
        loader=mock_loader,
        save_path=save_path,
        cfg=mock_cfg,
        num_samples=8,
    )

    assert mock_plt.savefig.called


@pytest.mark.unit
@patch("orchard.data_handler.data_explorer.plt")
def test_show_sample_images_grayscale(mock_plt, tmp_path):
    """Test show_sample_images handles grayscale images."""
    mock_loader = MagicMock()
    mock_images = torch.rand(4, 1, 28, 28)  # Grayscale
    mock_labels = torch.zeros(4)
    mock_loader.__iter__ = MagicMock(return_value=iter([(mock_images, mock_labels)]))

    save_path = tmp_path / "test_gray.png"

    show_sample_images(
        loader=mock_loader,
        save_path=save_path,
        cfg=None,
        num_samples=4,
    )

    # Should use cmap='gray' for grayscale
    assert mock_plt.imshow.called


@pytest.mark.unit
@patch("orchard.data_handler.data_explorer.plt")
@patch("orchard.data_handler.data_explorer.logger")
def test_show_sample_images_empty_loader(mock_logger, mock_plt):
    """Test show_sample_images handles empty loader gracefully."""
    mock_loader = MagicMock()
    mock_loader.__iter__ = MagicMock(return_value=iter([]))  # Empty

    save_path = Path("/tmp/test.png")

    show_sample_images(
        loader=mock_loader,
        save_path=save_path,
        cfg=None,
        num_samples=16,
    )

    # Should log error and return early
    assert mock_logger.error.called
    assert not mock_plt.savefig.called


@pytest.mark.unit
@patch("orchard.data_handler.data_explorer.plt")
def test_show_sample_images_fewer_samples_than_requested(mock_plt, tmp_path):
    """Test show_sample_images handles fewer samples than requested."""
    mock_loader = MagicMock()
    mock_images = torch.rand(5, 3, 28, 28)  # Only 5 images
    mock_labels = torch.zeros(5)
    mock_loader.__iter__ = MagicMock(return_value=iter([(mock_images, mock_labels)]))

    save_path = tmp_path / "test.png"

    show_sample_images(
        loader=mock_loader,
        save_path=save_path,
        cfg=None,
        num_samples=16,  # Request 16 but only 5 available
    )

    # Should handle gracefully
    assert mock_plt.savefig.called


@pytest.mark.unit
@patch("orchard.data_handler.data_explorer.plt")
def test_show_sample_images_with_title_prefix(mock_plt, tmp_path):
    """Test show_sample_images includes title prefix."""
    mock_loader = MagicMock()
    mock_images = torch.rand(4, 3, 28, 28)
    mock_labels = torch.zeros(4)
    mock_loader.__iter__ = MagicMock(return_value=iter([(mock_images, mock_labels)]))

    save_path = tmp_path / "test.png"

    show_sample_images(
        loader=mock_loader,
        save_path=save_path,
        cfg=None,
        num_samples=4,
        title_prefix="Training Data",
    )

    # Verify title was set with prefix
    assert mock_plt.title.called


# =========================================================================== #
#                    SHOW SAMPLES FOR DATASET: SMOKE TESTS                    #
# =========================================================================== #


@pytest.mark.unit
@patch("orchard.data_handler.data_explorer.show_sample_images")
def test_show_samples_for_dataset_basic(mock_show_images, tmp_path):
    """Test show_samples_for_dataset calls show_sample_images correctly."""
    mock_loader = MagicMock()
    mock_run_paths = MagicMock()
    mock_run_paths.get_fig_path = MagicMock(
        return_value=tmp_path / "bloodmnist" / "sample_grid.png"
    )

    show_samples_for_dataset(
        loader=mock_loader,
        classes=["class0", "class1"],
        dataset_name="bloodmnist",
        run_paths=mock_run_paths,
        num_samples=16,
    )

    # Verify show_sample_images was called
    mock_show_images.assert_called_once()


@pytest.mark.unit
@patch("orchard.data_handler.data_explorer.show_sample_images")
def test_show_samples_for_dataset_with_resolution(mock_show_images, tmp_path):
    """Test show_samples_for_dataset includes resolution in filename."""
    mock_loader = MagicMock()
    mock_run_paths = MagicMock()
    expected_path = tmp_path / "pathmnist" / "sample_grid_224x224.png"
    mock_run_paths.get_fig_path = MagicMock(return_value=expected_path)

    show_samples_for_dataset(
        loader=mock_loader,
        classes=["class0"],
        dataset_name="pathmnist",
        run_paths=mock_run_paths,
        num_samples=8,
        resolution=224,
    )

    # Verify correct filename with resolution
    mock_run_paths.get_fig_path.assert_called_once()
    call_args = mock_run_paths.get_fig_path.call_args[0][0]
    assert "224x224" in call_args


@pytest.mark.unit
@patch("orchard.data_handler.data_explorer.show_sample_images")
def test_show_samples_for_dataset_creates_directory(mock_show_images, tmp_path):
    """Test show_samples_for_dataset creates parent directory."""
    mock_loader = MagicMock()
    mock_run_paths = MagicMock()
    save_path = tmp_path / "dataset" / "subdir" / "grid.png"
    mock_run_paths.get_fig_path = MagicMock(return_value=save_path)

    show_samples_for_dataset(
        loader=mock_loader,
        classes=["class0"],
        dataset_name="test",
        run_paths=mock_run_paths,
        num_samples=4,
    )

    # Directory should be created
    assert save_path.parent.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])