"""
Unit tests for Galaxy10 Converter Module.

Tests download, conversion, splitting, and NPZ creation for Galaxy10 DECals dataset.
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import requests

from orchard.data_handler.fetchers.galaxy10_converter import (
    _create_splits,
    convert_galaxy10_to_npz,
    download_galaxy10_h5,
    ensure_galaxy10_npz,
)


# DOWNLOAD TESTS
@pytest.mark.unit
def test_download_galaxy10_h5_file_already_exists(tmp_path):
    """Test download_galaxy10_h5 skips if file exists."""
    target_h5 = tmp_path / "Galaxy10.h5"
    target_h5.touch()

    with patch("orchard.data_handler.fetchers.galaxy10_converter.requests") as mock_requests:
        with patch("orchard.data_handler.fetchers.galaxy10_converter.logger") as mock_logger:
            download_galaxy10_h5("https://example.com/data.h5", target_h5)
            mock_requests.get.assert_not_called()
            mock_logger.info.assert_called_once()


@pytest.mark.unit
def test_download_galaxy10_h5_success(tmp_path):
    """Test successful download of Galaxy10 HDF5."""
    target_h5 = tmp_path / "Galaxy10.h5"
    url = "https://example.com/galaxy10.h5"

    mock_response = Mock()
    mock_response.iter_content.return_value = [b"chunk1", b"chunk2", b"", b"chunk3"]
    mock_response.raise_for_status = Mock()

    with patch("orchard.data_handler.fetchers.galaxy10_converter.requests.get") as mock_get:
        with patch("orchard.data_handler.fetchers.galaxy10_converter.logger") as mock_logger:
            mock_get.return_value.__enter__.return_value = mock_response

            download_galaxy10_h5(url, target_h5, retries=3, timeout=60)

            assert target_h5.exists()
            mock_get.assert_called_once()
            assert mock_logger.info.call_count == 2


@pytest.mark.unit
def test_download_galaxy10_h5_retry_on_error(tmp_path):
    """Test download retries on error."""
    target_h5 = tmp_path / "Galaxy10.h5"
    url = "https://example.com/galaxy10.h5"

    with patch("orchard.data_handler.fetchers.galaxy10_converter.requests.get") as mock_get:
        with patch("orchard.data_handler.fetchers.galaxy10_converter.logger") as mock_logger:
            mock_get.side_effect = [
                Exception("Network error"),
                Exception("Network error"),
            ]

            with pytest.raises(RuntimeError, match="Failed to download Galaxy10 after 2 attempts"):
                download_galaxy10_h5(url, target_h5, retries=2, timeout=60)

            assert mock_get.call_count == 2
            assert mock_logger.warning.call_count == 1


@pytest.mark.unit
def test_download_galaxy10_h5_cleans_tmp_on_failure(tmp_path):
    """Test tmp file is cleaned up on download failure."""
    target_h5 = tmp_path / "Galaxy10.h5"
    tmp_file = target_h5.with_suffix(".tmp")
    url = "https://example.com/galaxy10.h5"

    def iter_with_failure(*_):
        yield b"chunk1"
        raise requests.ConnectionError("Network error during download")

    mock_response = Mock()
    mock_response.iter_content = iter_with_failure
    mock_response.raise_for_status = Mock()

    with patch("orchard.data_handler.fetchers.galaxy10_converter.requests.get") as mock_get:
        mock_get.return_value.__enter__.return_value = mock_response

        with pytest.raises(RuntimeError):
            download_galaxy10_h5(url, target_h5, retries=1, timeout=60)

        assert not tmp_file.exists()


# CONVERSION TESTS
@pytest.mark.unit
def test_convert_galaxy10_to_npz_no_resize(tmp_path):
    """Test conversion without resizing (already 224x224)."""
    h5_path = tmp_path / "Galaxy10.h5"
    output_npz = tmp_path / "galaxy10.npz"

    rng = np.random.default_rng(42)
    mock_images = rng.integers(0, 255, (10, 224, 224, 3), dtype=np.uint8)
    mock_labels = rng.integers(0, 3, 10, dtype=np.int64)

    mock_h5_file = MagicMock()
    mock_h5_file.__enter__.return_value = {
        "images": mock_images,
        "ans": mock_labels,
    }

    with patch(
        "orchard.data_handler.fetchers.galaxy10_converter.h5py.File", return_value=mock_h5_file
    ):
        with patch("orchard.data_handler.fetchers.galaxy10_converter.logger") as mock_logger:
            convert_galaxy10_to_npz(h5_path, output_npz, target_size=224, seed=42)

            assert output_npz.exists()
            assert mock_logger.info.call_count >= 3

            with np.load(output_npz) as data:
                assert "train_images" in data
                assert "train_labels" in data
                assert "val_images" in data
                assert "val_labels" in data
                assert "test_images" in data


@pytest.mark.unit
def test_convert_galaxy10_to_npz_with_resize(tmp_path):
    """Test conversion with image resizing."""
    h5_path = tmp_path / "Galaxy10.h5"
    output_npz = tmp_path / "galaxy10.npz"

    rng = np.random.default_rng(42)
    real_images = rng.integers(0, 255, (10, 16, 16, 3), dtype=np.uint8)
    real_labels = rng.integers(0, 3, 10, dtype=np.int64)

    mock_h5_file = MagicMock()
    mock_h5_file.__enter__.return_value = {
        "images": real_images,
        "ans": real_labels,
    }

    with patch(
        "orchard.data_handler.fetchers.galaxy10_converter.h5py.File", return_value=mock_h5_file
    ):
        with patch("orchard.data_handler.fetchers.galaxy10_converter.logger") as mock_logger:
            convert_galaxy10_to_npz(h5_path, output_npz, target_size=8, seed=42)

            assert output_npz.exists()
            assert mock_logger.info.call_count >= 4

            with np.load(output_npz) as data:
                assert data["train_images"].shape[1:] == (8, 8, 3)


# SPLIT CREATION TESTS
@pytest.mark.unit
def test_create_splits_stratified():
    """Test stratified splits maintain class distribution."""
    rng = np.random.default_rng(42)
    images = rng.integers(0, 255, (100, 28, 28, 3), dtype=np.uint8)
    labels = np.array([i % 5 for i in range(100)], dtype=np.int64).reshape(-1, 1)

    train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels = _create_splits(
        images, labels, seed=42, train_ratio=0.7, val_ratio=0.15
    )

    # Check split sizes
    assert len(train_imgs) == 70
    assert len(val_imgs) == 15
    assert len(test_imgs) == 15

    # Check all classes present in each split
    train_classes = set(train_labels.flatten())
    val_classes = set(val_labels.flatten())
    test_classes = set(test_labels.flatten())

    assert len(train_classes) == 5
    assert len(val_classes) == 5
    assert len(test_classes) == 5


@pytest.mark.unit
def test_create_splits_shapes():
    """Test split shapes are correct."""
    rng = np.random.default_rng(42)
    images = rng.integers(0, 255, (50, 224, 224, 3), dtype=np.uint8)
    labels = np.array([i % 3 for i in range(50)], dtype=np.int64).reshape(-1, 1)

    train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels = _create_splits(
        images, labels, seed=42
    )

    # Check image shapes
    assert train_imgs.shape[1:] == (224, 224, 3)
    assert val_imgs.shape[1:] == (224, 224, 3)
    assert test_imgs.shape[1:] == (224, 224, 3)

    # Check label shapes
    assert train_labels.shape[1] == 1
    assert val_labels.shape[1] == 1
    assert test_labels.shape[1] == 1


@pytest.mark.unit
def test_create_splits_deterministic():
    """Test splits are deterministic with same seed."""
    rng = np.random.default_rng(42)
    images = rng.integers(0, 255, (30, 28, 28, 3), dtype=np.uint8)
    labels = np.array([i % 3 for i in range(30)], dtype=np.int64).reshape(-1, 1)

    split1 = _create_splits(images, labels, seed=42)
    split2 = _create_splits(images, labels, seed=42)

    # Compare train images
    np.testing.assert_array_equal(split1[0], split2[0])
    np.testing.assert_array_equal(split1[1], split2[1])


@pytest.mark.unit
def test_ensure_galaxy10_npz_file_exists_valid_md5(tmp_path):
    """Test ensure_galaxy10_npz returns existing file with valid MD5."""
    target_npz = tmp_path / "galaxy10.npz"

    dummy_data = {
        "train_images": np.zeros((5, 10, 10, 3), dtype=np.uint8),
        "train_labels": np.zeros((5, 1), dtype=np.int64),
    }
    np.savez_compressed(target_npz, **dummy_data)

    mock_metadata = MagicMock()
    mock_metadata.path = target_npz
    mock_metadata.url = "https://example.com/galaxy10.h5"
    mock_metadata.md5_checksum = "abc123"
    mock_metadata.native_resolution = 224

    with patch("orchard.core.md5_checksum", return_value="abc123"):
        with patch("orchard.data_handler.fetchers.galaxy10_converter.logger") as mock_logger:
            result = ensure_galaxy10_npz(mock_metadata)

            assert result == target_npz
            mock_logger.debug.assert_called()


@pytest.mark.unit
def test_ensure_galaxy10_npz_file_exists_placeholder_md5(tmp_path):
    """Test ensure_galaxy10_npz returns existing file with placeholder MD5."""
    target_npz = tmp_path / "galaxy10.npz"

    dummy_data = {
        "train_images": np.zeros((5, 10, 10, 3), dtype=np.uint8),
        "train_labels": np.zeros((5, 1), dtype=np.int64),
    }
    np.savez_compressed(target_npz, **dummy_data)

    mock_metadata = MagicMock()
    mock_metadata.path = target_npz
    mock_metadata.url = "https://example.com/galaxy10.h5"
    mock_metadata.md5_checksum = "placeholder_will_be_calculated_after_conversion"
    mock_metadata.native_resolution = 224

    with patch("orchard.core.md5_checksum", return_value="real_md5"):
        with patch("orchard.data_handler.fetchers.galaxy10_converter.logger") as mock_logger:
            result = ensure_galaxy10_npz(mock_metadata)

            assert result == target_npz
            mock_logger.debug.assert_called()


@pytest.mark.unit
def test_ensure_galaxy10_npz_md5_mismatch(tmp_path):
    """Test ensure_galaxy10_npz regenerates file on MD5 mismatch."""
    target_npz = tmp_path / "galaxy10.npz"

    dummy_data = {
        "train_images": np.zeros((5, 10, 10, 3), dtype=np.uint8),
        "train_labels": np.zeros((5, 1), dtype=np.int64),
    }
    np.savez_compressed(target_npz, **dummy_data)

    mock_metadata = MagicMock()
    mock_metadata.path = target_npz
    mock_metadata.url = "https://example.com/galaxy10.h5"
    mock_metadata.md5_checksum = "expected_md5"
    mock_metadata.native_resolution = 224

    with patch("orchard.core.md5_checksum") as mock_md5:
        mock_md5.side_effect = ["wrong_md5", "new_md5"]

        with patch("orchard.data_handler.fetchers.galaxy10_converter.download_galaxy10_h5"):
            with patch("orchard.data_handler.fetchers.galaxy10_converter.convert_galaxy10_to_npz"):
                with patch(
                    "orchard.data_handler.fetchers.galaxy10_converter.logger"
                ) as mock_logger:
                    result = ensure_galaxy10_npz(mock_metadata)

                    assert mock_md5.call_count == 2
                    mock_logger.warning.assert_called_once()
                    assert mock_logger.info.call_count >= 1
                    assert result == target_npz


@pytest.mark.unit
def test_ensure_galaxy10_npz_download_and_convert(tmp_path):
    """Test ensure_galaxy10_npz downloads and converts when file missing."""
    target_npz = tmp_path / "galaxy10.npz"
    h5_path = tmp_path / "Galaxy10_DECals.h5"

    mock_metadata = MagicMock()
    mock_metadata.path = target_npz
    mock_metadata.url = "https://example.com/galaxy10.h5"
    mock_metadata.md5_checksum = "placeholder_will_be_calculated_after_conversion"
    mock_metadata.native_resolution = 224

    def mock_convert_impl(h5_path, output_npz, target_size=224, seed=42):
        dummy_data = {
            "train_images": np.zeros((5, 10, 10, 3), dtype=np.uint8),
            "train_labels": np.zeros((5, 1), dtype=np.int64),
            "val_images": np.zeros((2, 10, 10, 3), dtype=np.uint8),
            "val_labels": np.zeros((2, 1), dtype=np.int64),
            "test_images": np.zeros((3, 10, 10, 3), dtype=np.uint8),
            "test_labels": np.zeros((3, 1), dtype=np.int64),
        }
        np.savez_compressed(output_npz, **dummy_data)

    with patch(
        "orchard.data_handler.fetchers.galaxy10_converter.download_galaxy10_h5",
        side_effect=lambda url, path: None,
    ) as mock_download:
        with patch(
            "orchard.data_handler.fetchers.galaxy10_converter.convert_galaxy10_to_npz",
            side_effect=mock_convert_impl,
        ):
            with patch("orchard.core.md5_checksum", return_value="new_md5"):
                with patch(
                    "orchard.data_handler.fetchers.galaxy10_converter.logger"
                ) as mock_logger:
                    result = ensure_galaxy10_npz(mock_metadata)

                    mock_download.assert_called_once_with(mock_metadata.url, h5_path)
                    assert result == target_npz
                    assert target_npz.exists()
                    assert mock_logger.info.call_count >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
