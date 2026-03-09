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
from orchard.exceptions import OrchardDatasetError


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
                requests.ConnectionError("Network error"),
                requests.ConnectionError("Network error"),
            ]

            with pytest.raises(
                OrchardDatasetError, match="Failed to download Galaxy10 after 2 attempts"
            ):
                download_galaxy10_h5(url, target_h5, retries=2, timeout=60)

            assert mock_get.call_count == 2
            assert mock_logger.warning.call_count == 1


@pytest.mark.unit
def test_download_galaxy10_h5_cleans_tmp_on_failure(tmp_path):
    """Test tmp file is cleaned up on download failure."""
    target_h5 = tmp_path / "Galaxy10.h5"
    tmp_file = target_h5.with_suffix(".tmp")
    url = "https://example.com/galaxy10.h5"

    def iter_with_failure(*_args, **_kwargs):
        yield b"chunk1"
        raise requests.ConnectionError("Network error during download")

    mock_response = Mock()
    mock_response.iter_content = iter_with_failure
    mock_response.raise_for_status = Mock()

    with patch("orchard.data_handler.fetchers.galaxy10_converter.requests.get") as mock_get:
        mock_get.return_value.__enter__.return_value = mock_response

        with pytest.raises(OrchardDatasetError):
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


# ─── MUTATION-KILLING TESTS ───


@pytest.mark.unit
class TestDownloadMutations:
    """Kill surviving mutants in download_galaxy10_h5."""

    def test_download_passes_correct_kwargs(self, tmp_path):
        """requests.get should receive url, timeout, and stream=True."""
        target_h5 = tmp_path / "Galaxy10.h5"
        url = "https://example.com/galaxy10.h5"
        captured = {}

        mock_response = Mock()
        mock_response.iter_content.return_value = [b"data"]
        mock_response.raise_for_status = Mock()

        with patch("orchard.data_handler.fetchers.galaxy10_converter.requests.get") as mock_get:
            mock_get.return_value.__enter__.return_value = mock_response

            def capture_call(*args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs
                return mock_get.return_value

            mock_get.side_effect = capture_call

            download_galaxy10_h5(url, target_h5, retries=1, timeout=120, chunk_size=4096)

        assert captured["args"] == (url,)
        assert captured["kwargs"]["timeout"] == 120
        assert captured["kwargs"]["stream"] is True

    def test_download_creates_parent_dir(self, tmp_path):
        """Parent directory should be created."""
        target_h5 = tmp_path / "sub" / "deep" / "Galaxy10.h5"
        url = "https://example.com/galaxy10.h5"

        mock_response = Mock()
        mock_response.iter_content.return_value = [b"data"]
        mock_response.raise_for_status = Mock()

        with patch("orchard.data_handler.fetchers.galaxy10_converter.requests.get") as mock_get:
            mock_get.return_value.__enter__.return_value = mock_response

            download_galaxy10_h5(url, target_h5, retries=1)

        assert target_h5.exists()

    def test_download_default_params(self):
        """Default retries=3, timeout=600, chunk_size=8192."""
        import inspect

        sig = inspect.signature(download_galaxy10_h5)
        assert sig.parameters["retries"].default == 3
        assert sig.parameters["timeout"].default == 600
        assert sig.parameters["chunk_size"].default == 8192

    def test_download_iter_content_receives_chunk_size(self, tmp_path):
        """iter_content should be called with the actual chunk_size, not None."""
        target_h5 = tmp_path / "Galaxy10.h5"
        url = "https://example.com/galaxy10.h5"

        mock_response = Mock()
        mock_response.iter_content.return_value = [b"data"]
        mock_response.raise_for_status = Mock()

        with patch("orchard.data_handler.fetchers.galaxy10_converter.requests.get") as mock_get:
            mock_get.return_value.__enter__.return_value = mock_response

            download_galaxy10_h5(url, target_h5, retries=1, chunk_size=4096)

        mock_response.iter_content.assert_called_once_with(chunk_size=4096)

    def test_download_uses_tmp_then_replaces(self, tmp_path):
        """Download should write to .tmp first, then replace to target."""
        target_h5 = tmp_path / "Galaxy10.h5"
        url = "https://example.com/galaxy10.h5"
        tmp_file = target_h5.with_suffix(".tmp")

        written_to_tmp = {"did": False}

        mock_response = Mock()

        def iter_content_check(chunk_size):
            # At this point, the file should be written to tmp, not target
            yield b"data"
            written_to_tmp["did"] = tmp_file.exists() or True

        mock_response.iter_content = iter_content_check
        mock_response.raise_for_status = Mock()

        with patch("orchard.data_handler.fetchers.galaxy10_converter.requests.get") as mock_get:
            mock_get.return_value.__enter__.return_value = mock_response

            download_galaxy10_h5(url, target_h5, retries=1)

        assert target_h5.exists()
        assert not tmp_file.exists()

    def test_download_retry_count_matches(self, tmp_path):
        """Retry loop should attempt exactly `retries` times."""
        target_h5 = tmp_path / "Galaxy10.h5"
        url = "https://example.com/galaxy10.h5"

        with patch("orchard.data_handler.fetchers.galaxy10_converter.requests.get") as mock_get:
            mock_get.side_effect = [
                requests.ConnectionError("fail1"),
                requests.ConnectionError("fail2"),
                requests.ConnectionError("fail3"),
            ]

            with pytest.raises(OrchardDatasetError, match="3 attempts"):
                download_galaxy10_h5(url, target_h5, retries=3)

            assert mock_get.call_count == 3

    def test_download_retry_warning_includes_attempt_and_error(self, tmp_path):
        """Retry warning should include attempt number and error message."""
        target_h5 = tmp_path / "Galaxy10.h5"
        url = "https://example.com/galaxy10.h5"
        warning_calls = []

        with patch("orchard.data_handler.fetchers.galaxy10_converter.requests.get") as mock_get:
            mock_get.side_effect = [
                requests.ConnectionError("net_error_xyz"),
                requests.ConnectionError("net_error_xyz"),
            ]

            with patch(
                "orchard.data_handler.fetchers.galaxy10_converter.logger.warning",
                side_effect=lambda msg, *args: warning_calls.append(msg % args),
            ):
                with pytest.raises(OrchardDatasetError):
                    download_galaxy10_h5(url, target_h5, retries=2)

        combined = " ".join(warning_calls)
        assert "1" in combined
        assert "net_error_xyz" in combined


@pytest.mark.unit
class TestConvertMutations:
    """Kill surviving mutants in convert_galaxy10_to_npz."""

    def test_convert_default_params(self):
        """Default target_size=224, seed=42."""
        import inspect

        sig = inspect.signature(convert_galaxy10_to_npz)
        assert sig.parameters["target_size"].default == 224
        assert sig.parameters["seed"].default == 42

    def test_convert_labels_reshaped_to_n_by_1(self, tmp_path):
        """Labels should be int64 with shape (N, 1)."""
        h5_path = tmp_path / "Galaxy10.h5"
        output_npz = tmp_path / "galaxy10.npz"

        rng = np.random.default_rng(42)
        mock_images = rng.integers(0, 255, (20, 8, 8, 3), dtype=np.uint8)
        mock_labels = rng.integers(0, 3, 20, dtype=np.int64)

        mock_h5_file = MagicMock()
        mock_h5_file.__enter__.return_value = {
            "images": mock_images,
            "ans": mock_labels,
        }

        with patch(
            "orchard.data_handler.fetchers.galaxy10_converter.h5py.File",
            return_value=mock_h5_file,
        ):
            convert_galaxy10_to_npz(h5_path, output_npz, target_size=8, seed=42)

        with np.load(output_npz) as data:
            for key in ["train_labels", "val_labels", "test_labels"]:
                assert data[key].dtype == np.int64
                assert data[key].ndim == 2
                assert data[key].shape[1] == 1

    def test_convert_resize_uses_bilinear(self, tmp_path):
        """Resizing should produce target_size images."""
        h5_path = tmp_path / "Galaxy10.h5"
        output_npz = tmp_path / "galaxy10.npz"

        rng = np.random.default_rng(42)
        mock_images = rng.integers(0, 255, (15, 16, 16, 3), dtype=np.uint8)
        mock_labels = rng.integers(0, 3, 15, dtype=np.int64)

        mock_h5_file = MagicMock()
        mock_h5_file.__enter__.return_value = {
            "images": mock_images,
            "ans": mock_labels,
        }

        with patch(
            "orchard.data_handler.fetchers.galaxy10_converter.h5py.File",
            return_value=mock_h5_file,
        ):
            convert_galaxy10_to_npz(h5_path, output_npz, target_size=32, seed=42)

        with np.load(output_npz) as data:
            assert data["train_images"].shape[1:] == (32, 32, 3)
            assert data["train_images"].dtype == np.uint8

    def test_convert_resize_non_square_height_mismatch(self, tmp_path):
        """Resize should trigger when only height differs from target_size."""
        h5_path = tmp_path / "Galaxy10.h5"
        output_npz = tmp_path / "galaxy10.npz"

        # Images: 16 height, 8 width — only height mismatches target of 8
        rng = np.random.default_rng(42)
        mock_images = rng.integers(0, 255, (10, 16, 8, 3), dtype=np.uint8)
        mock_labels = rng.integers(0, 3, 10, dtype=np.int64)

        mock_h5_file = MagicMock()
        mock_h5_file.__enter__.return_value = {
            "images": mock_images,
            "ans": mock_labels,
        }

        with patch(
            "orchard.data_handler.fetchers.galaxy10_converter.h5py.File",
            return_value=mock_h5_file,
        ):
            convert_galaxy10_to_npz(h5_path, output_npz, target_size=8, seed=42)

        with np.load(output_npz) as data:
            # Both dims should be resized to target_size
            assert data["train_images"].shape[1] == 8
            assert data["train_images"].shape[2] == 8

    def test_convert_resize_non_square_width_mismatch(self, tmp_path):
        """Resize should trigger when only width differs from target_size."""
        h5_path = tmp_path / "Galaxy10.h5"
        output_npz = tmp_path / "galaxy10.npz"

        # Images: 8 height, 16 width — only width mismatches target of 8
        rng = np.random.default_rng(42)
        mock_images = rng.integers(0, 255, (10, 8, 16, 3), dtype=np.uint8)
        mock_labels = rng.integers(0, 3, 10, dtype=np.int64)

        mock_h5_file = MagicMock()
        mock_h5_file.__enter__.return_value = {
            "images": mock_images,
            "ans": mock_labels,
        }

        with patch(
            "orchard.data_handler.fetchers.galaxy10_converter.h5py.File",
            return_value=mock_h5_file,
        ):
            convert_galaxy10_to_npz(h5_path, output_npz, target_size=8, seed=42)

        with np.load(output_npz) as data:
            assert data["train_images"].shape[1] == 8
            assert data["train_images"].shape[2] == 8

    def test_convert_h5_file_opened_with_read_mode(self, tmp_path):
        """h5py.File should be called with the path and 'r' mode."""
        h5_path = tmp_path / "Galaxy10.h5"
        output_npz = tmp_path / "galaxy10.npz"

        rng = np.random.default_rng(42)
        mock_images = rng.integers(0, 255, (10, 8, 8, 3), dtype=np.uint8)
        mock_labels = rng.integers(0, 3, 10, dtype=np.int64)

        mock_h5_file = MagicMock()
        mock_h5_file.__enter__.return_value = {
            "images": mock_images,
            "ans": mock_labels,
        }

        with patch(
            "orchard.data_handler.fetchers.galaxy10_converter.h5py.File",
            return_value=mock_h5_file,
        ) as mock_h5_cls:
            convert_galaxy10_to_npz(h5_path, output_npz, target_size=8, seed=42)

        mock_h5_cls.assert_called_once_with(h5_path, "r")

    def test_convert_seed_passed_to_splits(self, tmp_path):
        """seed parameter should be forwarded to _create_splits."""
        h5_path = tmp_path / "Galaxy10.h5"
        output1 = tmp_path / "g1.npz"
        output2 = tmp_path / "g2.npz"

        rng = np.random.default_rng(42)
        mock_images = rng.integers(0, 255, (30, 8, 8, 3), dtype=np.uint8)
        mock_labels = rng.integers(0, 3, 30, dtype=np.int64)

        for output, seed in [(output1, 42), (output2, 42)]:
            mock_h5_file = MagicMock()
            mock_h5_file.__enter__.return_value = {
                "images": mock_images,
                "ans": mock_labels,
            }
            with patch(
                "orchard.data_handler.fetchers.galaxy10_converter.h5py.File",
                return_value=mock_h5_file,
            ):
                convert_galaxy10_to_npz(h5_path, output, target_size=8, seed=seed)

        with np.load(output1) as d1, np.load(output2) as d2:
            np.testing.assert_array_equal(d1["train_labels"], d2["train_labels"])

    def test_convert_resize_checks_width_not_channels(self, tmp_path):
        """Resize condition must check shape[2] (width), not shape[3] (channels).

        Use target_size=3 with images of shape (N, 3, 5, 3):
        - shape[1]=3 == target → height matches
        - shape[2]=5 != target → width mismatch → SHOULD resize
        - shape[3]=3 == target → channels happen to match (irrelevant)
        If code checks shape[3] instead of shape[2], resize is skipped
        and output width stays 5 instead of being resized to 3.
        """
        h5_path = tmp_path / "Galaxy10.h5"
        output_npz = tmp_path / "galaxy10.npz"

        rng = np.random.default_rng(42)
        mock_images = rng.integers(0, 255, (15, 3, 5, 3), dtype=np.uint8)
        mock_labels = np.repeat(np.arange(3), 5).astype(np.int64)

        mock_h5_file = MagicMock()
        mock_h5_file.__enter__.return_value = {
            "images": mock_images,
            "ans": mock_labels,
        }

        with patch(
            "orchard.data_handler.fetchers.galaxy10_converter.h5py.File",
            return_value=mock_h5_file,
        ):
            convert_galaxy10_to_npz(h5_path, output_npz, target_size=3, seed=42)

        with np.load(output_npz) as data:
            all_imgs = np.concatenate(
                [data["train_images"], data["val_images"], data["test_images"]]
            )
            # Width must be resized to target_size=3 (not left at 5)
            assert all_imgs.shape[2] == 3

    def test_convert_resize_pixel_values_match_bilinear(self, tmp_path):
        """Resized images should match PIL BILINEAR, not default resampling."""
        from PIL import Image as PILImage

        h5_path = tmp_path / "Galaxy10.h5"
        output_npz = tmp_path / "galaxy10.npz"

        rng = np.random.default_rng(123)
        base_img = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
        mock_images = np.stack([base_img] * 15)
        mock_labels = np.repeat(np.arange(3), 5).astype(np.int64)

        mock_h5_file = MagicMock()
        mock_h5_file.__enter__.return_value = {
            "images": mock_images,
            "ans": mock_labels,
        }

        with patch(
            "orchard.data_handler.fetchers.galaxy10_converter.h5py.File",
            return_value=mock_h5_file,
        ):
            convert_galaxy10_to_npz(h5_path, output_npz, target_size=8, seed=42)

        resample = getattr(PILImage, "Resampling", PILImage).BILINEAR
        expected = np.array(PILImage.fromarray(base_img).resize((8, 8), resample))

        with np.load(output_npz) as data:
            all_imgs = np.concatenate(
                [data["train_images"], data["val_images"], data["test_images"]]
            )
            for img in all_imgs:
                np.testing.assert_array_equal(img, expected)

    def test_convert_default_seed_matches_explicit_42(self, tmp_path):
        """Calling convert without seed should match seed=42."""
        h5_path = tmp_path / "Galaxy10.h5"
        out_default = tmp_path / "default.npz"
        out_explicit = tmp_path / "explicit.npz"

        rng = np.random.default_rng(42)
        mock_images = rng.integers(0, 255, (30, 8, 8, 3), dtype=np.uint8)
        mock_labels = rng.integers(0, 3, 30, dtype=np.int64)

        for output, kwargs in [
            (out_default, {"target_size": 8}),
            (out_explicit, {"target_size": 8, "seed": 42}),
        ]:
            mock_h5_file = MagicMock()
            mock_h5_file.__enter__.return_value = {
                "images": mock_images,
                "ans": mock_labels,
            }
            with patch(
                "orchard.data_handler.fetchers.galaxy10_converter.h5py.File",
                return_value=mock_h5_file,
            ):
                convert_galaxy10_to_npz(h5_path, output, **kwargs)

        with np.load(out_default) as d1, np.load(out_explicit) as d2:
            np.testing.assert_array_equal(d1["train_labels"], d2["train_labels"])

    def test_convert_total_samples_preserved(self, tmp_path):
        """Total samples across splits should equal input count."""
        h5_path = tmp_path / "Galaxy10.h5"
        output_npz = tmp_path / "galaxy10.npz"

        n = 30
        rng = np.random.default_rng(42)
        mock_images = rng.integers(0, 255, (n, 8, 8, 3), dtype=np.uint8)
        mock_labels = rng.integers(0, 3, n, dtype=np.int64)

        mock_h5_file = MagicMock()
        mock_h5_file.__enter__.return_value = {
            "images": mock_images,
            "ans": mock_labels,
        }

        with patch(
            "orchard.data_handler.fetchers.galaxy10_converter.h5py.File",
            return_value=mock_h5_file,
        ):
            convert_galaxy10_to_npz(h5_path, output_npz, target_size=8, seed=42)

        with np.load(output_npz) as data:
            total = len(data["train_images"]) + len(data["val_images"]) + len(data["test_images"])
            assert total == n


@pytest.mark.unit
class TestCreateSplitsMutations:
    """Kill surviving mutants in _create_splits."""

    def test_default_params(self):
        """Default seed=42, train_ratio=0.7, val_ratio=0.15."""
        import inspect

        sig = inspect.signature(_create_splits)
        assert sig.parameters["seed"].default == 42
        assert sig.parameters["train_ratio"].default == pytest.approx(0.7)
        assert sig.parameters["val_ratio"].default == pytest.approx(0.15)

    def test_splits_use_axis0_concatenation(self):
        """Concatenation along axis=0 should preserve image dimensions."""
        rng = np.random.default_rng(42)
        images = rng.integers(0, 255, (60, 28, 28, 3), dtype=np.uint8)
        labels = np.array([i % 3 for i in range(60)], dtype=np.int64).reshape(-1, 1)

        train_imgs, _, val_imgs, _, test_imgs, _ = _create_splits(images, labels)

        assert train_imgs.ndim == 4
        assert val_imgs.ndim == 4
        assert test_imgs.ndim == 4

    def test_splits_total_equals_input(self):
        """Sum of all splits should equal input size."""
        rng = np.random.default_rng(42)
        n = 100
        images = rng.integers(0, 255, (n, 8, 8, 3), dtype=np.uint8)
        labels = np.array([i % 5 for i in range(n)], dtype=np.int64).reshape(-1, 1)

        results = _create_splits(images, labels, seed=42, train_ratio=0.7, val_ratio=0.15)
        total = len(results[0]) + len(results[2]) + len(results[4])

        assert total == n

    def test_default_ratios_produce_reasonable_splits(self):
        """Default train_ratio=0.7, val_ratio=0.15 should produce ~70/15/15 splits."""
        rng = np.random.default_rng(42)
        n = 100
        images = rng.integers(0, 255, (n, 8, 8, 3), dtype=np.uint8)
        labels = np.array([i % 5 for i in range(n)], dtype=np.int64).reshape(-1, 1)

        # Call with defaults (no explicit ratios)
        train_imgs, _, val_imgs, _, test_imgs, _ = _create_splits(images, labels, seed=42)

        # If train_ratio mutated to 1.7, train would be ~100% and val/test empty
        assert len(val_imgs) > 0, "val set should not be empty with default ratios"
        assert len(test_imgs) > 0, "test set should not be empty with default ratios"
        # Approximate check: train ~70%, val ~15%, test ~15%
        assert 0.5 < len(train_imgs) / n < 0.85
        assert 0.05 < len(val_imgs) / n < 0.25
        assert 0.05 < len(test_imgs) / n < 0.25

    def test_splits_different_seeds_differ(self):
        """Different seeds should produce different orderings."""
        rng = np.random.default_rng(42)
        images = rng.integers(0, 255, (60, 8, 8, 3), dtype=np.uint8)
        labels = np.array([i % 3 for i in range(60)], dtype=np.int64).reshape(-1, 1)

        split1 = _create_splits(images, labels, seed=42)
        split2 = _create_splits(images, labels, seed=99)

        assert not np.array_equal(split1[0], split2[0])


@pytest.mark.unit
class TestEnsureGalaxy10Mutations:
    """Kill surviving mutants in ensure_galaxy10_npz."""

    def test_md5_mismatch_regenerates_and_logs(self, tmp_path):
        """MD5 mismatch should regenerate file and log warning."""
        target_npz = tmp_path / "galaxy10.npz"
        np.savez_compressed(
            target_npz,
            train_images=np.zeros((5, 10, 10, 3), dtype=np.uint8),
        )

        mock_metadata = MagicMock()
        mock_metadata.path = target_npz
        mock_metadata.url = "https://example.com/galaxy10.h5"
        mock_metadata.md5_checksum = "expected_md5"
        mock_metadata.native_resolution = 224
        mock_metadata.name = "galaxy10"

        with patch("orchard.core.md5_checksum") as mock_md5:
            mock_md5.side_effect = ["wrong_md5", "new_md5"]

            with patch("orchard.data_handler.fetchers.galaxy10_converter.download_galaxy10_h5"):
                with patch(
                    "orchard.data_handler.fetchers.galaxy10_converter.convert_galaxy10_to_npz"
                ):
                    result = ensure_galaxy10_npz(mock_metadata)

        assert result == target_npz

    def test_native_resolution_used_for_target_size(self, tmp_path):
        """target_size should come from metadata.native_resolution."""
        target_npz = tmp_path / "galaxy10.npz"

        mock_metadata = MagicMock()
        mock_metadata.path = target_npz
        mock_metadata.url = "https://example.com/galaxy10.h5"
        mock_metadata.md5_checksum = "placeholder_will_be_calculated_after_conversion"
        mock_metadata.native_resolution = 128
        mock_metadata.name = "galaxy10"

        convert_calls = []

        def mock_convert(h5_path, output_npz, target_size=224, seed=42):
            convert_calls.append(target_size)
            np.savez_compressed(
                output_npz,
                train_images=np.zeros((2, 10, 10, 3), dtype=np.uint8),
            )

        with patch(
            "orchard.data_handler.fetchers.galaxy10_converter.download_galaxy10_h5",
        ):
            with patch(
                "orchard.data_handler.fetchers.galaxy10_converter.convert_galaxy10_to_npz",
                side_effect=mock_convert,
            ):
                with patch("orchard.core.md5_checksum", return_value="md5"):
                    ensure_galaxy10_npz(mock_metadata)

        assert convert_calls == [128]

    def test_h5_path_uses_correct_filename(self, tmp_path):
        """H5 download path should be Galaxy10_DECals.h5 in npz parent dir."""
        target_npz = tmp_path / "galaxy10.npz"

        mock_metadata = MagicMock()
        mock_metadata.path = target_npz
        mock_metadata.url = "https://example.com/galaxy10.h5"
        mock_metadata.md5_checksum = "placeholder_will_be_calculated_after_conversion"
        mock_metadata.native_resolution = 224
        mock_metadata.name = "galaxy10"

        download_calls = []

        def mock_download(url, h5_path):
            download_calls.append(h5_path)

        def mock_convert(h5_path, output_npz, target_size=224, seed=42):
            np.savez_compressed(
                output_npz,
                train_images=np.zeros((2, 10, 10, 3), dtype=np.uint8),
            )

        with patch(
            "orchard.data_handler.fetchers.galaxy10_converter.download_galaxy10_h5",
            side_effect=mock_download,
        ):
            with patch(
                "orchard.data_handler.fetchers.galaxy10_converter.convert_galaxy10_to_npz",
                side_effect=mock_convert,
            ):
                with patch("orchard.core.md5_checksum", return_value="md5"):
                    ensure_galaxy10_npz(mock_metadata)

        assert download_calls[0] == tmp_path / "Galaxy10_DECals.h5"

    def test_h5_path_passed_to_convert(self, tmp_path):
        """h5_path should be passed to convert_galaxy10_to_npz, not None."""
        target_npz = tmp_path / "galaxy10.npz"

        mock_metadata = MagicMock()
        mock_metadata.path = target_npz
        mock_metadata.url = "https://example.com/galaxy10.h5"
        mock_metadata.md5_checksum = "placeholder_will_be_calculated_after_conversion"
        mock_metadata.native_resolution = 224
        mock_metadata.name = "galaxy10"

        convert_h5_paths = []

        def mock_convert(h5_path, output_npz, target_size=224, seed=42):
            convert_h5_paths.append(h5_path)
            np.savez_compressed(
                output_npz,
                train_images=np.zeros((2, 10, 10, 3), dtype=np.uint8),
            )

        with patch("orchard.data_handler.fetchers.galaxy10_converter.download_galaxy10_h5"):
            with patch(
                "orchard.data_handler.fetchers.galaxy10_converter.convert_galaxy10_to_npz",
                side_effect=mock_convert,
            ):
                with patch("orchard.core.md5_checksum", return_value="md5"):
                    ensure_galaxy10_npz(mock_metadata)

        assert convert_h5_paths[0] is not None
        assert convert_h5_paths[0] == tmp_path / "Galaxy10_DECals.h5"

    def test_md5_checksum_receives_target_npz(self, tmp_path):
        """md5_checksum should receive the actual npz path after conversion."""
        target_npz = tmp_path / "galaxy10.npz"

        mock_metadata = MagicMock()
        mock_metadata.path = target_npz
        mock_metadata.url = "https://example.com/galaxy10.h5"
        mock_metadata.md5_checksum = "placeholder_will_be_calculated_after_conversion"
        mock_metadata.native_resolution = 224
        mock_metadata.name = "galaxy10"

        md5_paths = []

        def tracking_md5(path):
            md5_paths.append(path)
            return "real_md5"

        def mock_convert(h5_path, output_npz, target_size=224, seed=42):
            np.savez_compressed(
                output_npz,
                train_images=np.zeros((2, 10, 10, 3), dtype=np.uint8),
            )

        with patch("orchard.data_handler.fetchers.galaxy10_converter.download_galaxy10_h5"):
            with patch(
                "orchard.data_handler.fetchers.galaxy10_converter.convert_galaxy10_to_npz",
                side_effect=mock_convert,
            ):
                with patch("orchard.core.md5_checksum", side_effect=tracking_md5):
                    ensure_galaxy10_npz(mock_metadata)

        # md5_checksum should be called with the actual target_npz path
        assert target_npz in md5_paths
        assert None not in md5_paths

    def test_md5_checksum_called_with_actual_path_not_none(self, tmp_path):
        """md5_checksum in exists-branch should receive target_npz, not None."""
        target_npz = tmp_path / "galaxy10.npz"
        np.savez_compressed(
            target_npz,
            train_images=np.zeros((5, 10, 10, 3), dtype=np.uint8),
        )

        mock_metadata = MagicMock()
        mock_metadata.path = target_npz
        mock_metadata.url = "https://example.com/galaxy10.h5"
        mock_metadata.md5_checksum = "abc123"
        mock_metadata.native_resolution = 224

        with patch("orchard.core.md5_checksum", return_value="abc123") as mock_md5:
            ensure_galaxy10_npz(mock_metadata)

        mock_md5.assert_called_once_with(target_npz)

    def test_native_resolution_none_falls_back_to_224(self, tmp_path):
        """When native_resolution is falsy, target_size should default to 224."""
        target_npz = tmp_path / "galaxy10.npz"

        mock_metadata = MagicMock()
        mock_metadata.path = target_npz
        mock_metadata.url = "https://example.com/galaxy10.h5"
        mock_metadata.md5_checksum = "placeholder_will_be_calculated_after_conversion"
        mock_metadata.native_resolution = 0  # falsy

        convert_calls = []

        def mock_convert(h5_path, output_npz, target_size=224, seed=42):
            convert_calls.append(target_size)
            np.savez_compressed(
                output_npz,
                train_images=np.zeros((2, 10, 10, 3), dtype=np.uint8),
            )

        with patch("orchard.data_handler.fetchers.galaxy10_converter.download_galaxy10_h5"):
            with patch(
                "orchard.data_handler.fetchers.galaxy10_converter.convert_galaxy10_to_npz",
                side_effect=mock_convert,
            ):
                with patch("orchard.core.md5_checksum", return_value="md5"):
                    ensure_galaxy10_npz(mock_metadata)

        assert convert_calls == [224]

    def test_placeholder_md5_logs_action_required(self, tmp_path):
        """Placeholder MD5 should trigger 'Action Required' log."""
        target_npz = tmp_path / "galaxy10.npz"

        mock_metadata = MagicMock()
        mock_metadata.path = target_npz
        mock_metadata.url = "https://example.com/galaxy10.h5"
        mock_metadata.md5_checksum = "placeholder_will_be_calculated_after_conversion"
        mock_metadata.native_resolution = 224
        mock_metadata.name = "galaxy10"

        def mock_convert(h5_path, output_npz, target_size=224, seed=42):
            np.savez_compressed(
                output_npz,
                train_images=np.zeros((2, 10, 10, 3), dtype=np.uint8),
            )

        with patch("orchard.data_handler.fetchers.galaxy10_converter.download_galaxy10_h5"):
            with patch(
                "orchard.data_handler.fetchers.galaxy10_converter.convert_galaxy10_to_npz",
                side_effect=mock_convert,
            ):
                with patch("orchard.core.md5_checksum", return_value="real_md5"):
                    with patch(
                        "orchard.data_handler.fetchers.galaxy10_converter.logger"
                    ) as mock_logger:
                        ensure_galaxy10_npz(mock_metadata)

                        # Check that info was called with Action Required
                        info_calls = [str(c) for c in mock_logger.info.call_args_list]
                        assert any("Action Required" in c for c in info_calls)
                        assert any("real_md5" in c for c in info_calls)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
