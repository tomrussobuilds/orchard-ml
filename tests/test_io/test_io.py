"""
Smoke Tests for Data I/O and Checkpoints Modules.

Tests to validate NPZ validation, checksums, and model loading.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from orchard.core.io.checkpoints import load_model_weights
from orchard.core.io.data_io import md5_checksum, validate_npz_keys


# VALIDATE NPZ KEYS
@pytest.mark.unit
def test_validate_npz_keys_valid_data():
    """Test validate_npz_keys passes with all required keys."""
    mock_npz = MagicMock()
    mock_npz.files = [
        "train_images",
        "train_labels",
        "val_images",
        "val_labels",
        "test_images",
        "test_labels",
    ]

    validate_npz_keys(mock_npz)


@pytest.mark.unit
def test_validate_npz_keys_missing_keys():
    """Test validate_npz_keys raises ValueError for missing keys."""
    mock_npz = MagicMock()
    mock_npz.files = ["train_images", "train_labels"]

    with pytest.raises(ValueError, match="Missing keys"):
        validate_npz_keys(mock_npz)


@pytest.mark.unit
def test_validate_npz_keys_extra_keys_allowed():
    """Test validate_npz_keys allows extra keys beyond required."""
    mock_npz = MagicMock()
    mock_npz.files = [
        "train_images",
        "train_labels",
        "val_images",
        "val_labels",
        "test_images",
        "test_labels",
        "extra_metadata",
    ]

    validate_npz_keys(mock_npz)


@pytest.mark.unit
def test_validate_npz_keys_empty_file():
    """Test validate_npz_keys raises ValueError for empty NPZ."""
    mock_npz = MagicMock()
    mock_npz.files = []

    with pytest.raises(ValueError, match="Missing keys"):
        validate_npz_keys(mock_npz)


# MD5 CHECKSUM
@pytest.mark.unit
def test_md5_checksum_basic(tmp_path):
    """Test md5_checksum calculates correct hash."""
    test_file = tmp_path / "test.txt"
    test_content = b"Hello, World!"
    test_file.write_bytes(test_content)

    expected_hash = "65a8e27d8879283831b664bd8b7f0ad4"

    result = md5_checksum(test_file)

    assert result == expected_hash


@pytest.mark.unit
def test_md5_checksum_empty_file(tmp_path):
    """Test md5_checksum handles empty file."""
    test_file = tmp_path / "empty.txt"
    test_file.write_bytes(b"")

    expected_hash = "d41d8cd98f00b204e9800998ecf8427e"

    result = md5_checksum(test_file)

    assert result == expected_hash


@pytest.mark.unit
def test_md5_checksum_large_file(tmp_path):
    """Test md5_checksum handles file larger than buffer size."""
    test_file = tmp_path / "large.bin"
    large_content = b"X" * 10000
    test_file.write_bytes(large_content)

    result = md5_checksum(test_file)

    assert isinstance(result, str)
    assert len(result) == 32
    assert all(c in "0123456789abcdef" for c in result)


@pytest.mark.unit
def test_md5_checksum_binary_content(tmp_path):
    """Test md5_checksum handles binary content."""
    test_file = tmp_path / "binary.bin"
    binary_content = bytes(range(256))
    test_file.write_bytes(binary_content)

    result = md5_checksum(test_file)

    assert isinstance(result, str)
    assert len(result) == 32


# LOAD MODEL WEIGHTS
@pytest.mark.unit
def test_load_model_weights_file_not_found():
    """Test load_model_weights raises FileNotFoundError for missing checkpoint."""
    mock_model = MagicMock()
    nonexistent_path = Path("/nonexistent/model.pth")
    device = torch.device("cpu")

    with pytest.raises(FileNotFoundError, match="not found"):
        load_model_weights(mock_model, nonexistent_path, device)


@pytest.mark.unit
@patch("torch.load")
def test_load_model_weights_success(mock_torch_load, tmp_path):
    """Test load_model_weights loads checkpoint successfully."""
    model = nn.Linear(10, 5)
    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.touch()

    # Return keys that exactly match the model's state_dict for architecture validation
    mock_state_dict = model.state_dict()
    mock_torch_load.return_value = mock_state_dict

    device = torch.device("cpu")

    load_model_weights(model, checkpoint_path, device)

    mock_torch_load.assert_called_once_with(checkpoint_path, map_location=device, weights_only=True)


@pytest.mark.unit
@patch("torch.load")
def test_load_model_weights_maps_to_device(mock_torch_load, tmp_path):
    """Test load_model_weights uses correct device mapping."""
    model = nn.Linear(10, 5)
    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.touch()

    mock_state_dict = {"weight": torch.randn(5, 10), "bias": torch.randn(5)}
    mock_torch_load.return_value = mock_state_dict

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    load_model_weights(model, checkpoint_path, device)

    call_kwargs = mock_torch_load.call_args.kwargs
    assert call_kwargs["map_location"] == device


@pytest.mark.unit
@patch("torch.load")
def test_load_model_weights_uses_weights_only(mock_torch_load, tmp_path):
    """Test load_model_weights uses weights_only=True for security."""
    model = nn.Linear(10, 5)
    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.touch()
    mock_state_dict = model.state_dict()
    mock_torch_load.return_value = mock_state_dict

    device = torch.device("cpu")

    load_model_weights(model, checkpoint_path, device)

    call_kwargs = mock_torch_load.call_args.kwargs
    assert call_kwargs["weights_only"] is True


@pytest.mark.unit
@patch("torch.load")
def test_load_model_weights_architecture_mismatch(mock_torch_load, tmp_path):
    """Test load_model_weights raises RuntimeError for mismatched checkpoint keys."""
    model = nn.Linear(10, 5)
    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.touch()

    # Return checkpoint with keys that do NOT match the model's state_dict
    mismatched_state_dict = {
        "encoder.weight": torch.randn(5, 10),
        "encoder.bias": torch.randn(5),
    }
    mock_torch_load.return_value = mismatched_state_dict

    device = torch.device("cpu")

    with pytest.raises(RuntimeError, match="architecture mismatch"):
        load_model_weights(model, checkpoint_path, device)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
