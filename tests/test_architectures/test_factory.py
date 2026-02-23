"""
Smoke Tests for Models Factory Module.

Quick coverage tests to validate factory pattern and model instantiation.
These are minimal tests to boost coverage from 0% to ~20%.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from orchard.architectures.factory import _suppress_download_noise, get_model


# FACTORY: BASIC INSTANTIATION
@pytest.mark.unit
def test_get_model_returns_nn_module():
    """Test get_model returns a torch.nn.Module instance."""
    device = torch.device("cpu")
    mock_cfg = MagicMock()
    mock_cfg.architecture.name = "mini_cnn"
    mock_cfg.architecture.dropout = 0.0
    mock_cfg.dataset.effective_in_channels = 3
    mock_cfg.dataset.num_classes = 10
    mock_cfg.dataset.img_size = 28

    model = get_model(device=device, cfg=mock_cfg)

    assert isinstance(model, nn.Module)


@pytest.mark.unit
def test_get_model_deploys_to_device():
    """Test get_model deploys model to specified device."""
    device = torch.device("cpu")
    mock_cfg = MagicMock()
    mock_cfg.architecture.name = "mini_cnn"
    mock_cfg.architecture.dropout = 0.0
    mock_cfg.dataset.effective_in_channels = 3
    mock_cfg.dataset.num_classes = 8
    mock_cfg.dataset.img_size = 28

    model = get_model(device=device, cfg=mock_cfg)

    assert next(model.parameters()).device.type == device.type


@pytest.mark.unit
def test_get_model_invalid_architecture():
    """Test get_model raises ValueError for unknown architecture."""
    device = torch.device("cpu")
    mock_cfg = MagicMock()
    mock_cfg.architecture.name = "invalid_model_xyz"
    mock_cfg.dataset.effective_in_channels = 3
    mock_cfg.dataset.num_classes = 10
    mock_cfg.dataset.img_size = 28

    with pytest.raises(ValueError, match="not registered"):
        get_model(device=device, cfg=mock_cfg)


@pytest.mark.unit
def test_get_model_case_insensitive():
    """Test get_model handles case-insensitive model names."""
    device = torch.device("cpu")
    mock_cfg = MagicMock()
    mock_cfg.architecture.name = "MINI_CNN"
    mock_cfg.architecture.dropout = 0.0
    mock_cfg.dataset.effective_in_channels = 3
    mock_cfg.dataset.num_classes = 10
    mock_cfg.dataset.img_size = 28

    model = get_model(device=device, cfg=mock_cfg)

    assert isinstance(model, nn.Module)


# FACTORY: REGISTRY VALIDATION
@pytest.mark.unit
@pytest.mark.parametrize(
    "model_name",
    ["mini_cnn", "resnet_18", "efficientnet_b0", "convnext_tiny", "vit_tiny"],
)
def test_get_model_all_registered_models(model_name):
    """Test get_model can instantiate all registered models."""
    device = torch.device("cpu")
    mock_cfg = MagicMock()
    mock_cfg.architecture.name = model_name
    mock_cfg.architecture.dropout = 0.0
    mock_cfg.dataset.effective_in_channels = 3
    mock_cfg.dataset.num_classes = 10
    mock_cfg.dataset.img_size = (
        224 if model_name in ("vit_tiny", "efficientnet_b0", "convnext_tiny") else 28
    )
    mock_cfg.dataset.resolution = mock_cfg.dataset.img_size
    mock_cfg.architecture.pretrained = False
    mock_cfg.architecture.weight_variant = None

    model = get_model(device=device, cfg=mock_cfg)

    assert isinstance(model, nn.Module)


@pytest.mark.unit
def test_get_model_verbose_false_suppresses_logs(caplog):
    """Test verbose=False suppresses factory and builder logs."""
    device = torch.device("cpu")
    mock_cfg = MagicMock()
    mock_cfg.architecture.name = "mini_cnn"
    mock_cfg.architecture.dropout = 0.0
    mock_cfg.dataset.effective_in_channels = 3
    mock_cfg.dataset.num_classes = 10
    mock_cfg.dataset.img_size = 28

    with caplog.at_level("INFO"):
        model = get_model(device=device, cfg=mock_cfg, verbose=False)

    assert isinstance(model, nn.Module)
    assert "Initializing Architecture" not in caplog.text
    assert "Model deployed" not in caplog.text


# FACTORY: DOWNLOAD NOISE SUPPRESSION
@pytest.mark.unit
def test_suppress_download_noise_sets_tqdm_disable():
    """Test context manager sets TQDM_DISABLE=1 inside and restores after."""
    os.environ.pop("TQDM_DISABLE", None)

    with _suppress_download_noise():
        assert os.environ.get("TQDM_DISABLE") == "1"

    assert "TQDM_DISABLE" not in os.environ


@pytest.mark.unit
def test_suppress_download_noise_restores_previous_tqdm_value():
    """Test context manager restores previous TQDM_DISABLE value."""
    os.environ["TQDM_DISABLE"] = "0"

    with _suppress_download_noise():
        assert os.environ.get("TQDM_DISABLE") == "1"

    assert os.environ.get("TQDM_DISABLE") == "0"
    os.environ.pop("TQDM_DISABLE", None)


@pytest.mark.unit
def test_suppress_download_noise_restores_on_exception():
    """Test context manager restores env even if body raises."""
    os.environ.pop("TQDM_DISABLE", None)

    with pytest.raises(RuntimeError):
        with _suppress_download_noise():
            raise RuntimeError("boom")

    assert "TQDM_DISABLE" not in os.environ


@pytest.mark.unit
def test_get_model_suppresses_download_bar():
    """Test get_model uses _suppress_download_noise during builder call."""
    device = torch.device("cpu")
    mock_cfg = MagicMock()
    mock_cfg.architecture.name = "mini_cnn"
    mock_cfg.architecture.pretrained = False
    mock_cfg.architecture.dropout = 0.0
    mock_cfg.dataset.effective_in_channels = 3
    mock_cfg.dataset.num_classes = 10
    mock_cfg.dataset.img_size = 28

    with patch("orchard.architectures.factory._suppress_download_noise") as mock_suppress:
        mock_suppress.return_value.__enter__ = MagicMock(return_value=None)
        mock_suppress.return_value.__exit__ = MagicMock(return_value=False)
        model = get_model(device=device, cfg=mock_cfg)

    mock_suppress.assert_called_once()
    assert isinstance(model, nn.Module)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
