"""
Smoke Tests for Models Factory Module.

Quick coverage tests to validate factory pattern and model instantiation.
These are minimal tests to boost coverage from 0% to ~20%.
"""

# =========================================================================== #
#                         Standard Imports                                    #
# =========================================================================== #
from unittest.mock import MagicMock

# =========================================================================== #
#                         Third-Party Imports                                 #
# =========================================================================== #
import pytest
import torch
import torch.nn as nn

# =========================================================================== #
#                         Internal Imports                                    #
# =========================================================================== #
from orchard.models.factory import get_model

# =========================================================================== #
#                    FACTORY: BASIC INSTANTIATION                             #
# =========================================================================== #


@pytest.mark.unit
def test_get_model_returns_nn_module():
    """Test get_model returns a torch.nn.Module instance."""
    device = torch.device("cpu")
    mock_cfg = MagicMock()
    mock_cfg.model.name = "mini_cnn"
    mock_cfg.model.dropout = 0.0  # Concrete value
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
    mock_cfg.model.name = "mini_cnn"
    mock_cfg.model.dropout = 0.0
    mock_cfg.dataset.effective_in_channels = 3
    mock_cfg.dataset.num_classes = 8
    mock_cfg.dataset.img_size = 28

    model = get_model(device=device, cfg=mock_cfg)

    # Check model is on correct device
    assert next(model.parameters()).device.type == device.type


@pytest.mark.unit
def test_get_model_invalid_architecture():
    """Test get_model raises ValueError for unknown architecture."""
    device = torch.device("cpu")
    mock_cfg = MagicMock()
    mock_cfg.model.name = "invalid_model_xyz"
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
    mock_cfg.model.name = "MINI_CNN"  # Uppercase
    mock_cfg.model.dropout = 0.0  # Concrete value
    mock_cfg.dataset.effective_in_channels = 3
    mock_cfg.dataset.num_classes = 10
    mock_cfg.dataset.img_size = 28

    model = get_model(device=device, cfg=mock_cfg)

    assert isinstance(model, nn.Module)


# =========================================================================== #
#                    FACTORY: REGISTRY VALIDATION                             #
# =========================================================================== #


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_name",
    ["mini_cnn", "resnet_18_adapted", "efficientnet_b0", "vit_tiny"],
)
def test_get_model_all_registered_models(model_name):
    """Test get_model can instantiate all registered models."""
    device = torch.device("cpu")
    mock_cfg = MagicMock()
    mock_cfg.model.name = model_name
    mock_cfg.model.dropout = 0.0  # Concrete value for mini_cnn
    mock_cfg.dataset.effective_in_channels = 3
    mock_cfg.dataset.num_classes = 10
    mock_cfg.dataset.img_size = 224 if model_name == "vit_tiny" else 28
    mock_cfg.model.pretrained = False
    mock_cfg.model.weight_variant = None

    model = get_model(device=device, cfg=mock_cfg)

    assert isinstance(model, nn.Module)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
