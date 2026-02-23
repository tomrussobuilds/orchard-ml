"""
Unit and integration tests for the ConvNeXt-Tiny architecture.
This module verifies the forward pass logic and validates output tensor shapes
for image classification datasets.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from orchard.architectures import build_convnext_tiny


# FIXTURES
@pytest.fixture
def mock_cfg():
    """Provides a standardized configuration mock for model building."""
    cfg = MagicMock()
    cfg.architecture.pretrained = False
    cfg.architecture.dropout = 0.2
    return cfg


@pytest.fixture
def device():
    """Resolves target device for test execution."""
    return torch.device("cpu")


# UNIT TESTS
@pytest.mark.unit
class TestConvNeXtTiny:
    """
    Test suite for the ConvNeXt-Tiny model encompassing forward pass
    verification and integration shape checks.
    """

    @pytest.mark.parametrize(
        "in_channels, num_classes",
        [
            (3, 10),
            (1, 5),
            (3, 100),
        ],
    )
    def test_convnext_tiny_output_shape(self, mock_cfg, device, in_channels, num_classes):
        """Verify output shape matches expected dimensions."""
        model = build_convnext_tiny(
            device, num_classes=num_classes, in_channels=in_channels, cfg=mock_cfg
        )
        model.eval()

        batch_size = 2
        dummy_input = torch.randn(batch_size, in_channels, 224, 224)

        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (batch_size, num_classes)

    def test_convnext_tiny_grayscale_adaptation(self, mock_cfg, device):
        """Verify grayscale input channel adaptation."""
        model = build_convnext_tiny(device, num_classes=10, in_channels=1, cfg=mock_cfg)

        first_conv = model.features[0][0]
        assert first_conv.in_channels == 1
        assert first_conv.out_channels == 96

    def test_convnext_tiny_rgb_standard(self, mock_cfg, device):
        """Verify RGB input channel configuration."""
        model = build_convnext_tiny(device, num_classes=10, in_channels=3, cfg=mock_cfg)

        first_conv = model.features[0][0]
        assert first_conv.in_channels == 3
        assert first_conv.out_channels == 96

    def test_convnext_tiny_pretrained_weight_morphing(self, mock_cfg, device):
        """Verify pretrained weights are loaded and morphed for grayscale."""
        mock_cfg.architecture.pretrained = True

        from orchard.architectures import convnext_tiny as convnext_module

        with patch.object(convnext_module, "models") as mock_models:
            mock_model = MagicMock()
            mock_conv = MagicMock()
            mock_conv.weight = torch.randn(96, 3, 4, 4)
            mock_conv.bias = torch.randn(96)
            mock_model.features = [[mock_conv]]
            mock_model.classifier = [None, None, MagicMock()]
            mock_model.classifier[2].in_features = 768
            mock_model.to = MagicMock(return_value=mock_model)
            mock_models.convnext_tiny.return_value = mock_model
            mock_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 = "mock_weights"

            _ = build_convnext_tiny(device, num_classes=5, in_channels=1, cfg=mock_cfg)

            mock_models.convnext_tiny.assert_called_once_with(weights="mock_weights")

    def test_convnext_tiny_classifier_head_replacement(self, mock_cfg, device):
        """Verify classification head is replaced with correct output size."""
        num_classes = 7
        model = build_convnext_tiny(device, num_classes=num_classes, in_channels=3, cfg=mock_cfg)

        assert model.classifier[2].out_features == num_classes
        assert model.classifier[2].in_features == 768

    def test_convnext_tiny_device_placement(self, mock_cfg):
        """Verify model is placed on correct device."""
        device = torch.device("cpu")
        model = build_convnext_tiny(device, num_classes=10, in_channels=3, cfg=mock_cfg)

        for param in model.parameters():
            assert param.device.type == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
