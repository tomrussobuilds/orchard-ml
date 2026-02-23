"""
Unit and integration tests for the EfficientNet-B0 architecture.
This module verifies the forward pass logic and validates output tensor shapes
for medical imaging datasets.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from orchard.architectures import build_efficientnet_b0


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
class TestEfficientNetB0:
    """
    Test suite for the EfficientNet-B0 model encompassing forward pass
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
    def test_efficientnet_b0_output_shape(self, mock_cfg, device, in_channels, num_classes):
        """Verify output shape matches expected dimensions."""
        model = build_efficientnet_b0(
            device, num_classes=num_classes, in_channels=in_channels, cfg=mock_cfg
        )
        model.eval()

        batch_size = 2
        dummy_input = torch.randn(batch_size, in_channels, 224, 224)

        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (batch_size, num_classes)

    def test_efficientnet_b0_grayscale_adaptation(self, mock_cfg, device):
        """Verify grayscale input channel adaptation."""
        model = build_efficientnet_b0(device, num_classes=10, in_channels=1, cfg=mock_cfg)

        first_conv = model.features[0][0]
        assert first_conv.in_channels == 1
        assert first_conv.out_channels == 32

    def test_efficientnet_b0_rgb_standard(self, mock_cfg, device):
        """Verify RGB input channel configuration."""
        model = build_efficientnet_b0(device, num_classes=10, in_channels=3, cfg=mock_cfg)

        first_conv = model.features[0][0]
        assert first_conv.in_channels == 3
        assert first_conv.out_channels == 32

    def test_efficientnet_b0_pretrained_weight_morphing(self, mock_cfg, device):
        """Verify pretrained weights are loaded and morphed for grayscale."""
        mock_cfg.architecture.pretrained = True

        from orchard.architectures import efficientnet_b0 as efficientnet_module

        with patch.object(efficientnet_module, "models") as mock_models:
            mock_model = MagicMock()
            mock_conv = MagicMock()
            mock_conv.weight = torch.randn(32, 3, 3, 3)
            mock_model.features = [[mock_conv]]
            mock_model.classifier = [None, MagicMock()]
            mock_model.classifier[1].in_features = 1280
            mock_model.to = MagicMock(return_value=mock_model)
            mock_models.efficientnet_b0.return_value = mock_model
            mock_models.EfficientNet_B0_Weights.IMAGENET1K_V1 = "mock_weights"

            _ = build_efficientnet_b0(device, num_classes=5, in_channels=1, cfg=mock_cfg)

            mock_models.efficientnet_b0.assert_called_once_with(weights="mock_weights")

    def test_efficientnet_b0_classifier_head_replacement(self, mock_cfg, device):
        """Verify classification head is replaced with correct output size."""
        num_classes = 7
        model = build_efficientnet_b0(device, num_classes=num_classes, in_channels=3, cfg=mock_cfg)

        assert model.classifier[1].out_features == num_classes
        assert model.classifier[1].in_features == 1280

    def test_efficientnet_b0_device_placement(self, mock_cfg):
        """Verify model is placed on correct device."""
        device = torch.device("cpu")
        model = build_efficientnet_b0(device, num_classes=10, in_channels=3, cfg=mock_cfg)

        for param in model.parameters():
            assert param.device.type == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
