"""
Provides unit tests for the Tiny ViT model architecture.
This module validates model initialization, forward pass consistency,
and output tensor shapes across various configurations.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from orchard.architectures.vit_tiny import build_vit_tiny


# FIXTURES
@pytest.fixture
def device():
    """Resolves target device for test execution."""
    return torch.device("cpu")


# UNIT TESTS
@pytest.mark.unit
class TestBuildViTTiny:
    """
    Test suite for Vision Transformer Tiny construction and adaptation.

    Coverage:
        - Architecture initialization (RGB/Grayscale)
        - Pretrained weight loading logic
        - Weight morphing for channel adaptation
        - Error handling for invalid variants
    """

    def test_build_vit_tiny_rgb(self, device):
        """Ensures standard RGB ViT-Tiny is built with correct dimensions."""
        num_classes = 5
        in_channels = 3

        model = build_vit_tiny(
            num_classes,
            in_channels,
            pretrained=True,
            weight_variant="vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        )

        assert isinstance(model, nn.Module)
        assert model.patch_embed.proj.in_channels == 3
        assert model.head.out_features == num_classes

        x = torch.randn(1, 3, 224, 224).to(device)
        output = model(x)
        assert output.shape == (1, num_classes)

    def test_build_vit_tiny_grayscale_morphing(self, device):
        """Validates the 1-channel adaptation and weight morphing (averaging)."""
        num_classes = 2
        in_channels = 1

        model = build_vit_tiny(
            num_classes,
            in_channels,
            pretrained=True,
            weight_variant="vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        )

        assert model.patch_embed.proj.in_channels == 1

        x = torch.randn(1, 1, 224, 224).to(device)
        output = model(x)
        assert output.shape == (1, num_classes)

    def test_build_vit_tiny_no_pretrained(self):
        """Tests initialization with random weights when pretrained flag is False."""
        num_classes = 10
        in_channels = 3

        with patch("orchard.architectures.vit_tiny.timm.create_model") as mock_timm:
            mock_model = MagicMock(spec=nn.Module)
            mock_timm.return_value = mock_model

            model = build_vit_tiny(num_classes, in_channels, pretrained=False)

            assert model == mock_model

            mock_timm.assert_called_once_with(
                "vit_tiny_patch16_224", pretrained=False, num_classes=num_classes, in_chans=3
            )

    def test_invalid_weight_variant_raises_error(self, device):
        """Verifies that an invalid timm variant triggers a descriptive ValueError."""
        with pytest.raises(ValueError, match="Invalid ViT weight variant"):
            build_vit_tiny(2, 3, pretrained=True, weight_variant="invalid_vit_model_name")

    def test_weight_copy_consistency(self, device):
        """Confirms that bias is preserved during patch embedding adaptation."""
        model = build_vit_tiny(
            2,
            1,
            pretrained=True,
            weight_variant="vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        )

        assert model.patch_embed.proj.bias is not None
        assert model.patch_embed.proj.weight.shape[1] == 1
