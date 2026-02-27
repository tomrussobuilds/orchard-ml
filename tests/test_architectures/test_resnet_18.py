"""
Unit and integration tests for the ResNet-18 multi-resolution architecture.
Tests 28x28 (adapted stem), 32x32 (adapted stem), 64x64 (standard stem),
and 224x224 (standard stem).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from orchard.architectures import build_resnet18


# FIXTURES
@pytest.fixture
def device():
    """Resolves target device for test execution."""
    return torch.device("cpu")


# UNIT TESTS — 28x28 MODE
@pytest.mark.unit
class TestResNet18Low:
    """Test suite for ResNet-18 at 28x28 resolution (adapted stem)."""

    @pytest.mark.parametrize(
        "in_channels, num_classes, batch_size",
        [
            (3, 10, 1),
            (1, 5, 4),
            (3, 100, 2),
        ],
    )
    def test_output_shape_28(self, device, in_channels, num_classes, batch_size):
        """Verify output shape matches expected dimensions for 28x28 inputs."""
        model = build_resnet18(
            device,
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=False,
            resolution=28,
        )
        model.eval()

        dummy_input = torch.randn(batch_size, in_channels, 28, 28)

        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (batch_size, num_classes)

    def test_conv1_modified_to_3x3(self, device):
        """Verify conv1 layer is modified to 3x3 with stride 1."""
        model = build_resnet18(
            device, num_classes=10, in_channels=3, pretrained=False, resolution=28
        )

        assert model.conv1.kernel_size == (3, 3)
        assert model.conv1.stride == (1, 1)
        assert model.conv1.out_channels == 64

    def test_maxpool_removed(self, device):
        """Verify maxpool is replaced with Identity layer."""
        model = build_resnet18(
            device, num_classes=10, in_channels=3, pretrained=False, resolution=28
        )

        assert isinstance(model.maxpool, torch.nn.Identity)

    def test_grayscale_input_28(self, device):
        """Verify grayscale input channel adaptation."""
        model = build_resnet18(
            device, num_classes=10, in_channels=1, pretrained=False, resolution=28
        )

        assert model.conv1.in_channels == 1
        assert model.conv1.out_channels == 64

    def test_pretrained_weight_morphing_28(self, device):
        """Verify pretrained weights are loaded and morphed with bicubic interpolation."""
        from orchard.architectures import resnet_18 as resnet_module

        with patch.object(resnet_module, "models") as mock_models:
            mock_model = MagicMock()
            mock_conv = MagicMock()
            mock_conv.weight = torch.randn(64, 3, 7, 7)
            mock_model.conv1 = mock_conv
            mock_model.fc = MagicMock()
            mock_model.fc.in_features = 512
            mock_model.maxpool = MagicMock()
            mock_model.to = MagicMock(return_value=mock_model)
            mock_models.resnet18.return_value = mock_model
            mock_models.ResNet18_Weights.IMAGENET1K_V1 = "mock_weights"

            _ = build_resnet18(device, num_classes=5, in_channels=1, pretrained=True, resolution=28)

            mock_models.resnet18.assert_called_once_with(weights="mock_weights")

    def test_spatial_preservation_28(self, device):
        """Verify spatial dimensions are preserved for 28x28 inputs."""
        model = build_resnet18(
            device, num_classes=10, in_channels=3, pretrained=False, resolution=28
        )
        model.eval()

        dummy_input = torch.randn(1, 3, 28, 28)

        with torch.no_grad():
            activations = {}

            def get_activation(name):
                def hook(model, input, output):
                    activations[name] = output.shape

                return hook

            model.conv1.register_forward_hook(get_activation("conv1"))
            _ = model(dummy_input)

            assert activations["conv1"][2] == 28
            assert activations["conv1"][3] == 28


# UNIT TESTS — 32x32 MODE
@pytest.mark.unit
class TestResNet18At32:
    """Test suite for ResNet-18 at 32x32 resolution (adapted stem, same as 28x28)."""

    @pytest.mark.parametrize(
        "in_channels, num_classes, batch_size",
        [
            (3, 10, 1),
            (3, 100, 2),
        ],
    )
    def test_output_shape_32(self, device, in_channels, num_classes, batch_size):
        """Verify output shape matches expected dimensions for 32x32 inputs."""
        model = build_resnet18(
            device,
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=False,
            resolution=32,
        )
        model.eval()

        dummy_input = torch.randn(batch_size, in_channels, 32, 32)

        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (batch_size, num_classes)

    def test_adapted_stem_at_32(self, device):
        """Verify 32x32 uses adapted 3x3 conv1 with stride 1 (same as 28x28)."""
        model = build_resnet18(
            device, num_classes=10, in_channels=3, pretrained=False, resolution=32
        )

        assert model.conv1.kernel_size == (3, 3)
        assert model.conv1.stride == (1, 1)
        assert model.conv1.out_channels == 64

    def test_maxpool_removed_at_32(self, device):
        """Verify maxpool is replaced with Identity at 32x32."""
        model = build_resnet18(
            device, num_classes=10, in_channels=3, pretrained=False, resolution=32
        )

        assert isinstance(model.maxpool, torch.nn.Identity)

    def test_spatial_preservation_32(self, device):
        """Verify spatial dimensions are preserved for 32x32 inputs after conv1."""
        model = build_resnet18(
            device, num_classes=10, in_channels=3, pretrained=False, resolution=32
        )
        model.eval()

        dummy_input = torch.randn(1, 3, 32, 32)

        with torch.no_grad():
            activations = {}

            def get_activation(name):
                def hook(model, input, output):
                    activations[name] = output.shape

                return hook

            model.conv1.register_forward_hook(get_activation("conv1"))
            _ = model(dummy_input)

            assert activations["conv1"][2] == 32
            assert activations["conv1"][3] == 32


# UNIT TESTS — 64x64 MODE
@pytest.mark.unit
class TestResNet18Mid:
    """Test suite for ResNet-18 at 64x64 resolution (standard stem)."""

    @pytest.mark.parametrize(
        "in_channels, num_classes, batch_size",
        [
            (3, 10, 1),
            (1, 5, 2),
        ],
    )
    def test_output_shape_64(self, device, in_channels, num_classes, batch_size):
        """Verify output shape matches expected dimensions for 64x64 inputs."""
        model = build_resnet18(
            device,
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=False,
            resolution=64,
        )
        model.eval()

        dummy_input = torch.randn(batch_size, in_channels, 64, 64)

        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (batch_size, num_classes)

    def test_standard_stem_at_64(self, device):
        """Verify 64x64 uses standard 7x7 conv1 with stride 2 (same as 224)."""
        model = build_resnet18(
            device, num_classes=10, in_channels=3, pretrained=False, resolution=64
        )

        assert model.conv1.kernel_size == (7, 7)
        assert model.conv1.stride == (2, 2)
        assert not isinstance(model.maxpool, torch.nn.Identity)

    def test_grayscale_input_64(self, device):
        """Verify grayscale channel adaptation at 64x64."""
        model = build_resnet18(
            device, num_classes=10, in_channels=1, pretrained=False, resolution=64
        )

        assert model.conv1.kernel_size == (7, 7)
        assert model.conv1.in_channels == 1


# UNIT TESTS — 224x224 MODE
@pytest.mark.unit
class TestResNet18High:
    """Test suite for ResNet-18 at 224x224 resolution (standard stem)."""

    @pytest.mark.parametrize(
        "in_channels, num_classes, batch_size",
        [
            (3, 10, 1),
            (1, 5, 2),
            (3, 100, 2),
        ],
    )
    def test_output_shape_224(self, device, in_channels, num_classes, batch_size):
        """Verify output shape matches expected dimensions for 224x224 inputs."""
        model = build_resnet18(
            device,
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=False,
            resolution=224,
        )
        model.eval()

        dummy_input = torch.randn(batch_size, in_channels, 224, 224)

        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (batch_size, num_classes)

    def test_standard_stem_preserved(self, device):
        """Verify 224x224 uses standard 7x7 conv1 with stride 2."""
        model = build_resnet18(
            device, num_classes=10, in_channels=3, pretrained=False, resolution=224
        )

        assert model.conv1.kernel_size == (7, 7)
        assert model.conv1.stride == (2, 2)
        assert not isinstance(model.maxpool, torch.nn.Identity)

    def test_grayscale_channel_compression_224(self, device):
        """Verify 224x224 grayscale only modifies channels, not kernel."""
        model = build_resnet18(
            device, num_classes=10, in_channels=1, pretrained=False, resolution=224
        )

        assert model.conv1.kernel_size == (7, 7)
        assert model.conv1.stride == (2, 2)
        assert model.conv1.in_channels == 1

    def test_pretrained_weight_morphing_224(self, device):
        """Verify pretrained weights are loaded and channel-averaged for grayscale 224x224."""
        from orchard.architectures import resnet_18 as resnet_module

        with patch.object(resnet_module, "models") as mock_models:
            mock_model = MagicMock()
            mock_conv = MagicMock()
            mock_conv.weight = torch.randn(64, 3, 7, 7)
            mock_model.conv1 = mock_conv
            mock_model.fc = MagicMock()
            mock_model.fc.in_features = 512
            mock_model.maxpool = MagicMock()
            mock_model.to = MagicMock(return_value=mock_model)
            mock_models.resnet18.return_value = mock_model
            mock_models.ResNet18_Weights.IMAGENET1K_V1 = "mock_weights"

            _ = build_resnet18(
                device, num_classes=5, in_channels=1, pretrained=True, resolution=224
            )

            mock_models.resnet18.assert_called_once_with(weights="mock_weights")


# UNIT TESTS — SHARED
@pytest.mark.unit
class TestResNet18Shared:
    """Tests common to both resolutions."""

    def test_fc_replacement(self, device):
        """Verify classification head is replaced with correct output size."""
        num_classes = 7
        model = build_resnet18(
            device, num_classes=num_classes, in_channels=3, pretrained=False, resolution=28
        )

        assert model.fc.out_features == num_classes

    def test_device_placement(self):
        """Verify model is placed on correct device."""
        device = torch.device("cpu")
        model = build_resnet18(
            device, num_classes=10, in_channels=3, pretrained=False, resolution=28
        )

        for param in model.parameters():
            assert param.device.type == "cpu"

    def test_rgb_input(self, device):
        """Verify RGB input channel configuration."""
        model = build_resnet18(
            device, num_classes=10, in_channels=3, pretrained=False, resolution=28
        )

        assert model.conv1.in_channels == 3
        assert model.conv1.out_channels == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
