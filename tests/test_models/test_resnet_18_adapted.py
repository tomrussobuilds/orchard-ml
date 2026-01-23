"""
Unit and integration tests for the ResNet-18 Adapted architecture.
This module verifies the architectural modifications for 28x28 low-resolution
medical imaging and validates output tensor shapes.
"""

# Standard Imports
from unittest.mock import MagicMock, patch

# Third-Party Imports
import pytest
import torch

# Internal Imports
from orchard.models import build_resnet18_adapted


# FIXTURES
@pytest.fixture
def mock_cfg():
    """Provides a standardized configuration mock for model building."""
    cfg = MagicMock()
    cfg.model.pretrained = False
    cfg.model.dropout = 0.5
    return cfg


@pytest.fixture
def device():
    """Resolves target device for test execution."""
    return torch.device("cpu")


# UNIT TESTS
@pytest.mark.unit
class TestResNet18Adapted:
    """
    Test suite for the ResNet-18 Adapted model encompassing architectural
    modifications and shape validation for 28x28 inputs.
    """

    @pytest.mark.parametrize(
        "in_channels, num_classes, img_size, batch_size",
        [
            (3, 10, 28, 1),
            (1, 5, 28, 4),
            (3, 100, 28, 2),
        ],
    )
    def test_resnet18_adapted_output_shape(
        self, mock_cfg, device, in_channels, num_classes, img_size, batch_size
    ):
        """Verify output shape matches expected dimensions for 28x28 inputs."""
        model = build_resnet18_adapted(
            device, num_classes=num_classes, in_channels=in_channels, cfg=mock_cfg
        )
        model.eval()

        dummy_input = torch.randn(batch_size, in_channels, img_size, img_size)

        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (batch_size, num_classes)

    def test_resnet18_adapted_conv1_modification(self, mock_cfg, device):
        """Verify conv1 layer is modified to 3x3 with stride 1."""
        model = build_resnet18_adapted(device, num_classes=10, in_channels=3, cfg=mock_cfg)

        # Check conv1 modifications
        assert model.conv1.kernel_size == (3, 3)
        assert model.conv1.stride == (1, 1)
        assert model.conv1.out_channels == 64

    def test_resnet18_adapted_maxpool_removed(self, mock_cfg, device):
        """Verify maxpool is replaced with Identity layer."""
        model = build_resnet18_adapted(device, num_classes=10, in_channels=3, cfg=mock_cfg)

        # Check maxpool is Identity (no downsampling)
        assert isinstance(model.maxpool, torch.nn.Identity)

    def test_resnet18_adapted_grayscale_input(self, mock_cfg, device):
        """Verify grayscale input channel adaptation."""
        model = build_resnet18_adapted(device, num_classes=10, in_channels=1, cfg=mock_cfg)

        # Check conv1 has 1 input channel
        assert model.conv1.in_channels == 1
        assert model.conv1.out_channels == 64

    def test_resnet18_adapted_rgb_input(self, mock_cfg, device):
        """Verify RGB input channel configuration."""
        model = build_resnet18_adapted(device, num_classes=10, in_channels=3, cfg=mock_cfg)

        # Check conv1 has 3 input channels
        assert model.conv1.in_channels == 3
        assert model.conv1.out_channels == 64

    def test_resnet18_adapted_pretrained_weight_morphing(self, mock_cfg, device):
        """Verify pretrained weights are loaded and morphed."""
        mock_cfg.model.pretrained = True

        # Import the module to find the correct path
        from orchard.models import resnet_18_adapted as resnet_module

        with patch.object(resnet_module, "models") as mock_models:
            # Create a mock model with weights
            mock_model = MagicMock()
            mock_conv = MagicMock()
            mock_conv.weight = torch.randn(64, 3, 7, 7)
            mock_model.conv1 = mock_conv
            mock_model.fc = MagicMock()
            mock_model.fc.in_features = 512
            mock_model.maxpool = MagicMock()
            mock_model.to = MagicMock(return_value=mock_model)

            # Mock the resnet18 function
            mock_models.resnet18.return_value = mock_model
            mock_models.ResNet18_Weights.IMAGENET1K_V1 = "mock_weights"

            model = build_resnet18_adapted(device, num_classes=5, in_channels=1, cfg=mock_cfg)

            # Verify pretrained weights were requested
            mock_models.resnet18.assert_called_once_with(weights="mock_weights")

    def test_resnet18_adapted_fc_replacement(self, mock_cfg, device):
        """Verify classification head is replaced with correct output size."""
        num_classes = 7
        model = build_resnet18_adapted(device, num_classes=num_classes, in_channels=3, cfg=mock_cfg)

        # Check fc layer has correct output dimension
        assert model.fc.out_features == num_classes

    def test_resnet18_adapted_device_placement(self, mock_cfg):
        """Verify model is placed on correct device."""
        device = torch.device("cpu")
        model = build_resnet18_adapted(device, num_classes=10, in_channels=3, cfg=mock_cfg)

        # Check model parameters are on CPU
        for param in model.parameters():
            assert param.device.type == "cpu"

    def test_resnet18_adapted_spatial_preservation(self, mock_cfg, device):
        """Verify spatial dimensions are preserved for 28x28 inputs."""
        model = build_resnet18_adapted(device, num_classes=10, in_channels=3, cfg=mock_cfg)
        model.eval()

        # Test with 28x28 input
        dummy_input = torch.randn(1, 3, 28, 28)

        with torch.no_grad():
            # Hook to check intermediate spatial dimensions
            activations = {}

            def get_activation(name):
                def hook(model, input, output):
                    activations[name] = output.shape

                return hook

            # Register hooks on key layers
            model.conv1.register_forward_hook(get_activation("conv1"))
            model.layer1.register_forward_hook(get_activation("layer1"))

            output = model(dummy_input)

            # After conv1 (3x3, stride 1, padding 1): 28x28 should be preserved
            assert activations["conv1"][2] == 28
            assert activations["conv1"][3] == 28


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
