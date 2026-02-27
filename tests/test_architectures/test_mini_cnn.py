"""
Unit and integration tests for the MiniCNN architecture.
This module verifies the forward pass logic using mocking and validates
output tensor shapes for the orchard model suite.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from orchard.architectures import build_mini_cnn


# FIXTURES
@pytest.fixture
def device():
    """Resolves target device for test execution."""
    return torch.device("cpu")


# UNIT TESTS
@pytest.mark.unit
class TestMiniCNN:
    """
    Test suite for the MiniCNN model encompassing forward pass sequence
    verification and integration shape checks.
    """

    @pytest.mark.parametrize(
        "in_channels, num_classes, img_size",
        [
            (3, 10, 28),
            (1, 5, 32),
        ],
    )
    def test_mini_cnn_forward_flow(self, device, in_channels, num_classes, img_size):
        """Verify that the internal layers' call sequence is correct."""
        model = build_mini_cnn(
            device, num_classes=num_classes, in_channels=in_channels, dropout=0.5
        )

        with (
            patch.object(model.conv1, "forward") as mock_c1,
            patch.object(model.conv2, "forward") as mock_c2,
            patch.object(model.fc, "forward") as mock_fc,
        ):

            mock_c1.return_value = torch.randn(1, 32, img_size // 2, img_size // 2)
            mock_c2.return_value = torch.randn(1, 64, img_size // 4, img_size // 4)
            mock_fc.return_value = torch.randn(1, num_classes)

            dummy_input = torch.randn(1, in_channels, img_size, img_size)
            output = model(dummy_input)

            mock_c1.assert_called_once()
            assert output.shape == (1, num_classes)

    @pytest.mark.parametrize(
        "in_channels, num_classes, img_size, batch_size",
        [
            (3, 10, 28, 1),
            (1, 2, 28, 4),
            (3, 100, 32, 2),
        ],
    )
    def test_mini_cnn_shape_integration(
        self, device, in_channels, num_classes, img_size, batch_size
    ):
        """Verify the integrity of the shapes produced by the real model (without patches)."""
        model = build_mini_cnn(
            device, num_classes=num_classes, in_channels=in_channels, dropout=0.5
        )
        model.eval()

        dummy_input = torch.randn(batch_size, in_channels, img_size, img_size)
        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (batch_size, num_classes)
