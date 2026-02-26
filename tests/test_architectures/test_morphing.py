"""
Unit tests for morph_conv_weights weight adaptation utility.

Tests cover spatial interpolation, channel reduction, bias transfer,
and gradient context verification.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from orchard.architectures._morphing import morph_conv_weights


@pytest.mark.unit
class TestMorphConvWeights:
    """Test suite for morph_conv_weights."""

    def test_rgb_to_grayscale_no_kernel_resize(self):
        """Channel reduction 3ch→1ch without spatial resize."""
        old_conv = nn.Conv2d(3, 64, kernel_size=7, bias=False)
        new_conv = nn.Conv2d(1, 64, kernel_size=7, bias=False)

        expected = old_conv.weight.mean(dim=1, keepdim=True).clone()
        morph_conv_weights(old_conv, new_conv, in_channels=1)

        assert new_conv.weight.shape == (64, 1, 7, 7)
        assert torch.allclose(new_conv.weight, expected)

    def test_rgb_to_grayscale_with_kernel_resize(self):
        """Channel reduction + spatial resize (7x7 → 3x3)."""
        old_conv = nn.Conv2d(3, 64, kernel_size=7, bias=False)
        new_conv = nn.Conv2d(1, 64, kernel_size=3, bias=False)

        morph_conv_weights(old_conv, new_conv, in_channels=1, target_kernel_size=(3, 3))

        assert new_conv.weight.shape == (64, 1, 3, 3)

    def test_kernel_resize_no_channel_reduction(self):
        """Spatial resize only (3ch stays 3ch)."""
        old_conv = nn.Conv2d(3, 64, kernel_size=7, bias=False)
        new_conv = nn.Conv2d(3, 64, kernel_size=3, bias=False)

        morph_conv_weights(old_conv, new_conv, in_channels=3, target_kernel_size=(3, 3))

        assert new_conv.weight.shape == (64, 3, 3, 3)

    def test_passthrough_copy(self):
        """No channel change, no kernel resize → exact copy."""
        old_conv = nn.Conv2d(3, 64, kernel_size=3, bias=False)
        new_conv = nn.Conv2d(3, 64, kernel_size=3, bias=False)

        morph_conv_weights(old_conv, new_conv, in_channels=3)

        assert torch.equal(new_conv.weight, old_conv.weight)

    def test_bias_transfer_both_have_bias(self):
        """Bias copied when both convolutions have bias."""
        old_conv = nn.Conv2d(3, 64, kernel_size=3, bias=True)
        new_conv = nn.Conv2d(3, 64, kernel_size=3, bias=True)

        morph_conv_weights(old_conv, new_conv, in_channels=3)

        assert torch.equal(new_conv.bias, old_conv.bias)

    def test_no_bias_transfer_old_no_bias(self):
        """No bias copy when old conv has no bias."""
        old_conv = nn.Conv2d(3, 64, kernel_size=3, bias=False)
        new_conv = nn.Conv2d(3, 64, kernel_size=3, bias=True)

        original_bias = new_conv.bias.clone()
        morph_conv_weights(old_conv, new_conv, in_channels=3)

        assert torch.equal(new_conv.bias, original_bias)

    def test_no_bias_transfer_new_no_bias(self):
        """No crash when new conv has no bias slot."""
        old_conv = nn.Conv2d(3, 64, kernel_size=3, bias=True)
        new_conv = nn.Conv2d(3, 64, kernel_size=3, bias=False)

        morph_conv_weights(old_conv, new_conv, in_channels=3)

        assert new_conv.bias is None

    def test_shape_after_full_morph(self):
        """Correct shape after both channel and spatial morph."""
        old_conv = nn.Conv2d(3, 32, kernel_size=5, bias=False)
        new_conv = nn.Conv2d(1, 32, kernel_size=3, bias=False)

        morph_conv_weights(old_conv, new_conv, in_channels=1, target_kernel_size=(3, 3))

        assert new_conv.weight.shape == (32, 1, 3, 3)

    def test_no_gradient_tracking(self):
        """Morphed weights have no grad_fn attached."""
        old_conv = nn.Conv2d(3, 64, kernel_size=3, bias=False)
        new_conv = nn.Conv2d(1, 64, kernel_size=3, bias=False)

        morph_conv_weights(old_conv, new_conv, in_channels=1)

        assert new_conv.weight.grad_fn is None
