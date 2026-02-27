"""
Tests for the generic timm backbone builder.

Validates model creation, forward pass, channel adaptation,
and error handling for arbitrary timm models.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from orchard.architectures.timm_backbone import build_timm_model
from orchard.core.config import ArchitectureConfig


# FIXTURES
@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def arch_cfg():
    """ArchitectureConfig for a lightweight timm model (no pretrained download)."""
    return ArchitectureConfig(name="timm/resnet10t", pretrained=False, dropout=0.1)


# UNIT TESTS
@pytest.mark.unit
class TestBuildTimmModel:
    """Test suite for generic timm backbone construction."""

    def test_build_returns_nn_module(self, device, arch_cfg):
        model = build_timm_model(device, num_classes=10, in_channels=3, arch_cfg=arch_cfg)
        assert isinstance(model, nn.Module)

    def test_forward_pass_shape(self, device, arch_cfg):
        num_classes = 8
        model = build_timm_model(device, num_classes=num_classes, in_channels=3, arch_cfg=arch_cfg)
        x = torch.randn(2, 3, 224, 224, device=device)
        output = model(x)
        assert output.shape == (2, num_classes)

    def test_single_channel_input(self, device, arch_cfg):
        """timm handles in_chans=1 with automatic weight morphing."""
        model = build_timm_model(device, num_classes=5, in_channels=1, arch_cfg=arch_cfg)
        x = torch.randn(1, 1, 224, 224, device=device)
        output = model(x)
        assert output.shape == (1, 5)

    def test_num_classes_respected(self, device, arch_cfg):
        for n_cls in [2, 10, 100]:
            model = build_timm_model(device, num_classes=n_cls, in_channels=3, arch_cfg=arch_cfg)
            x = torch.randn(1, 3, 224, 224, device=device)
            assert model(x).shape == (1, n_cls)

    def test_deploys_to_device(self, device, arch_cfg):
        model = build_timm_model(device, num_classes=10, in_channels=3, arch_cfg=arch_cfg)
        assert next(model.parameters()).device.type == device.type

    def test_invalid_model_raises_valueerror(self, device):
        arch_cfg = ArchitectureConfig(
            name="timm/absolutely_fake_model_xyz", pretrained=False, dropout=0.0
        )

        with pytest.raises(ValueError, match="Failed to create timm model"):
            build_timm_model(device, num_classes=10, in_channels=3, arch_cfg=arch_cfg)

    def test_model_id_extraction(self, device):
        """Verify the timm/ prefix is correctly stripped."""
        arch_cfg = ArchitectureConfig(
            name="timm/mobilenetv3_small_050", pretrained=False, dropout=0.0
        )

        model = build_timm_model(device, num_classes=10, in_channels=3, arch_cfg=arch_cfg)
        assert isinstance(model, nn.Module)


@pytest.mark.unit
class TestTimmFactoryRouting:
    """Test that the factory correctly routes timm/ names."""

    def test_factory_routes_timm_model(self, device):
        from orchard.architectures.factory import get_model

        cfg = MagicMock()
        cfg.architecture.name = "timm/resnet10t"
        cfg.architecture.pretrained = False
        cfg.architecture.dropout = 0.0
        cfg.dataset.effective_in_channels = 3
        cfg.dataset.num_classes = 10
        cfg.dataset.img_size = 224

        model = get_model(device=device, dataset_cfg=cfg.dataset, arch_cfg=cfg.architecture)
        assert isinstance(model, nn.Module)

    def test_factory_still_routes_builtin(self, device):
        from orchard.architectures.factory import get_model

        cfg = MagicMock()
        cfg.architecture.name = "mini_cnn"
        cfg.architecture.dropout = 0.0
        cfg.dataset.effective_in_channels = 3
        cfg.dataset.num_classes = 10
        cfg.dataset.img_size = 28

        model = get_model(device=device, dataset_cfg=cfg.dataset, arch_cfg=cfg.architecture)
        assert isinstance(model, nn.Module)

    def test_factory_rejects_unknown_builtin(self, device):
        from orchard.architectures.factory import get_model

        cfg = MagicMock()
        cfg.architecture.name = "nonexistent_model"
        cfg.dataset.effective_in_channels = 3
        cfg.dataset.num_classes = 10
        cfg.dataset.img_size = 224

        with pytest.raises(ValueError, match="not registered"):
            get_model(device=device, dataset_cfg=cfg.dataset, arch_cfg=cfg.architecture)
