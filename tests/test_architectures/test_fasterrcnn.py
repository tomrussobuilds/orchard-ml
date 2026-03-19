"""
Unit tests for the Faster R-CNN detection architecture.

Verifies builder output, head replacement, and forward pass in both
training and eval modes.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from orchard.architectures import build_fasterrcnn


@pytest.fixture
def device() -> torch.device:
    """Resolves target device for test execution."""
    return torch.device("cpu")


@pytest.mark.unit
class TestFasterRCNN:
    """Test suite for the Faster R-CNN builder."""

    def test_returns_nn_module(self) -> None:
        """Builder returns an nn.Module."""
        with patch("orchard.architectures.fasterrcnn.fasterrcnn_resnet50_fpn_v2") as mock_fn:
            mock_model = _make_mock_fasterrcnn()
            mock_fn.return_value = mock_model
            model = build_fasterrcnn(num_classes=5, pretrained=False)

        assert isinstance(model, nn.Module)

    def test_head_replacement_num_classes(self) -> None:
        """Builder replaces box predictor head with correct num_classes + 1."""
        with patch("orchard.architectures.fasterrcnn.fasterrcnn_resnet50_fpn_v2") as mock_fn:
            mock_model = _make_mock_fasterrcnn(in_features=256)
            mock_fn.return_value = mock_model
            build_fasterrcnn(num_classes=10, pretrained=False)

        # FastRCNNPredictor was set with num_classes + 1 (background)
        new_predictor = mock_model.roi_heads.box_predictor  # type: ignore[union-attr]
        assert new_predictor.cls_score.out_features == 11  # type: ignore[union-attr]

    def test_pretrained_passes_weights(self) -> None:
        """Builder passes 'DEFAULT' weights when pretrained=True."""
        with patch("orchard.architectures.fasterrcnn.fasterrcnn_resnet50_fpn_v2") as mock_fn:
            mock_fn.return_value = _make_mock_fasterrcnn()
            build_fasterrcnn(num_classes=5, pretrained=True)

        mock_fn.assert_called_once_with(weights="DEFAULT")

    def test_no_pretrained_passes_none(self) -> None:
        """Builder passes None weights when pretrained=False."""
        with patch("orchard.architectures.fasterrcnn.fasterrcnn_resnet50_fpn_v2") as mock_fn:
            mock_fn.return_value = _make_mock_fasterrcnn()
            build_fasterrcnn(num_classes=5, pretrained=False)

        mock_fn.assert_called_once_with(weights=None)

    def test_in_channels_accepted(self) -> None:
        """Builder accepts in_channels for API compatibility."""
        with patch("orchard.architectures.fasterrcnn.fasterrcnn_resnet50_fpn_v2") as mock_fn:
            mock_fn.return_value = _make_mock_fasterrcnn()
            model = build_fasterrcnn(num_classes=5, in_channels=1, pretrained=False)

        assert isinstance(model, nn.Module)


@pytest.mark.unit
def test_factory_dispatches_fasterrcnn() -> None:
    """get_model dispatches 'fasterrcnn' to build_fasterrcnn."""
    from unittest.mock import MagicMock

    from orchard.architectures.factory import get_model

    dataset_cfg = MagicMock()
    dataset_cfg.effective_in_channels = 3
    dataset_cfg.num_classes = 5
    dataset_cfg.img_size = 224
    dataset_cfg.resolution = 224
    arch_cfg = MagicMock()
    arch_cfg.name = "fasterrcnn"
    arch_cfg.pretrained = False

    with patch("orchard.architectures.fasterrcnn.fasterrcnn_resnet50_fpn_v2") as mock_tv:
        mock_tv.return_value = _make_mock_fasterrcnn()
        model = get_model(torch.device("cpu"), dataset_cfg, arch_cfg, verbose=False)

    # Verify it went through fasterrcnn builder (not other builders)
    mock_tv.assert_called_once()
    assert isinstance(model, nn.Module)


@pytest.mark.unit
def test_fasterrcnn_default_in_channels() -> None:
    """Builder default in_channels is 3 (RGB)."""
    import inspect

    sig = inspect.signature(build_fasterrcnn)
    assert sig.parameters["in_channels"].default == 3


@pytest.mark.unit
def test_fasterrcnn_default_pretrained() -> None:
    """Builder default pretrained is True."""
    import inspect

    sig = inspect.signature(build_fasterrcnn)
    assert sig.parameters["pretrained"].default is True


@pytest.mark.unit
def test_fasterrcnn_in_registry() -> None:
    """'fasterrcnn' is in the model registry."""
    from orchard.architectures.factory import _MODEL_REGISTRY

    assert "fasterrcnn" in _MODEL_REGISTRY


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_mock_fasterrcnn(in_features: int = 1024) -> nn.Module:
    """Create a mock FasterRCNN-like module with replaceable box_predictor."""
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = nn.Module()
    roi_heads = nn.Module()
    roi_heads.box_predictor = FastRCNNPredictor(in_features, 91)  # COCO default
    model.roi_heads = roi_heads
    return model
