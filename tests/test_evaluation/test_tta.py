"""
Pytest test suite for the Test-Time Augmentation (TTA) module.

Validates transform selection logic and ensemble inference behavior
under anatomical, texture-based, and hardware-dependent constraints.
Forced to CPU for consistent testing.
"""

# Standard Imports
from unittest.mock import MagicMock

# Third-Party Imports
import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close

# Internal Imports
from orchard.core import Config
from orchard.evaluation import _get_tta_transforms, adaptive_tta_predict


# FIXTURES
@pytest.fixture
def device():
    """Forced CPU device for consistent unit testing."""
    return torch.device("cpu")


@pytest.fixture
def mock_cfg():
    """Returns a mock configuration object with defined augmentation parameters."""
    cfg = MagicMock(spec=Config)
    cfg.augmentation = MagicMock()
    cfg.augmentation.tta_translate = 5
    cfg.augmentation.tta_scale = 1.1
    cfg.augmentation.tta_blur_sigma = 0.5
    cfg.dataset = MagicMock()
    cfg.dataset.num_classes = 3
    return cfg


@pytest.fixture
def dummy_input():
    """Creates a dummy input tensor with batch size 4 and 3x32x32 images."""
    return torch.randn(4, 3, 32, 32)


@pytest.fixture
def mock_model(mock_cfg):
    """Creates a mock model that returns consistent logits."""
    model = MagicMock(spec=nn.Module)
    num_classes = mock_cfg.dataset.num_classes
    mock_logits = torch.randn(4, num_classes)

    # Mock both forward and __call__ for compatibility
    model.return_value = mock_logits
    model.forward.return_value = mock_logits
    model.to.return_value = model
    return model


# TEST CASES: BASE TRANSFORMS
@pytest.mark.unit
def test_get_tta_transforms_base(dummy_input, device, mock_cfg):
    """Test the generation of base transforms (identity and horizontal flip)."""
    transforms = _get_tta_transforms(
        device, is_anatomical=False, is_texture_based=False, cfg=mock_cfg
    )

    assert len(transforms) >= 2, "Base transforms (identity + flip) are missing."

    # Test identity transform
    transformed = transforms[0](dummy_input)
    assert torch.equal(transformed, dummy_input), "Identity transform modified the input."

    # Test horizontal flip
    flipped = transforms[1](dummy_input)
    assert not torch.equal(flipped, dummy_input), "Horizontal flip failed to modify the input."


@pytest.mark.unit
def test_get_tta_transforms_texture_based(dummy_input, device, mock_cfg):
    """Test the generation of texture-based transformations."""
    transforms = _get_tta_transforms(
        device, is_anatomical=False, is_texture_based=True, cfg=mock_cfg
    )

    # Should contain more than just base transforms
    assert len(transforms) > 2, "Texture-based augmentations were not added."

    # Verify a transform produces different output
    modified = transforms[-1](dummy_input)
    assert not torch.equal(
        modified, dummy_input
    ), "Texture-based transform did not change the image."


@pytest.mark.unit
def test_get_tta_transforms_gpu_rotations_mocked():
    """Test GPU branch adds rotations for non-anatomical datasets (mocked)."""
    mock_cfg = MagicMock()
    mock_cfg.augmentation.tta_translate = 2
    mock_cfg.augmentation.tta_scale = 1.05
    mock_cfg.augmentation.tta_blur_sigma = 0.5

    # Create a mock GPU device
    mock_device = MagicMock()
    mock_device.type = "cuda"  # ← Force GPU type

    # This will trigger the GPU branch even on CPU-only CI
    gpu_transforms = _get_tta_transforms(
        device=mock_device,
        is_anatomical=False,
        is_texture_based=False,
        cfg=mock_cfg,
    )

    # GPU: base(2) + texture(4) + rotations(3) = 9
    assert len(gpu_transforms) == 9


@pytest.mark.unit
def test_get_tta_transforms_cpu_fallback_vertical_flip():
    """Test CPU fallback adds vertical flip for non-anatomical datasets."""
    mock_cfg = MagicMock()
    mock_cfg.augmentation.tta_translate = 2
    mock_cfg.augmentation.tta_scale = 1.05
    mock_cfg.augmentation.tta_blur_sigma = 0.5

    # Real CPU device
    cpu_device = torch.device("cpu")

    cpu_transforms = _get_tta_transforms(
        device=cpu_device,
        is_anatomical=False,
        is_texture_based=False,
        cfg=mock_cfg,
    )

    # CPU: base(2) + texture(4) + vflip(1) = 7
    assert len(cpu_transforms) == 7

    # Verify last transform is vertical flip
    test_tensor = torch.ones(1, 3, 4, 4)
    test_tensor[0, 0, 0, :] = 0

    vflip_result = cpu_transforms[-1](test_tensor)

    # Top row moved to bottom
    assert torch.all(vflip_result[0, 0, -1, :] == 0)
    assert torch.all(vflip_result[0, 0, 0, :] == 1)


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_get_tta_transforms_gpu_rotations():
    """Test GPU adds 90/180/270 rotations for non-anatomical datasets."""
    mock_cfg = MagicMock()
    mock_cfg.augmentation.tta_translate = 2
    mock_cfg.augmentation.tta_scale = 1.05
    mock_cfg.augmentation.tta_blur_sigma = 0.5

    # GPU device, non-anatomical
    gpu_transforms = _get_tta_transforms(
        device=torch.device("cuda"),
        is_anatomical=False,
        is_texture_based=False,
        cfg=mock_cfg,
    )

    # GPU should have rotations instead of just vertical flip
    # base(2) + texture(4) + rotations(3) = 9
    assert len(gpu_transforms) == 9


# TEST CASES: ADAPTIVE TTA PREDICT
@pytest.mark.unit
def test_adaptive_tta_predict_logic(mock_model, dummy_input, device, mock_cfg):
    """Test TTA prediction logic: output shape and type validation."""
    model = mock_model
    model.to(device)

    result = adaptive_tta_predict(
        model, dummy_input, device, is_anatomical=False, is_texture_based=False, cfg=mock_cfg
    )

    assert isinstance(result, torch.Tensor)
    assert result.shape == (4, mock_cfg.dataset.num_classes)


@pytest.mark.unit
def test_tta_is_deterministic_under_eval(mock_model, dummy_input, device, mock_cfg):
    """Ensures that TTA prediction doesn't introduce random noise if model is in eval mode."""
    model = mock_model
    model.to(device)

    # First pass
    res1 = adaptive_tta_predict(model, dummy_input, device, False, False, mock_cfg)
    # Second pass
    res2 = adaptive_tta_predict(model, dummy_input, device, False, False, mock_cfg)

    assert_close(res1, res2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
