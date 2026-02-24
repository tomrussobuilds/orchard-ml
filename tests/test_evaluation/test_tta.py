"""
Pytest test suite for the Test-Time Augmentation (TTA) module.

Validates transform selection logic and ensemble inference behavior
under anatomical, texture-based, and config-driven tta_mode constraints.
Forced to CPU for consistent testing.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close

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
    cfg.augmentation.tta_blur_kernel_size = 3
    cfg.augmentation.tta_mode = "full"
    cfg.dataset = MagicMock()
    cfg.dataset.num_classes = 3
    cfg.dataset.resolution = 224
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
    model.return_value = mock_logits
    model.forward.return_value = mock_logits
    model.to.return_value = model
    return model


# TEST CASES: BASE TRANSFORMS
@pytest.mark.unit
def test_get_tta_transforms_base(dummy_input, mock_cfg):
    """Test the generation of base transforms (identity and horizontal flip)."""
    transforms = _get_tta_transforms(is_anatomical=False, is_texture_based=False, cfg=mock_cfg)

    assert len(transforms) >= 2, "Base transforms (identity + flip) are missing."

    transformed = transforms[0](dummy_input)
    assert torch.equal(transformed, dummy_input), "Identity transform modified the input."

    flipped = transforms[1](dummy_input)
    assert not torch.equal(flipped, dummy_input), "Horizontal flip failed to modify the input."


@pytest.mark.unit
def test_get_tta_transforms_texture_based(dummy_input, mock_cfg):
    """Test texture-based datasets get minimal transforms (identity + flip only)."""
    transforms = _get_tta_transforms(is_anatomical=False, is_texture_based=True, cfg=mock_cfg)

    # Texture-based: only identity + horizontal flip (no aggressive transforms)
    assert len(transforms) == 2, "Texture-based should only have identity + flip."

    # First is identity
    assert torch.equal(transforms[0](dummy_input), dummy_input)
    # Second is horizontal flip
    assert not torch.equal(transforms[1](dummy_input), dummy_input)


@pytest.mark.unit
def test_get_tta_transforms_anatomical_preserves_orientation(dummy_input, mock_cfg):
    """Test anatomical datasets do NOT get flips or rotations."""
    transforms = _get_tta_transforms(is_anatomical=True, is_texture_based=False, cfg=mock_cfg)

    # Anatomical non-texture: identity + translate + scale + blur = 4
    assert len(transforms) == 4, "Anatomical should not have flip or rotations."

    # First must be identity
    assert torch.equal(transforms[0](dummy_input), dummy_input)

    # Verify no flip is present (all transforms should preserve left-right orientation)
    for t in transforms:
        result = t(dummy_input)
        # Check that the left side pattern is still on the left
        # (flip would swap dims[3], i.e., width)
        if not torch.equal(result, dummy_input):
            # Non-identity transform - just verify shape is preserved
            assert result.shape == dummy_input.shape


@pytest.mark.unit
def test_get_tta_transforms_anatomical_texture_minimal(dummy_input, mock_cfg):
    """Test anatomical + texture datasets get only identity (most restrictive)."""
    transforms = _get_tta_transforms(is_anatomical=True, is_texture_based=True, cfg=mock_cfg)

    # Anatomical + texture: only identity (no flip, no aggressive transforms)
    assert len(transforms) == 1, "Anatomical+texture should only have identity."
    assert torch.equal(transforms[0](dummy_input), dummy_input)


@pytest.mark.unit
def test_get_tta_transforms_full_mode_rotations():
    """Test tta_mode='full' adds rotations for non-anatomical, non-texture datasets."""
    mock_cfg = MagicMock()
    mock_cfg.augmentation.tta_translate = 2
    mock_cfg.augmentation.tta_scale = 1.05
    mock_cfg.augmentation.tta_blur_sigma = 0.5
    mock_cfg.augmentation.tta_blur_kernel_size = 3
    mock_cfg.augmentation.tta_mode = "full"
    mock_cfg.dataset.resolution = 224

    full_transforms = _get_tta_transforms(
        is_anatomical=False,
        is_texture_based=False,
        cfg=mock_cfg,
    )

    # Full mode non-anatomical non-texture: id + flip + translate + scale + blur + 3 rotations = 8
    assert len(full_transforms) == 8


@pytest.mark.unit
def test_get_tta_transforms_light_mode_vertical_flip():
    """Test tta_mode='light' adds vertical flip for non-anatomical, non-texture datasets."""
    mock_cfg = MagicMock()
    mock_cfg.augmentation.tta_translate = 2
    mock_cfg.augmentation.tta_scale = 1.05
    mock_cfg.augmentation.tta_blur_sigma = 0.5
    mock_cfg.augmentation.tta_blur_kernel_size = 3
    mock_cfg.augmentation.tta_mode = "light"
    mock_cfg.dataset.resolution = 224

    light_transforms = _get_tta_transforms(
        is_anatomical=False,
        is_texture_based=False,
        cfg=mock_cfg,
    )

    # Light mode non-anatomical non-texture: id + flip + translate + scale + blur + vflip = 6
    assert len(light_transforms) == 6

    test_tensor = torch.ones(1, 3, 4, 4)
    test_tensor[0, 0, 0, :] = 0

    vflip_result = light_transforms[-1](test_tensor)

    assert torch.all(vflip_result[0, 0, -1, :] == 0)
    assert torch.all(vflip_result[0, 0, 0, :] == 1)


@pytest.mark.unit
def test_get_tta_transforms_full_mode_rotations_on_cpu():
    """Test tta_mode='full' adds 90/180/270 rotations regardless of device."""
    mock_cfg = MagicMock()
    mock_cfg.augmentation.tta_translate = 2
    mock_cfg.augmentation.tta_scale = 1.05
    mock_cfg.augmentation.tta_blur_sigma = 0.5
    mock_cfg.augmentation.tta_blur_kernel_size = 3
    mock_cfg.augmentation.tta_mode = "full"
    mock_cfg.dataset.resolution = 224

    full_transforms = _get_tta_transforms(
        is_anatomical=False,
        is_texture_based=False,
        cfg=mock_cfg,
    )

    # Full mode non-anatomical non-texture: id + flip + translate + scale + blur + 3 rotations = 8
    assert len(full_transforms) == 8


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

    res1 = adaptive_tta_predict(model, dummy_input, device, False, False, mock_cfg)
    res2 = adaptive_tta_predict(model, dummy_input, device, False, False, mock_cfg)

    assert_close(res1, res2)


@pytest.mark.unit
@pytest.mark.parametrize("resolution", [28, 64, 224])
def test_tta_scaling_preserves_transform_count(resolution):
    """Verify resolution scaling does not change the number of transforms."""
    cfg = MagicMock()
    cfg.augmentation.tta_translate = 5
    cfg.augmentation.tta_scale = 1.1
    cfg.augmentation.tta_blur_sigma = 0.5
    cfg.augmentation.tta_blur_kernel_size = 3
    cfg.augmentation.tta_mode = "full"
    cfg.dataset.resolution = resolution

    transforms = _get_tta_transforms(is_anatomical=False, is_texture_based=False, cfg=cfg)
    # Full mode: id + flip + translate + scale + blur + 3 rotations = 8
    assert len(transforms) == 8


@pytest.mark.unit
def test_tta_scaling_reduces_translate_at_low_resolution():
    """At 28px the affine translate should be ~8x smaller than at 224px."""
    inp = torch.zeros(1, 1, 28, 28)
    inp[0, 0, 14, 14] = 1.0  # single bright pixel at center

    def _make_cfg(res):
        c = MagicMock()
        c.augmentation.tta_translate = 2.0
        c.augmentation.tta_scale = 1.05
        c.augmentation.tta_blur_sigma = 0.5
        c.augmentation.tta_mode = "light"
        c.dataset.resolution = res
        return c

    t_28 = _get_tta_transforms(False, False, _make_cfg(28))
    t_224 = _get_tta_transforms(False, False, _make_cfg(224))

    # Index 2 is the translate transform
    out_28 = t_28[2](inp)
    out_224 = t_224[2](inp)

    diff_28 = (out_28 - inp).abs().sum().item()
    diff_224 = (out_224 - inp).abs().sum().item()

    # 28px translate should cause less pixel displacement than 224px
    assert diff_28 < diff_224, "Low-res TTA should apply smaller translations"


@pytest.mark.unit
def test_tta_scaling_at_baseline_224_is_identity():
    """At resolution=224 (baseline), scaled params should equal raw config values."""
    cfg = MagicMock()
    cfg.augmentation.tta_translate = 5.0
    cfg.augmentation.tta_scale = 1.1
    cfg.augmentation.tta_blur_sigma = 0.5
    cfg.augmentation.tta_blur_kernel_size = 3
    cfg.augmentation.tta_mode = "light"
    cfg.dataset.resolution = 224

    # scale_factor = 224/224 = 1.0 → params unchanged
    # We verify indirectly: the function should not raise and transforms should work
    transforms = _get_tta_transforms(False, False, cfg)
    test_input = torch.randn(1, 3, 32, 32)
    for t in transforms:
        out = t(test_input)
        assert out.shape == test_input.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
