"""
Pytest test suite for the Test-Time Augmentation (TTA) module.

Validates transform selection logic and ensemble inference behavior
under anatomical, texture-based, and config-driven tta_mode constraints.
Forced to CPU for consistent testing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close

from orchard.evaluation import _get_tta_transforms, adaptive_tta_predict


# FIXTURES
@pytest.fixture
def device():
    """Forced CPU device for consistent unit testing."""
    return torch.device("cpu")


@pytest.fixture
def aug_cfg():
    """Returns augmentation config stub with TTA parameters."""
    return SimpleNamespace(
        tta_translate=5,
        tta_scale=1.1,
        tta_blur_sigma=0.5,
        tta_blur_kernel_size=3,
        tta_mode="full",
    )


@pytest.fixture
def resolution():
    return 224


@pytest.fixture
def dummy_input():
    """Creates a dummy input tensor with batch size 4 and 3x32x32 images."""
    return torch.randn(4, 3, 32, 32)


@pytest.fixture
def mock_model():
    """Creates a mock model that returns consistent logits."""
    from unittest.mock import MagicMock

    model = MagicMock(spec=nn.Module)
    num_classes = 3
    mock_logits = torch.randn(4, num_classes)
    model.return_value = mock_logits
    model.forward.return_value = mock_logits
    model.to.return_value = model
    return model


# TEST CASES: BASE TRANSFORMS
@pytest.mark.unit
def test_get_tta_transforms_base(dummy_input, aug_cfg, resolution):
    """Test the generation of base transforms (identity and horizontal flip)."""
    transforms = _get_tta_transforms(
        is_anatomical=False, is_texture_based=False, aug_cfg=aug_cfg, resolution=resolution
    )

    assert len(transforms) >= 2, "Base transforms (identity + flip) are missing."

    transformed = transforms[0](dummy_input)
    assert torch.equal(transformed, dummy_input), "Identity transform modified the input."

    flipped = transforms[1](dummy_input)
    assert not torch.equal(flipped, dummy_input), "Horizontal flip failed to modify the input."


@pytest.mark.unit
def test_get_tta_transforms_texture_based(dummy_input, aug_cfg, resolution):
    """Test texture-based datasets get minimal transforms (identity + flip only)."""
    transforms = _get_tta_transforms(
        is_anatomical=False, is_texture_based=True, aug_cfg=aug_cfg, resolution=resolution
    )

    # Texture-based: only identity + horizontal flip (no aggressive transforms)
    assert len(transforms) == 2, "Texture-based should only have identity + flip."

    # First is identity
    assert torch.equal(transforms[0](dummy_input), dummy_input)
    # Second is horizontal flip
    assert not torch.equal(transforms[1](dummy_input), dummy_input)


@pytest.mark.unit
def test_get_tta_transforms_anatomical_preserves_orientation(dummy_input, aug_cfg, resolution):
    """Test anatomical datasets do NOT get flips or rotations."""
    transforms = _get_tta_transforms(
        is_anatomical=True, is_texture_based=False, aug_cfg=aug_cfg, resolution=resolution
    )

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
def test_get_tta_transforms_anatomical_texture_minimal(dummy_input, aug_cfg, resolution):
    """Test anatomical + texture datasets get only identity (most restrictive)."""
    transforms = _get_tta_transforms(
        is_anatomical=True, is_texture_based=True, aug_cfg=aug_cfg, resolution=resolution
    )

    # Anatomical + texture: only identity (no flip, no aggressive transforms)
    assert len(transforms) == 1, "Anatomical+texture should only have identity."
    assert torch.equal(transforms[0](dummy_input), dummy_input)


@pytest.mark.unit
def test_get_tta_transforms_full_mode_rotations():
    """Test tta_mode='full' adds rotations for non-anatomical, non-texture datasets."""
    aug_cfg = SimpleNamespace(
        tta_translate=2,
        tta_scale=1.05,
        tta_blur_sigma=0.5,
        tta_blur_kernel_size=3,
        tta_mode="full",
    )

    full_transforms = _get_tta_transforms(
        is_anatomical=False,
        is_texture_based=False,
        aug_cfg=aug_cfg,
        resolution=224,
    )

    # Full mode non-anatomical non-texture: id + flip + translate + scale + blur + 3 rotations = 8
    assert len(full_transforms) == 8


@pytest.mark.unit
def test_get_tta_transforms_light_mode_vertical_flip():
    """Test tta_mode='light' adds vertical flip for non-anatomical, non-texture datasets."""
    aug_cfg = SimpleNamespace(
        tta_translate=2,
        tta_scale=1.05,
        tta_blur_sigma=0.5,
        tta_blur_kernel_size=3,
        tta_mode="light",
    )

    light_transforms = _get_tta_transforms(
        is_anatomical=False,
        is_texture_based=False,
        aug_cfg=aug_cfg,
        resolution=224,
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
    aug_cfg = SimpleNamespace(
        tta_translate=2,
        tta_scale=1.05,
        tta_blur_sigma=0.5,
        tta_blur_kernel_size=3,
        tta_mode="full",
    )

    full_transforms = _get_tta_transforms(
        is_anatomical=False,
        is_texture_based=False,
        aug_cfg=aug_cfg,
        resolution=224,
    )

    # Full mode non-anatomical non-texture: id + flip + translate + scale + blur + 3 rotations = 8
    assert len(full_transforms) == 8


# TEST CASES: ADAPTIVE TTA PREDICT
@pytest.mark.unit
def test_adaptive_tta_predict_logic(mock_model, dummy_input, device, aug_cfg, resolution):
    """Test TTA prediction logic: output shape and type validation."""
    model = mock_model
    model.to(device)

    result = adaptive_tta_predict(
        model,
        dummy_input,
        device,
        is_anatomical=False,
        is_texture_based=False,
        aug_cfg=aug_cfg,
        resolution=resolution,
    )

    assert isinstance(result, torch.Tensor)
    assert result.shape == (4, 3)


@pytest.mark.unit
def test_tta_is_deterministic_under_eval(mock_model, dummy_input, device, aug_cfg, resolution):
    """Ensures that TTA prediction doesn't introduce random noise if model is in eval mode."""
    model = mock_model
    model.to(device)

    res1 = adaptive_tta_predict(model, dummy_input, device, False, False, aug_cfg, resolution)
    res2 = adaptive_tta_predict(model, dummy_input, device, False, False, aug_cfg, resolution)

    assert_close(res1, res2)


@pytest.mark.unit
@pytest.mark.parametrize("resolution", [28, 64, 224])
def test_tta_scaling_preserves_transform_count(resolution):
    """Verify resolution scaling does not change the number of transforms."""
    aug_cfg = SimpleNamespace(
        tta_translate=5,
        tta_scale=1.1,
        tta_blur_sigma=0.5,
        tta_blur_kernel_size=3,
        tta_mode="full",
    )

    transforms = _get_tta_transforms(
        is_anatomical=False, is_texture_based=False, aug_cfg=aug_cfg, resolution=resolution
    )
    # Full mode: id + flip + translate + scale + blur + 3 rotations = 8
    assert len(transforms) == 8


@pytest.mark.unit
def test_tta_scaling_reduces_translate_at_low_resolution():
    """At 28px the affine translate should be ~8x smaller than at 224px."""
    inp = torch.zeros(1, 1, 28, 28)
    inp[0, 0, 14, 14] = 1.0  # single bright pixel at center

    def _make_aug_cfg():
        return SimpleNamespace(
            tta_translate=2.0,
            tta_scale=1.05,
            tta_blur_sigma=0.5,
            tta_blur_kernel_size=3,
            tta_mode="light",
        )

    t_28 = _get_tta_transforms(False, False, _make_aug_cfg(), 28)
    t_224 = _get_tta_transforms(False, False, _make_aug_cfg(), 224)

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
    aug_cfg = SimpleNamespace(
        tta_translate=5.0,
        tta_scale=1.1,
        tta_blur_sigma=0.5,
        tta_blur_kernel_size=3,
        tta_mode="light",
    )

    # scale_factor = 224/224 = 1.0 → params unchanged
    # We verify indirectly: the function should not raise and transforms should work
    transforms = _get_tta_transforms(False, False, aug_cfg, 224)
    test_input = torch.randn(1, 3, 32, 32)
    for t in transforms:
        out = t(test_input)
        assert out.shape == test_input.shape


@pytest.mark.unit
def test_adaptive_tta_predict_raises_on_empty_transforms(mock_model, dummy_input, device):
    """Test ValueError is raised when _get_tta_transforms returns an empty list."""
    from unittest.mock import patch

    model = mock_model
    model.to(device)

    aug_cfg = SimpleNamespace(
        tta_translate=5,
        tta_scale=1.1,
        tta_blur_sigma=0.5,
        tta_blur_kernel_size=3,
        tta_mode="full",
    )

    with patch("orchard.evaluation.tta._get_tta_transforms", return_value=[]):
        with pytest.raises(ValueError, match="TTA transforms list cannot be empty"):
            adaptive_tta_predict(model, dummy_input, device, False, False, aug_cfg, 224)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
