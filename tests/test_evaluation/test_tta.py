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


#                                Test Fixtures                               #
@pytest.fixture
def device():
    """
    Forced CPU device for consistent unit testing.
    """
    return torch.device("cpu")


@pytest.fixture
def mock_cfg():
    """
    Returns a mock configuration object with defined augmentation parameters.
    """
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
    """
    Creates a dummy input tensor with batch size 4 and 3x32x32 images.
    """
    return torch.randn(4, 3, 32, 32)


@pytest.fixture
def mock_model(mock_cfg):
    """
    Creates a mock model that returns consistent logits.
    """
    model = MagicMock(spec=nn.Module)
    # Mocking both forward and __call__ to ensure compatibility
    num_classes = mock_cfg.dataset.num_classes
    mock_logits = torch.randn(4, num_classes)

    model.return_value = mock_logits
    model.forward.return_value = mock_logits
    # Ensure to_device doesn't break the mock
    model.to.return_value = model
    return model


#                              Test Cases                                     #
@pytest.mark.unit
def test_get_tta_transforms_base(dummy_input, device, mock_cfg):
    """
    Test the generation of base transforms (identity and horizontal flip).
    """
    transforms = _get_tta_transforms(
        device, is_anatomical=False, is_texture_based=False, cfg=mock_cfg
    )

    assert len(transforms) >= 2, "Base transforms (identity + flip) are missing."

    # Test identity transform (Index 0 is typically Identity)
    transformed = transforms[0](dummy_input)
    assert torch.equal(transformed, dummy_input), "Identity transform modified the input."

    # Test horizontal flip (Index 1)
    flipped = transforms[1](dummy_input)
    assert not torch.equal(flipped, dummy_input), "Horizontal flip failed to modify the input."


@pytest.mark.unit
def test_get_tta_transforms_texture_based(dummy_input, device, mock_cfg):
    """
    Test the generation of texture-based transformations.
    """
    transforms = _get_tta_transforms(
        device, is_anatomical=False, is_texture_based=True, cfg=mock_cfg
    )

    # Should contain more than just base transforms
    assert len(transforms) > 2, "Texture-based augmentations were not added."

    # Verify a transform (e.g., Gaussian Blur or Affine) produces a different output
    modified = transforms[-1](dummy_input)
    assert not torch.equal(
        modified, dummy_input
    ), "Texture-based transform did not change the image."


@pytest.mark.unit
def test_adaptive_tta_predict_logic(mock_model, dummy_input, device, mock_cfg):
    """
    Test TTA prediction logic: output shape and type validation.
    """
    model = mock_model
    model.to(device)

    result = adaptive_tta_predict(
        model, dummy_input, device, is_anatomical=False, is_texture_based=False, cfg=mock_cfg
    )

    assert isinstance(result, torch.Tensor)
    assert result.shape == (4, mock_cfg.dataset.num_classes)


@pytest.mark.unit
def test_tta_is_deterministic_under_eval(mock_model, dummy_input, device, mock_cfg):
    """
    Ensures that TTA prediction doesn't introduce random noise if model is in eval mode.
    """
    model = mock_model
    model.to(device)

    # First pass
    res1 = adaptive_tta_predict(model, dummy_input, device, False, False, mock_cfg)
    # Second pass
    res2 = adaptive_tta_predict(model, dummy_input, device, False, False, mock_cfg)

    assert_close(res1, res2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
