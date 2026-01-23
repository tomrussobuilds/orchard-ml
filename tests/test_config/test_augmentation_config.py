"""
Test Suite for AugmentationConfig.

Tests training augmentation parameters and TTA configuration
with validation of probability ranges and geometric constraints.
"""

# Standard Imports
from argparse import Namespace

# Third-Party Imports
import pytest
from pydantic import ValidationError

# Internal Imports
from orchard.core.config import AugmentationConfig


# AUGMENTATION CONFIG: DEFAULTS
@pytest.mark.unit
def test_augmentation_config_defaults():
    """Test AugmentationConfig with default values."""
    config = AugmentationConfig()

    assert config.hflip == 0.5
    assert config.rotation_angle == 10
    assert config.jitter_val == 0.2
    assert config.min_scale == 0.9
    assert config.tta_translate == 2.0
    assert config.tta_scale == 1.1
    assert config.tta_blur_sigma == 0.4


@pytest.mark.unit
def test_augmentation_config_custom_values():
    """Test AugmentationConfig with custom parameters."""
    config = AugmentationConfig(hflip=0.7, rotation_angle=15, jitter_val=0.3, min_scale=0.85)

    assert config.hflip == 0.7
    assert config.rotation_angle == 15
    assert config.jitter_val == 0.3
    assert config.min_scale == 0.85


# AUGMENTATION CONFIG: VALIDATION
@pytest.mark.unit
def test_hflip_probability_bounds():
    """Test hflip probability must be in [0, 1]."""
    # Valid
    config = AugmentationConfig(hflip=0.0)
    assert config.hflip == 0.0

    config = AugmentationConfig(hflip=1.0)
    assert config.hflip == 1.0

    # Invalid
    with pytest.raises(ValidationError):
        AugmentationConfig(hflip=-0.1)

    with pytest.raises(ValidationError):
        AugmentationConfig(hflip=1.5)


@pytest.mark.unit
def test_rotation_angle_bounds():
    """Test rotation_angle must be in [0, 360]."""
    # Valid
    config = AugmentationConfig(rotation_angle=0)
    assert config.rotation_angle == 0

    config = AugmentationConfig(rotation_angle=360)
    assert config.rotation_angle == 360

    # Invalid
    with pytest.raises(ValidationError):
        AugmentationConfig(rotation_angle=-10)

    with pytest.raises(ValidationError):
        AugmentationConfig(rotation_angle=400)


@pytest.mark.unit
def test_jitter_val_non_negative():
    """Test jitter_val must be non-negative."""
    # Valid
    config = AugmentationConfig(jitter_val=0.0)
    assert config.jitter_val == 0.0

    # Invalid
    with pytest.raises(ValidationError):
        AugmentationConfig(jitter_val=-0.1)


@pytest.mark.unit
def test_min_scale_probability_bounds():
    """Test min_scale must be in (0, 1]."""
    # Valid
    config = AugmentationConfig(min_scale=0.5)
    assert config.min_scale == 0.5

    config = AugmentationConfig(min_scale=1.0)
    assert config.min_scale == 1.0

    with pytest.raises(ValidationError):
        AugmentationConfig(min_scale=1.5)


# AUGMENTATION CONFIG: TTA PARAMS
@pytest.mark.unit
def test_tta_translate_bounds():
    """Test tta_translate must be in [0, 50]."""
    # Valid
    config = AugmentationConfig(tta_translate=0.0)
    assert config.tta_translate == 0.0

    config = AugmentationConfig(tta_translate=10.0)
    assert config.tta_translate == 10.0

    # Invalid
    with pytest.raises(ValidationError):
        AugmentationConfig(tta_translate=-1.0)

    with pytest.raises(ValidationError):
        AugmentationConfig(tta_translate=100.0)


@pytest.mark.unit
def test_tta_scale_bounds():
    """Test tta_scale must be in (0, 2]."""
    # Valid
    config = AugmentationConfig(tta_scale=1.0)
    assert config.tta_scale == 1.0

    config = AugmentationConfig(tta_scale=2.0)
    assert config.tta_scale == 2.0

    # Invalid
    with pytest.raises(ValidationError):
        AugmentationConfig(tta_scale=0.0)

    with pytest.raises(ValidationError):
        AugmentationConfig(tta_scale=3.0)


@pytest.mark.unit
def test_tta_blur_sigma_bounds():
    """Test tta_blur_sigma must be in [0, 5]."""
    # Valid
    config = AugmentationConfig(tta_blur_sigma=0.0)
    assert config.tta_blur_sigma == 0.0

    config = AugmentationConfig(tta_blur_sigma=5.0)
    assert config.tta_blur_sigma == 5.0

    # Invalid
    with pytest.raises(ValidationError):
        AugmentationConfig(tta_blur_sigma=-0.5)

    with pytest.raises(ValidationError):
        AugmentationConfig(tta_blur_sigma=10.0)


# AUGMENTATION CONFIG: FROM ARGS
@pytest.mark.unit
def test_from_args():
    """Test AugmentationConfig.from_args() factory."""
    args = Namespace(hflip=0.7, rotation_angle=20, jitter_val=0.25, min_scale=0.8)

    config = AugmentationConfig.from_args(args)

    assert config.hflip == 0.7
    assert config.rotation_angle == 20
    assert config.jitter_val == 0.25
    assert config.min_scale == 0.8


@pytest.mark.unit
def test_from_args_partial():
    """Test from_args() with partial arguments uses defaults."""
    args = Namespace(hflip=0.3)

    config = AugmentationConfig.from_args(args)

    assert config.hflip == 0.3
    assert config.rotation_angle == 10  # Default
    assert config.jitter_val == 0.2  # Default


@pytest.mark.unit
def test_from_args_empty():
    """Test from_args() with no arguments uses all defaults."""
    args = Namespace()

    config = AugmentationConfig.from_args(args)

    assert config.hflip == 0.5
    assert config.rotation_angle == 10
    assert config.jitter_val == 0.2


# AUGMENTATION CONFIG: IMMUTABILITY
@pytest.mark.unit
def test_config_is_frozen():
    """Test AugmentationConfig is immutable after creation."""
    config = AugmentationConfig()

    with pytest.raises(ValidationError):
        config.hflip = 0.9


@pytest.mark.unit
def test_config_forbids_extra_fields():
    """Test AugmentationConfig rejects unknown fields."""
    with pytest.raises(ValidationError):
        AugmentationConfig(unknown_field="value")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
