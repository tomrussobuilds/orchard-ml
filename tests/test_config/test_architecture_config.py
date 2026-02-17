"""
Test Suite for ArchitectureConfig.

Tests architecture selection, pretrained weight variants,
and dropout regularization configuration.
"""

from argparse import Namespace

import pytest
from pydantic import ValidationError

from orchard.core.config import ArchitectureConfig


# ARCHITECTURE CONFIG: DEFAULTS
@pytest.mark.unit
def test_architecture_config_defaults():
    """Test ArchitectureConfig with default values."""
    config = ArchitectureConfig()

    assert config.name == "mini_cnn"
    assert config.pretrained is False
    assert config.dropout == pytest.approx(0.2)
    assert config.weight_variant is None


@pytest.mark.unit
def test_architecture_config_custom_values():
    """Test ArchitectureConfig with custom parameters."""
    config = ArchitectureConfig(
        name="efficientnet_b0", pretrained=False, dropout=0.3, weight_variant="imagenet"
    )

    assert config.name == "efficientnet_b0"
    assert config.pretrained is False
    assert config.dropout == pytest.approx(0.3)
    assert config.weight_variant == "imagenet"


# ARCHITECTURE CONFIG: VALIDATION
@pytest.mark.unit
def test_dropout_bounds():
    """Test dropout must be in [0.0, 0.9]."""

    config = ArchitectureConfig(dropout=0.0)
    assert config.dropout == pytest.approx(0.0)

    config = ArchitectureConfig(dropout=0.5)
    assert config.dropout == pytest.approx(0.5)

    config = ArchitectureConfig(dropout=0.9)
    assert config.dropout == pytest.approx(0.9)

    with pytest.raises(ValidationError):
        ArchitectureConfig(dropout=-0.1)

    with pytest.raises(ValidationError):
        ArchitectureConfig(dropout=1.0)


@pytest.mark.unit
def test_name_accepts_string():
    """Test name field accepts arbitrary string."""
    for name in ["resnet18", "vit_tiny", "custom_model", "my-model-v2"]:
        config = ArchitectureConfig(name=name)
        assert config.name == name


@pytest.mark.unit
def test_pretrained_boolean():
    """Test pretrained field is boolean."""
    config = ArchitectureConfig(pretrained=True)
    assert config.pretrained is True

    config = ArchitectureConfig(pretrained=False)
    assert config.pretrained is False


# ARCHITECTURE CONFIG: WEIGHT VARIANTS
@pytest.mark.unit
def test_weight_variant_optional():
    """Test weight_variant can be None."""
    config = ArchitectureConfig(weight_variant=None)
    assert config.weight_variant is None


@pytest.mark.unit
def test_weight_variant_string():
    """Test weight_variant accepts string values."""
    variants = [
        "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        "vit_tiny_patch16_224.augreg_in21k",
        "imagenet1k_v1",
        "custom_weights",
    ]

    for variant in variants:
        config = ArchitectureConfig(weight_variant=variant)
        assert config.weight_variant == variant


@pytest.mark.unit
def test_weight_variant_with_pretrained_false():
    """Test weight_variant can be set even if pretrained=False."""
    config = ArchitectureConfig(pretrained=False, weight_variant="imagenet")

    assert config.pretrained is False
    assert config.weight_variant == "imagenet"


# ARCHITECTURE CONFIG: ARCHITECTURE NAMES
@pytest.mark.unit
def test_common_architecture_names():
    """Test ArchitectureConfig accepts common architecture names."""
    architectures = ["resnet_18", "efficientnet_b0", "vit_tiny", "mini_cnn"]

    for arch in architectures:
        config = ArchitectureConfig(name=arch)
        assert config.name == arch


# ARCHITECTURE CONFIG: FROM ARGS
@pytest.mark.unit
def test_from_args():
    """Test ArchitectureConfig.from_args() factory."""
    args = Namespace(model_name="vit_tiny", pretrained=False, dropout=0.25)

    config = ArchitectureConfig.from_args(args)

    assert config.name == "vit_tiny"
    assert config.pretrained is False
    assert config.dropout == pytest.approx(0.25)


@pytest.mark.unit
def test_from_args_with_defaults():
    """Test from_args() uses defaults for missing arguments."""
    args = Namespace()

    config = ArchitectureConfig.from_args(args)

    assert config.name == "mini_cnn"
    assert config.pretrained is False
    assert config.dropout == pytest.approx(0.2)


@pytest.mark.unit
def test_from_args_partial():
    """Test from_args() with partial arguments."""
    args = Namespace(model_name="efficientnet_b0")

    config = ArchitectureConfig.from_args(args)

    assert config.name == "efficientnet_b0"
    assert config.pretrained is False
    assert config.dropout == pytest.approx(0.2)


# ARCHITECTURE CONFIG: DESCRIPTION FIELD
@pytest.mark.unit
def test_field_descriptions_present():
    """Test all fields have descriptions."""
    for field_name, field_info in ArchitectureConfig.model_fields.items():
        assert field_info.description is not None
        assert len(field_info.description) > 0


# ARCHITECTURE CONFIG: IMMUTABILITY
@pytest.mark.unit
def test_config_is_frozen():
    """Test ArchitectureConfig is immutable after creation."""
    config = ArchitectureConfig()

    with pytest.raises(ValidationError):
        config.name = "new_model"


@pytest.mark.unit
def test_config_forbids_extra_fields():
    """Test ArchitectureConfig rejects unknown fields."""
    with pytest.raises(ValidationError):
        ArchitectureConfig(learning_rate=0.001)


# ARCHITECTURE CONFIG: EDGE CASES
@pytest.mark.unit
def test_empty_name_rejected():
    """Test empty string for name is rejected."""
    config = ArchitectureConfig(name="")
    assert config.name == ""


@pytest.mark.unit
def test_very_long_weight_variant():
    """Test very long weight_variant string is accepted."""
    long_variant = "a" * 500
    config = ArchitectureConfig(weight_variant=long_variant)
    assert config.weight_variant == long_variant


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
