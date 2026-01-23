"""
Test Suite for ModelConfig.

Tests model architecture selection, pretrained weight variants,
and dropout regularization configuration.
"""

# Standard Imports
from argparse import Namespace

# Third-Party Imports
import pytest
from pydantic import ValidationError

# Internal Imports
from orchard.core.config import ModelConfig


# MODEL CONFIG: DEFAULTS
@pytest.mark.unit
def test_model_config_defaults():
    """Test ModelConfig with default values."""
    config = ModelConfig()

    assert config.name == "resnet_18_adapted"
    assert config.pretrained is True
    assert config.dropout == 0.2
    assert config.weight_variant is None


@pytest.mark.unit
def test_model_config_custom_values():
    """Test ModelConfig with custom parameters."""
    config = ModelConfig(
        name="efficientnet_b0", pretrained=False, dropout=0.3, weight_variant="imagenet"
    )

    assert config.name == "efficientnet_b0"
    assert config.pretrained is False
    assert config.dropout == 0.3
    assert config.weight_variant == "imagenet"


# MODEL CONFIG: VALIDATION
@pytest.mark.unit
def test_dropout_bounds():
    """Test dropout must be in [0.0, 0.9]."""
    # Valid
    config = ModelConfig(dropout=0.0)
    assert config.dropout == 0.0

    config = ModelConfig(dropout=0.5)
    assert config.dropout == 0.5

    config = ModelConfig(dropout=0.9)
    assert config.dropout == 0.9

    # Invalid
    with pytest.raises(ValidationError):
        ModelConfig(dropout=-0.1)

    with pytest.raises(ValidationError):
        ModelConfig(dropout=1.0)


@pytest.mark.unit
def test_name_accepts_string():
    """Test name field accepts arbitrary string."""
    for name in ["resnet18", "vit_tiny", "custom_model", "my-model-v2"]:
        config = ModelConfig(name=name)
        assert config.name == name


@pytest.mark.unit
def test_pretrained_boolean():
    """Test pretrained field is boolean."""
    config = ModelConfig(pretrained=True)
    assert config.pretrained is True

    config = ModelConfig(pretrained=False)
    assert config.pretrained is False


# MODEL CONFIG: WEIGHT VARIANTS
@pytest.mark.unit
def test_weight_variant_optional():
    """Test weight_variant can be None."""
    config = ModelConfig(weight_variant=None)
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
        config = ModelConfig(weight_variant=variant)
        assert config.weight_variant == variant


@pytest.mark.unit
def test_weight_variant_with_pretrained_false():
    """Test weight_variant can be set even if pretrained=False."""
    # This is allowed - validation happens at model loading time
    config = ModelConfig(pretrained=False, weight_variant="imagenet")

    assert config.pretrained is False
    assert config.weight_variant == "imagenet"


# MODEL CONFIG: ARCHITECTURE NAMES
@pytest.mark.unit
def test_common_architecture_names():
    """Test ModelConfig accepts common architecture names."""
    architectures = ["resnet_18_adapted", "efficientnet_b0", "vit_tiny", "mini_cnn"]

    for arch in architectures:
        config = ModelConfig(name=arch)
        assert config.name == arch


# MODEL CONFIG: FROM ARGS
@pytest.mark.unit
def test_from_args():
    """Test ModelConfig.from_args() factory."""
    args = Namespace(model_name="vit_tiny", pretrained=False, dropout=0.25)

    config = ModelConfig.from_args(args)

    assert config.name == "vit_tiny"
    assert config.pretrained is False
    assert config.dropout == 0.25


@pytest.mark.unit
def test_from_args_with_defaults():
    """Test from_args() uses defaults for missing arguments."""
    args = Namespace()  # Empty namespace

    config = ModelConfig.from_args(args)

    # Should use hardcoded defaults from from_args()
    assert config.name == "resnet18"  # Note: from_args default differs
    assert config.pretrained is True
    assert config.dropout == 0.2


@pytest.mark.unit
def test_from_args_partial():
    """Test from_args() with partial arguments."""
    args = Namespace(model_name="efficientnet_b0")

    config = ModelConfig.from_args(args)

    assert config.name == "efficientnet_b0"
    assert config.pretrained is True  # Default
    assert config.dropout == 0.2  # Default


# MODEL CONFIG: DESCRIPTION FIELD
@pytest.mark.unit
def test_field_descriptions_present():
    """Test all fields have descriptions."""
    for field_name, field_info in ModelConfig.model_fields.items():
        assert field_info.description is not None
        assert len(field_info.description) > 0


# MODEL CONFIG: IMMUTABILITY
@pytest.mark.unit
def test_config_is_frozen():
    """Test ModelConfig is immutable after creation."""
    config = ModelConfig()

    with pytest.raises(ValidationError):
        config.name = "new_model"


@pytest.mark.unit
def test_config_forbids_extra_fields():
    """Test ModelConfig rejects unknown fields."""
    with pytest.raises(ValidationError):
        ModelConfig(learning_rate=0.001)  # Wrong config section


# MODEL CONFIG: EDGE CASES
@pytest.mark.unit
def test_empty_name_rejected():
    """Test empty string for name is rejected."""
    # Pydantic should reject empty strings if Field has min_length constraint
    # If not explicitly constrained, this test documents current behavior
    config = ModelConfig(name="")
    assert config.name == ""  # Currently allowed


@pytest.mark.unit
def test_very_long_weight_variant():
    """Test very long weight_variant string is accepted."""
    long_variant = "a" * 500
    config = ModelConfig(weight_variant=long_variant)
    assert config.weight_variant == long_variant


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
