"""
Test Suite for TrainingConfig.

Tests hyperparameter validation, LR bounds, batch size limits,
and cross-field validation logic.
"""

# =========================================================================== #
#                         Standard Imports                                    #
# =========================================================================== #
import argparse

# =========================================================================== #
#                         Third-Party Imports                                 #
# =========================================================================== #
import pytest
from pydantic import ValidationError

# =========================================================================== #
#                         Internal Imports                                    #
# =========================================================================== #
from orchard.core.config import TrainingConfig

# =========================================================================== #
#                         UNIT TESTS: DEFAULTS                                #
# =========================================================================== #


@pytest.mark.unit
def test_training_config_defaults():
    """Test TrainingConfig with default values."""
    config = TrainingConfig()

    assert config.seed == 42
    assert config.batch_size == 16
    assert config.epochs == 60
    assert config.learning_rate == 0.008
    assert config.min_lr == 1e-6
    assert config.momentum == 0.9
    assert config.weight_decay == 5e-4


# =========================================================================== #
#                         UNIT TESTS: LEARNING RATE VALIDATION                #
# =========================================================================== #


@pytest.mark.unit
def test_lr_within_bounds():
    """Test valid learning rate values."""
    config = TrainingConfig(learning_rate=0.001, min_lr=1e-7)

    assert config.learning_rate == 0.001
    assert config.min_lr == 1e-7


@pytest.mark.unit
def test_lr_zero_rejected():
    """Test learning_rate=0 is rejected."""
    with pytest.raises(ValidationError):
        TrainingConfig(learning_rate=0.0)


@pytest.mark.unit
def test_lr_negative_rejected():
    """Test negative learning rate is rejected."""
    with pytest.raises(ValidationError):
        TrainingConfig(learning_rate=-0.001)


@pytest.mark.unit
def test_lr_too_large_rejected():
    """Test learning_rate > 1.0 is rejected."""
    with pytest.raises(ValidationError):
        TrainingConfig(learning_rate=1.5)


# =========================================================================== #
#                         UNIT TESTS: BATCH SIZE VALIDATION                   #
# =========================================================================== #


@pytest.mark.unit
def test_batch_size_valid_range():
    """Test batch size within valid range."""
    config = TrainingConfig(batch_size=64)
    assert config.batch_size == 64


@pytest.mark.unit
def test_batch_size_too_large_rejected():
    """Test batch_size > 128 is rejected."""
    with pytest.raises(ValidationError, match="too large"):
        TrainingConfig(batch_size=256)


@pytest.mark.unit
def test_batch_size_zero_rejected():
    """Test batch_size=0 is rejected."""
    with pytest.raises(ValidationError):
        TrainingConfig(batch_size=0)


@pytest.mark.unit
def test_batch_size_negative_rejected():
    """Test negative batch size is rejected."""
    with pytest.raises(ValidationError):
        TrainingConfig(batch_size=-1)


# =========================================================================== #
#                         UNIT TESTS: AMP VALIDATION                          #
# =========================================================================== #


@pytest.mark.unit
def test_amp_with_small_batch_rejected():
    """Test AMP + batch_size < 4 is rejected."""
    with pytest.raises(ValidationError, match="AMP.*small batch"):
        TrainingConfig(use_amp=True, batch_size=2)


@pytest.mark.unit
def test_amp_with_sufficient_batch_allowed():
    """Test AMP + batch_size >= 4 is allowed."""
    config = TrainingConfig(use_amp=True, batch_size=16)

    assert config.use_amp is True
    assert config.batch_size == 16


# =========================================================================== #
#                         UNIT TESTS: REGULARIZATION                          #
# =========================================================================== #


@pytest.mark.unit
def test_label_smoothing_bounds():
    """Test label_smoothing within valid range."""
    config = TrainingConfig(label_smoothing=0.1)
    assert config.label_smoothing == 0.1

    # Maximum
    config = TrainingConfig(label_smoothing=0.3)
    assert config.label_smoothing == 0.3


@pytest.mark.unit
def test_label_smoothing_too_large_rejected():
    """Test label_smoothing > 0.3 is rejected."""
    with pytest.raises(ValidationError):
        TrainingConfig(label_smoothing=0.5)


@pytest.mark.unit
def test_label_smoothing_negative_rejected():
    """Test negative label_smoothing is rejected."""
    with pytest.raises(ValidationError):
        TrainingConfig(label_smoothing=-0.1)


@pytest.mark.unit
def test_mixup_alpha_non_negative():
    """Test mixup_alpha >= 0."""
    config = TrainingConfig(mixup_alpha=0.2)
    assert config.mixup_alpha == 0.2

    # Zero is valid (disables mixup)
    config = TrainingConfig(mixup_alpha=0.0)
    assert config.mixup_alpha == 0.0


@pytest.mark.unit
def test_weight_decay_bounds():
    """Test weight_decay within valid range."""
    config = TrainingConfig(weight_decay=1e-4)
    assert config.weight_decay == 1e-4

    # Maximum
    config = TrainingConfig(weight_decay=0.2)
    assert config.weight_decay == 0.2


@pytest.mark.unit
def test_weight_decay_too_large_rejected():
    """Test weight_decay > 0.2 is rejected."""
    with pytest.raises(ValidationError):
        TrainingConfig(weight_decay=0.5)


# =========================================================================== #
#                         UNIT TESTS: MOMENTUM                                #
# =========================================================================== #


@pytest.mark.unit
def test_momentum_bounds():
    """Test momentum within valid range [0, 1)."""
    config = TrainingConfig(momentum=0.9)
    assert config.momentum == 0.9

    # Minimum
    config = TrainingConfig(momentum=0.0)
    assert config.momentum == 0.0


@pytest.mark.unit
def test_momentum_one_rejected():
    """Test momentum=1.0 is rejected."""
    with pytest.raises(ValidationError):
        TrainingConfig(momentum=1.0)


@pytest.mark.unit
def test_momentum_negative_rejected():
    """Test negative momentum is rejected."""
    with pytest.raises(ValidationError):
        TrainingConfig(momentum=-0.1)


# =========================================================================== #
#                         UNIT TESTS: GRADIENT CLIPPING                       #
# =========================================================================== #


@pytest.mark.unit
def test_grad_clip_valid():
    """Test gradient clipping within valid range."""
    config = TrainingConfig(grad_clip=1.0)
    assert config.grad_clip == 1.0


@pytest.mark.unit
def test_grad_clip_none_allowed():
    """Test grad_clip=None disables clipping."""
    config = TrainingConfig(grad_clip=None)
    assert config.grad_clip is None


@pytest.mark.unit
def test_grad_clip_too_large_rejected():
    """Test grad_clip > 100 is rejected."""
    with pytest.raises(ValidationError):
        TrainingConfig(grad_clip=150.0)


# =========================================================================== #
#                         UNIT TESTS: FROM_ARGS FACTORY                       #
# =========================================================================== #


@pytest.mark.unit
def test_from_args_basic():
    """Test TrainingConfig.from_args() with basic arguments."""
    args = argparse.Namespace(
        epochs=100, batch_size=32, learning_rate=0.001, min_lr=1e-7, use_amp=True
    )

    config = TrainingConfig.from_args(args)

    assert config.epochs == 100
    assert config.batch_size == 32
    assert config.learning_rate == 0.001
    assert config.use_amp is True


@pytest.mark.unit
def test_from_args_ignores_none_values():
    """Test from_args uses defaults for None values."""
    args = argparse.Namespace(epochs=100, batch_size=None)  # Should use default

    config = TrainingConfig.from_args(args)

    assert config.epochs == 100
    assert config.batch_size == 16  # Default


@pytest.mark.unit
def test_from_args_only_valid_fields():
    """Test from_args ignores invalid field names."""
    args = argparse.Namespace(epochs=100, invalid_field="should_be_ignored")

    config = TrainingConfig.from_args(args)

    assert config.epochs == 100
    assert not hasattr(config, "invalid_field")


# =========================================================================== #
#                         EDGE CASES & REGRESSION TESTS                       #
# =========================================================================== #


@pytest.mark.unit
def test_frozen_immutability():
    """Test TrainingConfig is frozen (immutable)."""
    config = TrainingConfig()

    with pytest.raises(ValidationError):
        config.epochs = 200


@pytest.mark.unit
def test_scheduler_types():
    """Test valid scheduler types."""
    for scheduler in ["cosine", "plateau", "step", "none"]:
        config = TrainingConfig(scheduler_type=scheduler)
        assert config.scheduler_type == scheduler


@pytest.mark.unit
def test_criterion_types():
    """Test valid criterion types."""
    for criterion in ["cross_entropy", "focal", "bce_logit"]:
        config = TrainingConfig(criterion_type=criterion)
        assert config.criterion_type == criterion


@pytest.mark.unit
def test_cosine_fraction_probability():
    """Test cosine_fraction is probability [0, 1]."""
    config = TrainingConfig(cosine_fraction=0.5)
    assert config.cosine_fraction == 0.5

    with pytest.raises(ValidationError):
        TrainingConfig(cosine_fraction=1.5)
