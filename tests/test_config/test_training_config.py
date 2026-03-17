"""
Test Suite for TrainingConfig.

Tests hyperparameter validation, LR bounds, batch size limits,
and cross-field validation logic.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orchard.core.config import TrainingConfig


# UNIT TESTS: DEFAULTS
@pytest.mark.unit
def test_training_config_defaults() -> None:
    """Test TrainingConfig with default values."""
    config = TrainingConfig()

    assert config.seed == 42
    assert config.batch_size == 16
    assert config.epochs == 60
    assert config.learning_rate == pytest.approx(0.008)
    assert config.min_lr == pytest.approx(1e-6)
    assert config.momentum == pytest.approx(0.9)
    assert config.weight_decay == pytest.approx(5e-4)


# UNIT TESTS: LEARNING RATE VALIDATION
@pytest.mark.unit
def test_lr_within_bounds() -> None:
    """Test valid learning rate values."""
    config = TrainingConfig(learning_rate=0.001, min_lr=1e-7)

    assert config.learning_rate == pytest.approx(0.001)
    assert config.min_lr == pytest.approx(1e-7)


@pytest.mark.unit
def test_lr_negative_rejected() -> None:
    """Test negative learning rate is rejected."""
    with pytest.raises(ValidationError):
        TrainingConfig(learning_rate=-0.001)


# UNIT TESTS: BATCH SIZE VALIDATION
@pytest.mark.unit
def test_batch_size_valid_range() -> None:
    """Test batch size within valid range."""
    config = TrainingConfig(batch_size=64)
    assert config.batch_size == 64


@pytest.mark.unit
def test_batch_size_too_large_rejected() -> None:
    """Test batch_size > 128 is rejected."""
    with pytest.raises(ValidationError, match="less than or equal to 128"):
        TrainingConfig(batch_size=256)


@pytest.mark.unit
def test_batch_size_zero_rejected() -> None:
    """Test batch_size=0 is rejected."""
    with pytest.raises(ValidationError):
        TrainingConfig(batch_size=0)


@pytest.mark.unit
def test_batch_size_negative_rejected() -> None:
    """Test negative batch size is rejected."""
    with pytest.raises(ValidationError):
        TrainingConfig(batch_size=-1)


# UNIT TESTS: AMP VALIDATION
@pytest.mark.unit
def test_amp_with_small_batch_rejected() -> None:
    """Test AMP + batch_size < 4 is rejected."""
    with pytest.raises(ValidationError, match="AMP.*small batch"):
        TrainingConfig(use_amp=True, batch_size=2)


@pytest.mark.unit
def test_amp_with_sufficient_batch_allowed() -> None:
    """Test AMP + batch_size >= 4 is allowed."""
    config = TrainingConfig(use_amp=True, batch_size=16)

    assert config.use_amp is True
    assert config.batch_size == 16


# UNIT TESTS: REGULARIZATION
@pytest.mark.unit
def test_label_smoothing_bounds() -> None:
    """Test label_smoothing within valid range."""
    config = TrainingConfig(label_smoothing=0.1)
    assert config.label_smoothing == pytest.approx(0.1)

    # Maximum
    config = TrainingConfig(label_smoothing=0.3)
    assert config.label_smoothing == pytest.approx(0.3)


@pytest.mark.unit
def test_label_smoothing_too_large_rejected() -> None:
    """Test label_smoothing > 0.3 is rejected."""
    with pytest.raises(ValidationError):
        TrainingConfig(label_smoothing=0.5)


@pytest.mark.unit
def test_label_smoothing_negative_rejected() -> None:
    """Test negative label_smoothing is rejected."""
    with pytest.raises(ValidationError):
        TrainingConfig(label_smoothing=-0.1)


@pytest.mark.unit
def test_mixup_alpha_non_negative() -> None:
    """Test mixup_alpha >= 0."""
    config = TrainingConfig(mixup_alpha=0.2)
    assert config.mixup_alpha == pytest.approx(0.2)

    # Zero is valid (disables mixup)
    config = TrainingConfig(mixup_alpha=0.0)
    assert config.mixup_alpha == pytest.approx(0.0)


@pytest.mark.unit
def test_weight_decay_bounds() -> None:
    """Test weight_decay within valid range."""
    config = TrainingConfig(weight_decay=1e-4)
    assert config.weight_decay == pytest.approx(1e-4)

    # Maximum
    config = TrainingConfig(weight_decay=0.2)
    assert config.weight_decay == pytest.approx(0.2)


@pytest.mark.unit
def test_weight_decay_too_large_rejected() -> None:
    """Test weight_decay > 0.2 is rejected."""
    with pytest.raises(ValidationError):
        TrainingConfig(weight_decay=0.5)


# UNIT TESTS: MOMENTUM
@pytest.mark.unit
def test_momentum_bounds() -> None:
    """Test momentum within valid range [0, 1)."""
    config = TrainingConfig(momentum=0.9)
    assert config.momentum == pytest.approx(0.9)
    config = TrainingConfig(momentum=0.0)
    assert config.momentum == pytest.approx(0.0)


@pytest.mark.unit
def test_momentum_one_rejected() -> None:
    """Test momentum=1.0 is rejected."""
    with pytest.raises(ValidationError):
        TrainingConfig(momentum=1.0)


@pytest.mark.unit
def test_momentum_negative_rejected() -> None:
    """Test negative momentum is rejected."""
    with pytest.raises(ValidationError):
        TrainingConfig(momentum=-0.1)


# UNIT TESTS: GRADIENT CLIPPING
@pytest.mark.unit
def test_grad_clip_valid() -> None:
    """Test gradient clipping within valid range."""
    config = TrainingConfig(grad_clip=1.0)
    assert config.grad_clip == pytest.approx(1.0)


@pytest.mark.unit
def test_grad_clip_none_allowed() -> None:
    """Test grad_clip=None disables clipping."""
    config = TrainingConfig(grad_clip=None)
    assert config.grad_clip is None


@pytest.mark.unit
def test_grad_clip_too_large_rejected() -> None:
    """Test grad_clip > 100 is rejected."""
    with pytest.raises(ValidationError):
        TrainingConfig(grad_clip=150.0)


# EDGE CASES & REGRESSION TESTS
@pytest.mark.unit
def test_frozen_immutability() -> None:
    """Test TrainingConfig is frozen (immutable)."""
    config = TrainingConfig()

    with pytest.raises(ValidationError):
        config.epochs = 200


@pytest.mark.unit
def test_scheduler_types() -> None:
    """Test valid scheduler types."""
    for scheduler in ["cosine", "plateau", "step", "none"]:
        config = TrainingConfig(scheduler_type=scheduler)
        assert config.scheduler_type == scheduler


@pytest.mark.unit
def test_criterion_types() -> None:
    """Test valid criterion types."""
    for criterion in ["cross_entropy", "focal"]:
        config = TrainingConfig(criterion_type=criterion)
        assert config.criterion_type == criterion


@pytest.mark.unit
def test_monitor_metric_accepts_any_string() -> None:
    """Test monitor_metric accepts any non-empty string (task-agnostic)."""
    for metric in ["auc", "accuracy", "f1", "mAP", "loss", "mae"]:
        config = TrainingConfig(monitor_metric=metric)
        assert config.monitor_metric == metric


@pytest.mark.unit
def test_monitor_direction_values() -> None:
    """Test monitor_direction accepts maximize/minimize and rejects invalid."""
    for direction in ["maximize", "minimize"]:
        config = TrainingConfig(monitor_direction=direction)
        assert config.monitor_direction == direction

    with pytest.raises(ValidationError):
        TrainingConfig(monitor_direction="invalid")


@pytest.mark.unit
def test_monitor_direction_default() -> None:
    """Test monitor_direction defaults to maximize."""
    config = TrainingConfig()
    assert config.monitor_direction == "maximize"


# UNIT TESTS: OPTIMIZER TYPE
@pytest.mark.unit
def test_optimizer_type_default() -> None:
    """Test optimizer_type defaults to sgd."""
    config = TrainingConfig()
    assert config.optimizer_type == "sgd"


@pytest.mark.unit
def test_optimizer_type_adamw() -> None:
    """Test optimizer_type accepts adamw."""
    config = TrainingConfig(optimizer_type="adamw")
    assert config.optimizer_type == "adamw"


@pytest.mark.unit
def test_optimizer_type_invalid_rejected() -> None:
    """Test invalid optimizer_type is rejected by Literal."""
    with pytest.raises(ValidationError):
        TrainingConfig(optimizer_type="adam")
