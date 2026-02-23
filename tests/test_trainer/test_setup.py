"""
Test Suite for Optimization Setup Module.

Covers get_criterion, get_optimizer, and get_scheduler factories.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from orchard.trainer import setup


# FIXTURES
@pytest.fixture
def simple_model():
    """Simple linear model for testing."""
    return nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))


@pytest.fixture
def base_cfg():
    """Mock Config as a SimpleNamespace to satisfy factories."""
    cfg = SimpleNamespace()
    cfg.training = SimpleNamespace(
        epochs=10,
        learning_rate=0.01,
        min_lr=0.001,
        momentum=0.9,
        weight_decay=1e-4,
        optimizer_type="sgd",
        scheduler_type="cosine",
        scheduler_factor=0.5,
        scheduler_patience=2,
        step_size=3,
        label_smoothing=0.1,
        focal_gamma=2.0,
        weighted_loss=True,
        criterion_type="cross_entropy",
    )
    cfg.architecture = SimpleNamespace(name="resnet_18")
    return cfg


# TESTS: CRITERION
@pytest.mark.unit
@pytest.mark.parametrize("crit_type", ["cross_entropy", "focal"])
def test_get_criterion_types(base_cfg, crit_type):
    """Test all valid criterion types."""
    base_cfg.training.criterion_type = crit_type

    criterion = setup.get_criterion(base_cfg.training)
    assert isinstance(criterion, nn.Module)


@pytest.mark.unit
def test_get_criterion_invalid_type(base_cfg):
    """Test unknown criterion type raises ValueError."""
    base_cfg.training.criterion_type = "unknown_type"
    with pytest.raises(ValueError, match="Unknown criterion type"):
        setup.get_criterion(base_cfg.training)


# TESTS: OPTIMIZER
@pytest.mark.unit
def test_get_optimizer_sgd(base_cfg, simple_model):
    """Test SGD optimizer via optimizer_type config."""
    base_cfg.training.optimizer_type = "sgd"
    optimizer = setup.get_optimizer(simple_model, base_cfg.training)
    assert isinstance(optimizer, optim.SGD)


@pytest.mark.unit
def test_get_optimizer_adamw(base_cfg, simple_model):
    """Test AdamW optimizer via optimizer_type config."""
    base_cfg.training.optimizer_type = "adamw"
    optimizer = setup.get_optimizer(simple_model, base_cfg.training)
    assert isinstance(optimizer, optim.AdamW)


@pytest.mark.unit
def test_get_optimizer_adamw_with_resnet_name(base_cfg, simple_model):
    """Test AdamW is used when optimizer_type=adamw regardless of model name."""
    base_cfg.training.optimizer_type = "adamw"
    base_cfg.architecture.name = "resnet_18"
    optimizer = setup.get_optimizer(simple_model, base_cfg.training)
    assert isinstance(optimizer, optim.AdamW)


@pytest.mark.unit
def test_get_optimizer_invalid_type(base_cfg, simple_model):
    """Test unknown optimizer type raises ValueError."""
    base_cfg.training.optimizer_type = "invalid_opt"
    with pytest.raises(ValueError, match="Unknown optimizer type"):
        setup.get_optimizer(simple_model, base_cfg.training)


# TESTS: SCHEDULER
@pytest.mark.unit
@pytest.mark.parametrize("sched_type", ["cosine", "plateau", "step", "none"])
def test_get_scheduler_types(base_cfg, simple_model, sched_type):
    """Test all scheduler types."""
    base_cfg.training.scheduler_type = sched_type
    optimizer = setup.get_optimizer(simple_model, base_cfg.training)
    scheduler = setup.get_scheduler(optimizer, base_cfg.training)

    if sched_type == "cosine":
        assert isinstance(scheduler, lr_scheduler.CosineAnnealingLR)
    elif sched_type == "plateau":
        assert isinstance(scheduler, lr_scheduler.ReduceLROnPlateau)
    elif sched_type == "step":
        assert isinstance(scheduler, lr_scheduler.StepLR)
    elif sched_type == "none":
        assert isinstance(scheduler, lr_scheduler.LambdaLR)


@pytest.mark.unit
def test_get_scheduler_invalid_type(base_cfg, simple_model):
    """Test invalid scheduler type raises ValueError."""
    base_cfg.training.scheduler_type = "invalid_sched"
    optimizer = setup.get_optimizer(simple_model, base_cfg.training)
    with pytest.raises(ValueError, match="Unsupported scheduler_type"):
        setup.get_scheduler(optimizer, base_cfg.training)
