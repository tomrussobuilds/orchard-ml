"""
Test Suite for Optimization Setup Module.

Covers get_criterion, get_optimizer, and get_scheduler factories.
"""

# Standard Imports
from types import SimpleNamespace

# Third-Party Imports
import pytest
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

# Internal Imports
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
        scheduler_type="cosine",
        scheduler_factor=0.5,
        scheduler_patience=2,
        step_size=3,
        label_smoothing=0.1,
        focal_gamma=2.0,
        weighted_loss=True,
        criterion_type="cross_entropy",
    )
    cfg.model = SimpleNamespace(name="resnet_18_adapted")
    return cfg


# TESTS: CRITERION
@pytest.mark.unit
@pytest.mark.parametrize("crit_type", ["cross_entropy", "bce_logit", "focal"])
def test_get_criterion_types(base_cfg, crit_type):
    """Test all valid criterion types."""
    base_cfg.training.criterion_type = crit_type

    criterion = setup.get_criterion(base_cfg)
    assert isinstance(criterion, nn.Module)


@pytest.mark.unit
def test_get_criterion_invalid_type(base_cfg):
    """Test unknown criterion type raises ValueError."""
    base_cfg.training.criterion_type = "unknown_type"
    with pytest.raises(ValueError, match="Unknown criterion type"):
        setup.get_criterion(base_cfg)


# TESTS: OPTIMIZER
@pytest.mark.unit
def test_get_optimizer_resnet(base_cfg, simple_model):
    """Test SGD optimizer for ResNet variants."""
    base_cfg.model.name = "resnet_18_adapted"
    optimizer = setup.get_optimizer(simple_model, base_cfg)
    assert isinstance(optimizer, optim.SGD)


@pytest.mark.unit
def test_get_optimizer_other_model(base_cfg, simple_model):
    """Test AdamW optimizer for non-ResNet models."""
    base_cfg.model.name = "vit_tiny"
    optimizer = setup.get_optimizer(simple_model, base_cfg)
    assert isinstance(optimizer, optim.AdamW)


# TESTS: SCHEDULER
@pytest.mark.unit
@pytest.mark.parametrize("sched_type", ["cosine", "plateau", "step", "none"])
def test_get_scheduler_types(base_cfg, simple_model, sched_type):
    """Test all scheduler types."""
    base_cfg.training.scheduler_type = sched_type
    optimizer = setup.get_optimizer(simple_model, base_cfg)
    scheduler = setup.get_scheduler(optimizer, base_cfg)

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
    optimizer = setup.get_optimizer(simple_model, base_cfg)
    with pytest.raises(ValueError, match="Unsupported scheduler_type"):
        setup.get_scheduler(optimizer, base_cfg)
