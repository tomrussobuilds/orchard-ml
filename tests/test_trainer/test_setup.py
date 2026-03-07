"""
Test Suite for Optimization Setup Module.

Covers get_criterion, get_optimizer, and get_scheduler factories.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from orchard.exceptions import OrchardConfigError
from orchard.trainer import setup
from orchard.trainer.losses import FocalLoss


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


# TESTS: COMPUTE_CLASS_WEIGHTS
@pytest.mark.unit
def test_compute_class_weights_balanced():
    """Balanced classes produce equal weights."""
    labels = np.array([0, 0, 1, 1, 2, 2])
    w = setup.compute_class_weights(labels, num_classes=3, device=torch.device("cpu"))
    assert w.shape == (3,)
    assert w.dtype == torch.float
    # All weights should be 1.0 for balanced classes: 6 / (3 * 2) = 1.0
    assert torch.allclose(w, torch.ones(3))


@pytest.mark.unit
def test_compute_class_weights_imbalanced():
    """Imbalanced classes produce correct sklearn-formula weights."""
    # 4 samples of class 0, 1 of class 1 → N/(n_classes*count)
    labels = np.array([0, 0, 0, 0, 1])
    w = setup.compute_class_weights(labels, num_classes=2, device=torch.device("cpu"))
    # class 0: 5/(2*4) = 0.625, class 1: 5/(2*1) = 2.5
    assert w[0] == pytest.approx(0.625)
    assert w[1] == pytest.approx(2.5)


@pytest.mark.unit
def test_compute_class_weights_missing_class_defaults_to_one():
    """Classes absent from labels get fallback weight 1.0 (not None or 2.0)."""
    labels = np.array([0, 0, 0])
    w = setup.compute_class_weights(labels, num_classes=3, device=torch.device("cpu"))
    # class 0: 3/(3*3) = 1/3, classes 1 and 2: missing → default 1.0
    assert w[0] == pytest.approx(1.0 / 3.0, rel=1e-4)
    assert w[1] == pytest.approx(1.0)
    assert w[2] == pytest.approx(1.0)


@pytest.mark.unit
def test_compute_class_weights_dtype_is_float():
    """Weights tensor has dtype=torch.float (not default int or None)."""
    labels = np.array([0, 1])
    w = setup.compute_class_weights(labels, num_classes=2, device=torch.device("cpu"))
    assert w.dtype == torch.float


@pytest.mark.unit
def test_compute_class_weights_on_device():
    """Weights tensor is on the requested device."""
    labels = np.array([0, 1])
    w = setup.compute_class_weights(labels, num_classes=2, device=torch.device("cpu"))
    assert w.device == torch.device("cpu")


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
    with pytest.raises(OrchardConfigError, match="Unknown criterion type"):
        setup.get_criterion(base_cfg.training)


@pytest.mark.unit
def test_get_criterion_cross_entropy_label_smoothing(base_cfg):
    """CrossEntropyLoss receives label_smoothing from config (not None or omitted)."""
    base_cfg.training.criterion_type = "cross_entropy"
    base_cfg.training.label_smoothing = 0.15
    criterion = setup.get_criterion(base_cfg.training)
    assert isinstance(criterion, nn.CrossEntropyLoss)
    assert criterion.label_smoothing == pytest.approx(0.15)


@pytest.mark.unit
def test_get_criterion_cross_entropy_weighted(base_cfg):
    """CrossEntropyLoss receives class weights when weighted_loss=True."""
    base_cfg.training.criterion_type = "cross_entropy"
    base_cfg.training.weighted_loss = True
    class_weights = torch.tensor([1.0, 2.0, 0.5])
    criterion = setup.get_criterion(base_cfg.training, class_weights=class_weights)
    assert isinstance(criterion, nn.CrossEntropyLoss)
    assert criterion.weight is not None
    assert torch.equal(criterion.weight, class_weights)


@pytest.mark.unit
def test_get_criterion_cross_entropy_unweighted(base_cfg):
    """CrossEntropyLoss gets weight=None when weighted_loss=False."""
    base_cfg.training.criterion_type = "cross_entropy"
    base_cfg.training.weighted_loss = False
    class_weights = torch.tensor([1.0, 2.0])
    criterion = setup.get_criterion(base_cfg.training, class_weights=class_weights)
    assert isinstance(criterion, nn.CrossEntropyLoss)
    assert criterion.weight is None


@pytest.mark.unit
def test_get_criterion_focal_gamma(base_cfg):
    """FocalLoss receives gamma from config."""
    base_cfg.training.criterion_type = "focal"
    base_cfg.training.focal_gamma = 3.0
    criterion = setup.get_criterion(base_cfg.training)
    assert isinstance(criterion, FocalLoss)
    assert criterion.gamma == pytest.approx(3.0)


@pytest.mark.unit
def test_get_criterion_focal_weighted(base_cfg):
    """FocalLoss receives class weights when weighted_loss=True."""
    base_cfg.training.criterion_type = "focal"
    base_cfg.training.weighted_loss = True
    class_weights = torch.tensor([1.0, 2.0])
    criterion = setup.get_criterion(base_cfg.training, class_weights=class_weights)
    assert isinstance(criterion, FocalLoss)
    assert criterion.weight is not None
    assert torch.equal(criterion.weight, class_weights)


@pytest.mark.unit
def test_get_criterion_focal_unweighted(base_cfg):
    """FocalLoss gets weight=None when weighted_loss=False."""
    base_cfg.training.criterion_type = "focal"
    base_cfg.training.weighted_loss = False
    class_weights = torch.tensor([1.0, 2.0])
    criterion = setup.get_criterion(base_cfg.training, class_weights=class_weights)
    assert isinstance(criterion, FocalLoss)
    assert criterion.weight is None


# TESTS: OPTIMIZER
@pytest.mark.unit
def test_get_optimizer_sgd(base_cfg, simple_model):
    """Test SGD optimizer via optimizer_type config."""
    base_cfg.training.optimizer_type = "sgd"
    optimizer = setup.get_optimizer(simple_model, base_cfg.training)
    assert isinstance(optimizer, optim.SGD)


@pytest.mark.unit
def test_get_optimizer_sgd_params(base_cfg, simple_model):
    """SGD receives lr, momentum, weight_decay from config."""
    base_cfg.training.optimizer_type = "sgd"
    base_cfg.training.learning_rate = 0.05
    base_cfg.training.momentum = 0.8
    base_cfg.training.weight_decay = 1e-3
    opt = setup.get_optimizer(simple_model, base_cfg.training)
    pg = opt.param_groups[0]
    assert pg["lr"] == pytest.approx(0.05)
    assert pg["momentum"] == pytest.approx(0.8)
    assert pg["weight_decay"] == pytest.approx(1e-3)


@pytest.mark.unit
def test_get_optimizer_adamw(base_cfg, simple_model):
    """Test AdamW optimizer via optimizer_type config."""
    base_cfg.training.optimizer_type = "adamw"
    optimizer = setup.get_optimizer(simple_model, base_cfg.training)
    assert isinstance(optimizer, optim.AdamW)


@pytest.mark.unit
def test_get_optimizer_adamw_params(base_cfg, simple_model):
    """AdamW receives lr and weight_decay from config."""
    base_cfg.training.optimizer_type = "adamw"
    base_cfg.training.learning_rate = 0.003
    base_cfg.training.weight_decay = 0.02
    opt = setup.get_optimizer(simple_model, base_cfg.training)
    pg = opt.param_groups[0]
    assert pg["lr"] == pytest.approx(0.003)
    assert pg["weight_decay"] == pytest.approx(0.02)


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
    with pytest.raises(OrchardConfigError, match="Unknown optimizer type"):
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
def test_get_scheduler_cosine_params(base_cfg, simple_model):
    """CosineAnnealingLR receives T_max and eta_min from config."""
    base_cfg.training.scheduler_type = "cosine"
    base_cfg.training.epochs = 20
    base_cfg.training.min_lr = 1e-5
    opt = setup.get_optimizer(simple_model, base_cfg.training)
    sched = setup.get_scheduler(opt, base_cfg.training)
    assert isinstance(sched, lr_scheduler.CosineAnnealingLR)
    assert sched.T_max == 20
    assert sched.eta_min == pytest.approx(1e-5)


@pytest.mark.unit
def test_get_scheduler_plateau_params(base_cfg, simple_model):
    """ReduceLROnPlateau receives mode, factor, patience, min_lr from config."""
    base_cfg.training.scheduler_type = "plateau"
    base_cfg.training.scheduler_factor = 0.3
    base_cfg.training.scheduler_patience = 5
    base_cfg.training.min_lr = 1e-6
    opt = setup.get_optimizer(simple_model, base_cfg.training)
    sched = setup.get_scheduler(opt, base_cfg.training)
    assert isinstance(sched, lr_scheduler.ReduceLROnPlateau)
    assert sched.mode == "max"
    assert sched.factor == pytest.approx(0.3)
    assert sched.patience == 5
    assert sched.min_lrs == [pytest.approx(1e-6)]


@pytest.mark.unit
def test_get_scheduler_step_params(base_cfg, simple_model):
    """StepLR receives step_size and gamma from config."""
    base_cfg.training.scheduler_type = "step"
    base_cfg.training.step_size = 7
    base_cfg.training.scheduler_factor = 0.2
    opt = setup.get_optimizer(simple_model, base_cfg.training)
    sched = setup.get_scheduler(opt, base_cfg.training)
    assert isinstance(sched, lr_scheduler.StepLR)
    assert sched.step_size == 7
    assert sched.gamma == pytest.approx(0.2)


@pytest.mark.unit
def test_get_scheduler_none_keeps_lr_constant(base_cfg, simple_model):
    """'none' scheduler keeps LR at 1.0x (lambda returns 1.0, not 2.0)."""
    base_cfg.training.scheduler_type = "none"
    base_cfg.training.learning_rate = 0.05
    opt = setup.get_optimizer(simple_model, base_cfg.training)
    sched = setup.get_scheduler(opt, base_cfg.training)
    assert isinstance(sched, lr_scheduler.LambdaLR)
    # Step the scheduler a few times — LR should not change
    for _ in range(5):
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.05)


@pytest.mark.unit
def test_get_scheduler_invalid_type(base_cfg, simple_model):
    """Test invalid scheduler type raises ValueError with exact options list."""
    base_cfg.training.scheduler_type = "invalid_sched"
    optimizer = setup.get_optimizer(simple_model, base_cfg.training)
    with pytest.raises(OrchardConfigError, match="Unsupported scheduler_type"):
        setup.get_scheduler(optimizer, base_cfg.training)


@pytest.mark.unit
def test_get_scheduler_error_message_lists_options(base_cfg, simple_model):
    """Error message lists available options with exact casing."""
    base_cfg.training.scheduler_type = "bad"
    optimizer = setup.get_optimizer(simple_model, base_cfg.training)
    with pytest.raises(OrchardConfigError) as exc_info:
        setup.get_scheduler(optimizer, base_cfg.training)
    msg = str(exc_info.value)
    # Exact prefix — kills "XX" wrapper mutants
    assert msg.endswith("Available options: ['cosine', 'plateau', 'step', 'none']")
