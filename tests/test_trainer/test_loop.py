"""
Test Suite for TrainingLoop and Factory Functions.

Tests cover the shared training loop kernel, AMP scaler factory,
and MixUp factory function.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from orchard.trainer._loop import LoopOptions, TrainingLoop, create_amp_scaler, create_mixup_fn

# ── FIXTURES ───────────────────────────────────────────────────────────────


@pytest.fixture
def mock_cfg():
    """Mock Config with training parameters."""
    cfg = MagicMock()
    cfg.training.use_amp = False
    cfg.training.mixup_alpha = 0.0
    cfg.training.mixup_epochs = 0
    cfg.training.seed = 42
    cfg.training.grad_clip = 1.0
    cfg.training.epochs = 5
    cfg.training.use_tqdm = False
    return cfg


@pytest.fixture
def loop():
    """TrainingLoop instance with mocked components."""
    return TrainingLoop(
        model=MagicMock(spec=nn.Module),
        train_loader=MagicMock(),
        val_loader=MagicMock(),
        optimizer=MagicMock(),
        scheduler=MagicMock(),
        criterion=MagicMock(spec=nn.Module),
        device=torch.device("cpu"),
        scaler=None,
        mixup_fn=None,
        options=LoopOptions(
            grad_clip=1.0,
            total_epochs=5,
            mixup_epochs=3,
            use_tqdm=False,
            monitor_metric="auc",
        ),
    )


# ── TESTS: FACTORY FUNCTIONS ──────────────────────────────────────────────


@pytest.mark.unit
def test_create_amp_scaler_disabled(mock_cfg):
    """AMP scaler is None when use_amp=False."""
    mock_cfg.training.use_amp = False
    assert create_amp_scaler(mock_cfg) is None


@pytest.mark.filterwarnings("ignore:.*GradScaler is enabled, but CUDA is not available.*")
@pytest.mark.unit
def test_create_amp_scaler_enabled(mock_cfg):
    """AMP scaler is GradScaler when use_amp=True."""
    mock_cfg.training.use_amp = True
    scaler = create_amp_scaler(mock_cfg)
    assert isinstance(scaler, torch.amp.GradScaler)


@pytest.mark.unit
def test_create_mixup_fn_disabled(mock_cfg):
    """MixUp function is None when alpha=0."""
    mock_cfg.training.mixup_alpha = 0.0
    assert create_mixup_fn(mock_cfg) is None


@pytest.mark.unit
def test_create_mixup_fn_enabled(mock_cfg):
    """MixUp function is callable when alpha > 0."""
    mock_cfg.training.mixup_alpha = 0.4
    fn = create_mixup_fn(mock_cfg)
    assert fn is not None
    assert callable(fn)


@pytest.mark.unit
def test_create_mixup_fn_deterministic(mock_cfg):
    """Two calls with same seed produce same MixUp lambdas."""
    mock_cfg.training.mixup_alpha = 0.4
    mock_cfg.training.seed = 123

    fn1 = create_mixup_fn(mock_cfg)
    fn2 = create_mixup_fn(mock_cfg)

    x = torch.randn(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))

    _, _, _, lam1 = fn1(x, y)
    _, _, _, lam2 = fn2(x, y)
    assert lam1 == pytest.approx(lam2)


# ── TESTS: TrainingLoop.run_train_step ─────────────────────────────────────


@pytest.mark.unit
@patch("orchard.trainer._loop.train_one_epoch", return_value=0.42)
def test_run_train_step_basic(mock_train, loop):
    """run_train_step delegates to train_one_epoch and returns loss."""
    loss = loop.run_train_step(epoch=1)
    assert loss == pytest.approx(0.42)
    mock_train.assert_called_once()


@pytest.mark.unit
@patch("orchard.trainer._loop.train_one_epoch", return_value=0.5)
def test_run_train_step_mixup_cutoff(mock_train, loop):
    """MixUp is passed when epoch <= mixup_epochs, None after."""
    loop.mixup_fn = MagicMock()

    # Epoch 2 → mixup active
    loop.run_train_step(epoch=2)
    call_kwargs = mock_train.call_args[1]
    assert call_kwargs["mixup_fn"] is loop.mixup_fn

    mock_train.reset_mock()

    # Epoch 4 → mixup disabled
    loop.run_train_step(epoch=4)
    call_kwargs = mock_train.call_args[1]
    assert call_kwargs["mixup_fn"] is None


@pytest.mark.unit
@patch("orchard.trainer._loop.train_one_epoch", return_value=0.3)
def test_run_train_step_passes_all_args(mock_train, loop):
    """All loop attributes are forwarded to train_one_epoch."""
    loop.run_train_step(epoch=2)
    call_kwargs = mock_train.call_args[1]

    assert call_kwargs["model"] is loop.model
    assert call_kwargs["loader"] is loop.train_loader
    assert call_kwargs["criterion"] is loop.criterion
    assert call_kwargs["optimizer"] is loop.optimizer
    assert call_kwargs["device"] is loop.device
    assert call_kwargs["scaler"] is loop.scaler
    assert call_kwargs["grad_clip"] == loop.options.grad_clip
    assert call_kwargs["epoch"] == 2
    assert call_kwargs["total_epochs"] == loop.options.total_epochs
    assert call_kwargs["use_tqdm"] is loop.options.use_tqdm


# ── TESTS: TrainingLoop.run_epoch ──────────────────────────────────────────


@pytest.mark.unit
@patch("orchard.trainer._loop.step_scheduler")
@patch(
    "orchard.trainer._loop.validate_epoch",
    return_value={"loss": 0.3, "accuracy": 0.9, "auc": 0.95},
)
@patch("orchard.trainer._loop.train_one_epoch", return_value=0.42)
def test_run_epoch_returns_loss_and_metrics(mock_train, mock_val, mock_sched, loop):
    """run_epoch returns (train_loss, val_metrics) tuple."""
    train_loss, val_metrics = loop.run_epoch(epoch=1)

    assert train_loss == pytest.approx(0.42)
    assert val_metrics["accuracy"] == pytest.approx(0.9)
    assert val_metrics["auc"] == pytest.approx(0.95)


@pytest.mark.unit
@patch("orchard.trainer._loop.step_scheduler")
@patch(
    "orchard.trainer._loop.validate_epoch",
    return_value={"loss": 0.3, "accuracy": 0.9, "auc": 0.95},
)
@patch("orchard.trainer._loop.train_one_epoch", return_value=0.42)
def test_run_epoch_steps_scheduler(mock_train, mock_val, mock_sched, loop):
    """run_epoch calls step_scheduler with scheduler and monitor_metric value."""
    loop.run_epoch(epoch=1)

    mock_sched.assert_called_once_with(loop.scheduler, 0.95)


@pytest.mark.unit
@patch("orchard.trainer._loop.step_scheduler")
@patch(
    "orchard.trainer._loop.validate_epoch",
    return_value={"loss": 0.3, "accuracy": 0.9, "auc": 0.95},
)
@patch("orchard.trainer._loop.train_one_epoch", return_value=0.42)
def test_run_epoch_calls_validate_with_correct_args(mock_train, mock_val, mock_sched, loop):
    """run_epoch passes model, val_loader, criterion, device to validate_epoch."""
    loop.run_epoch(epoch=1)

    mock_val.assert_called_once_with(
        model=loop.model,
        val_loader=loop.val_loader,
        criterion=loop.criterion,
        device=loop.device,
    )
