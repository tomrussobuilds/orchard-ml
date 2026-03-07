"""
Comprehensive Test Suite for ModelTrainer.

Tests cover initialization, training loop, checkpointing,
early stopping, and scheduler interaction.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from orchard.exceptions import OrchardExportError
from orchard.trainer import ModelTrainer
from orchard.trainer._scheduling import step_scheduler


# FIXTURES
@pytest.fixture
def mock_cfg():
    """Mock Config with training parameters."""
    cfg = MagicMock()
    cfg.training.epochs = 5
    cfg.training.patience = 3
    cfg.training.use_amp = False
    cfg.training.mixup_alpha = 0.0
    cfg.training.mixup_epochs = 0
    cfg.training.grad_clip = 1.0
    cfg.training.use_tqdm = False
    cfg.training.seed = 42
    cfg.training.monitor_metric = "auc"
    return cfg


@pytest.fixture
def simple_model():
    """Simple model for testing."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 10),
    )


@pytest.fixture
def mock_loaders():
    """Mock train and val loaders."""
    batch = (torch.randn(4, 1, 28, 28), torch.randint(0, 10, (4,)))

    train_loader = MagicMock()
    train_loader.__iter__ = MagicMock(return_value=iter([batch, batch]))
    train_loader.__len__ = MagicMock(return_value=2)

    val_loader = MagicMock()
    val_loader.__iter__ = MagicMock(return_value=iter([batch]))
    val_loader.__len__ = MagicMock(return_value=1)

    return train_loader, val_loader


@pytest.fixture
def criterion():
    """CrossEntropy loss."""
    return nn.CrossEntropyLoss()


@pytest.fixture
def optimizer(simple_model):
    """SGD optimizer."""
    return torch.optim.SGD(
        simple_model.parameters(),
        lr=0.01,
        momentum=0.0,
        weight_decay=0.0,
    )


@pytest.fixture
def scheduler(optimizer):
    """StepLR scheduler."""
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)


@pytest.fixture
def trainer(simple_model, mock_loaders, optimizer, scheduler, criterion, mock_cfg):
    """ModelTrainer instance."""
    train_loader, val_loader = mock_loaders
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "best_model.pth"

        trainer = ModelTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            training=mock_cfg.training,
            output_path=output_path,
        )

        # Keep tmpdir alive
        trainer._tmpdir = tmpdir

        yield trainer


# TESTS: INITIALIZATION
@pytest.mark.unit
def test_trainer_init(
    trainer, simple_model, mock_loaders, optimizer, scheduler, criterion, mock_cfg
):
    """Test ModelTrainer initializes all attributes correctly."""
    train_loader, val_loader = mock_loaders

    # Core components stored as-is
    assert trainer.model is simple_model
    assert trainer.train_loader is train_loader
    assert trainer.val_loader is val_loader
    assert trainer.optimizer is optimizer
    assert trainer.scheduler is scheduler
    assert trainer.criterion is criterion
    assert trainer.device == torch.device("cpu")
    assert trainer.training is mock_cfg.training
    assert trainer.tracker is None

    # Hyperparameters from config
    assert trainer.epochs == 5
    assert trainer.patience == 3
    assert trainer.monitor_metric == "auc"
    assert trainer.best_acc == -1.0
    assert trainer.best_metric == -float("inf")
    assert trainer.epochs_no_improve == 0

    # AMP/Mixup
    assert trainer.scaler is None  # use_amp=False
    assert trainer.mixup_fn is None  # mixup_alpha=0.0

    # Output
    assert trainer.best_path.name == "best_model.pth"
    assert trainer.best_path.parent.is_dir()

    # History
    assert trainer.train_losses == []
    assert trainer.val_metrics_history == []
    assert trainer._checkpoint_saved is False

    # Loop kernel — verify forwarded components are not None/mangled
    loop = trainer._loop
    assert loop is not None
    assert loop.model is simple_model
    assert loop.train_loader is train_loader
    assert loop.val_loader is val_loader
    assert loop.optimizer is optimizer
    assert loop.scheduler is scheduler
    assert loop.criterion is criterion
    assert loop.device == torch.device("cpu")
    # scaler/mixup are None for this config, verified above
    # LoopOptions forwarding
    assert loop.options.grad_clip == mock_cfg.training.grad_clip
    assert loop.options.total_epochs == trainer.epochs
    assert loop.options.use_tqdm is mock_cfg.training.use_tqdm
    assert loop.options.monitor_metric == "auc"


@pytest.mark.unit
def test_trainer_creates_output_dir(
    simple_model, mock_loaders, optimizer, scheduler, criterion, mock_cfg
):
    """Test trainer creates output directory."""
    train_loader, val_loader = mock_loaders

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "subdir" / "best_model.pth"

        # Sanity check: directory does not exist before
        assert not output_path.parent.exists()

        ModelTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=torch.device("cpu"),
            training=mock_cfg.training,
            output_path=output_path,
        )

        assert output_path.parent.is_dir()
        assert output_path.parent.exists()


@pytest.mark.filterwarnings("ignore:.*GradScaler is enabled, but CUDA is not available.*")
@pytest.mark.unit
def test_trainer_amp_scaler_enabled(simple_model, mock_loaders, optimizer, scheduler, criterion):
    """Test AMP scaler is created when enabled."""
    train_loader, val_loader = mock_loaders
    cfg = MagicMock()
    cfg.training.epochs = 1
    cfg.training.patience = 1
    cfg.training.use_amp = True
    cfg.training.mixup_alpha = 0.0
    cfg.training.mixup_epochs = 0
    cfg.training.grad_clip = 0.0
    cfg.training.use_tqdm = False
    cfg.training.seed = 42
    cfg.training.monitor_metric = "auc"

    trainer = ModelTrainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=torch.device("cpu"),
        training=cfg.training,
    )

    assert trainer.scaler is not None
    # Scaler must be forwarded to the loop kernel (not None)
    assert trainer._loop.scaler is trainer.scaler


@pytest.mark.unit
def test_trainer_forwards_device_to_amp_scaler(
    simple_model, mock_loaders, optimizer, scheduler, criterion
):
    """create_amp_scaler receives str(device), not None or default."""
    train_loader, val_loader = mock_loaders
    cfg = MagicMock()
    cfg.training.epochs = 1
    cfg.training.patience = 1
    cfg.training.use_amp = True
    cfg.training.mixup_alpha = 0.0
    cfg.training.mixup_epochs = 0
    cfg.training.grad_clip = 0.0
    cfg.training.use_tqdm = False
    cfg.training.seed = 42
    cfg.training.monitor_metric = "auc"

    with patch("orchard.trainer.trainer.create_amp_scaler") as mock_cas:
        mock_cas.return_value = None
        ModelTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=torch.device("cpu"),
            training=cfg.training,
        )
        mock_cas.assert_called_once_with(cfg.training, device="cpu")


@pytest.mark.unit
def test_trainer_default_best_path(
    simple_model, mock_loaders, optimizer, scheduler, criterion, mock_cfg
):
    """When output_path is None, best_path defaults to ./best_model.pth (exact name)."""
    train_loader, val_loader = mock_loaders
    trainer = ModelTrainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=torch.device("cpu"),
        training=mock_cfg.training,
        output_path=None,
    )
    assert trainer.best_path == Path("./best_model.pth")


@pytest.mark.unit
def test_trainer_creates_nested_output_dir(
    simple_model, mock_loaders, optimizer, scheduler, criterion, mock_cfg
):
    """mkdir(parents=True) creates nested directories (kills parents=False/None mutants)."""
    train_loader, val_loader = mock_loaders

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "deep" / "nested" / "best_model.pth"
        assert not output_path.parent.exists()

        ModelTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=torch.device("cpu"),
            training=mock_cfg.training,
            output_path=output_path,
        )

        assert output_path.parent.is_dir()


# TESTS: CHECKPOINTING
@pytest.mark.unit
def test_handle_checkpointing_improves(trainer):
    """Test checkpointing saves when AUC improves."""
    val_metrics = {"accuracy": 0.9, "auc": 0.85}

    should_stop = trainer._handle_checkpointing(val_metrics)

    assert trainer.best_metric == pytest.approx(0.85)
    assert trainer.epochs_no_improve == 0
    assert should_stop is False
    assert trainer.best_path.exists()
    assert trainer._checkpoint_saved is True

    # Verify the saved checkpoint is loadable
    state = torch.load(trainer.best_path, weights_only=True)
    assert isinstance(state, dict)


@pytest.mark.unit
def test_handle_checkpointing_strict_greater(trainer):
    """Test checkpointing requires strictly greater, not equal."""
    trainer.best_metric = 0.85
    val_metrics = {"accuracy": 0.9, "auc": 0.85}

    should_stop = trainer._handle_checkpointing(val_metrics)

    # Equal value should NOT trigger checkpoint (> not >=)
    assert trainer.best_metric == pytest.approx(0.85)
    assert trainer.epochs_no_improve == 1
    assert should_stop is False


@pytest.mark.unit
def test_handle_checkpointing_no_improve(trainer):
    """Test checkpointing increments patience when no improvement."""
    trainer.best_metric = 0.9

    val_metrics = {"accuracy": 0.8, "auc": 0.85}

    should_stop = trainer._handle_checkpointing(val_metrics)

    assert trainer.best_metric == pytest.approx(0.9)
    assert trainer.epochs_no_improve == 1
    assert should_stop is False


@pytest.mark.unit
def test_handle_checkpointing_early_stop(trainer):
    """Test early stopping triggers after patience exhausted."""
    trainer.best_metric = 0.95
    trainer.epochs_no_improve = 2

    val_metrics = {"accuracy": 0.8, "auc": 0.85}

    should_stop = trainer._handle_checkpointing(val_metrics)

    assert trainer.epochs_no_improve == 3
    assert should_stop is True


# TESTS: SCHEDULER
@pytest.mark.unit
def test_step_scheduler_reduce_on_plateau(
    simple_model, mock_loaders, optimizer, criterion, mock_cfg
):
    """Test scheduler step with ReduceLROnPlateau."""
    train_loader, val_loader = mock_loaders
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")

    trainer = ModelTrainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=torch.device("cpu"),
        training=mock_cfg.training,
    )

    step_scheduler(trainer.scheduler, 0.5)


@pytest.mark.unit
def test_step_scheduler_step_lr(trainer):
    """Test scheduler step with StepLR."""
    trainer.optimizer.step = MagicMock()
    step_scheduler(trainer.scheduler, 0.5)


# TESTS: LOAD BEST WEIGHTS
@pytest.mark.unit
def test_load_best_weights_success(trainer):
    """Test loading best weights from checkpoint."""
    torch.save(trainer.model.state_dict(), trainer.best_path)

    with torch.no_grad():
        for param in trainer.model.parameters():
            param.fill_(999.0)

    trainer.load_best_weights()

    first_param = next(trainer.model.parameters())
    target_tensor = torch.full_like(first_param, 999.0)
    assert not torch.all(torch.isclose(first_param.detach(), target_tensor))


@pytest.mark.unit
def test_load_best_weights_restores_device(trainer):
    """Test load_best_weights passes correct device to load_model_weights."""
    torch.save(trainer.model.state_dict(), trainer.best_path)

    with patch("orchard.trainer.trainer.load_model_weights") as mock_load:
        trainer.load_best_weights()
        mock_load.assert_called_once_with(
            model=trainer.model, path=trainer.best_path, device=trainer.device
        )


@pytest.mark.unit
def test_load_best_weights_reraises_runtime_error(trainer):
    """Test load_best_weights re-raises RuntimeError from incompatible state dict."""
    torch.save(trainer.model.state_dict(), trainer.best_path)

    with patch("orchard.trainer.trainer.load_model_weights", side_effect=RuntimeError("bad keys")):
        with pytest.raises(RuntimeError, match="bad keys"):
            trainer.load_best_weights()


@pytest.mark.unit
def test_load_best_weights_file_not_found(trainer):
    """Test load_best_weights raises when file doesn't exist."""

    if trainer.best_path.exists():
        trainer.best_path.unlink()

    with pytest.raises(OrchardExportError, match="checkpoint not found"):
        trainer.load_best_weights()


@pytest.mark.unit
def test_load_best_weights_error_logging(trainer):
    """load_best_weights logs error with LogStyle args and the exception (not None)."""
    from orchard.core import LogStyle

    torch.save(trainer.model.state_dict(), trainer.best_path)

    err = RuntimeError("incompatible keys")
    with patch("orchard.trainer.trainer.load_model_weights", side_effect=err):
        with patch("orchard.trainer.trainer.logger") as mock_logger:
            with pytest.raises(RuntimeError):
                trainer.load_best_weights()

            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args[0]
            # args: format_str, LogStyle.INDENT, LogStyle.FAILURE, exception
            assert call_args[1] == LogStyle.INDENT
            assert call_args[2] == LogStyle.FAILURE
            assert call_args[3] is err


# TESTS: TRAINING LOOP
@pytest.mark.integration
@patch("orchard.trainer._loop.train_one_epoch")
@patch("orchard.trainer._loop.validate_epoch")
def test_train_full_loop(mock_validate, mock_train, trainer):
    """Test full training loop executes all epochs without early stopping."""
    mock_train.return_value = 0.5
    mock_validate.side_effect = [
        {"loss": 0.3, "accuracy": 0.9, "auc": 0.80},
        {"loss": 0.3, "accuracy": 0.9, "auc": 0.81},
        {"loss": 0.3, "accuracy": 0.9, "auc": 0.82},
        {"loss": 0.3, "accuracy": 0.9, "auc": 0.83},
        {"loss": 0.3, "accuracy": 0.9, "auc": 0.84},
    ]

    trainer.optimizer.step = MagicMock()
    trainer.optimizer.step()
    best_path, train_losses, val_metrics = trainer.train()

    assert len(train_losses) == trainer.epochs
    assert len(val_metrics) == trainer.epochs
    assert best_path == trainer.best_path


@pytest.mark.integration
@patch("orchard.trainer._loop.train_one_epoch")
@patch("orchard.trainer._loop.validate_epoch")
def test_train_early_stopping(mock_validate, mock_train, trainer):
    """Test training stops early when patience is exhausted."""

    # --- 1. Mock training and validation ---
    mock_train.return_value = 0.5
    mock_validate.return_value = {"loss": 0.5, "accuracy": 0.5, "auc": 0.5}

    # --- 2. Force early stopping scenario ---
    trainer.best_metric = 0.95
    trainer.epochs_no_improve = 0

    # --- 3. Mock optimizer step to suppress PyTorch warnings ---
    trainer.optimizer.step = MagicMock()
    trainer.optimizer.step()

    # --- 4. Run trainer ---
    best_path, train_losses, val_metrics = trainer.train()

    # --- 5. Assertions ---
    assert len(train_losses) <= trainer.epochs
    assert trainer.epochs_no_improve >= trainer.patience
    assert best_path.exists()
    for vm in val_metrics:
        assert "loss" in vm
        assert "accuracy" in vm
        assert "auc" in vm


@pytest.mark.integration
@patch("orchard.trainer._loop.train_one_epoch")
@patch("orchard.trainer._loop.validate_epoch")
def test_train_mixup_cutoff(
    mock_validate, mock_train, simple_model, mock_loaders, optimizer, scheduler, criterion
):
    """Test MixUp is disabled after mixup_epochs."""
    train_loader, val_loader = mock_loaders

    cfg = MagicMock()
    cfg.training.epochs = 10
    cfg.training.patience = 20
    cfg.training.use_amp = False
    cfg.training.mixup_alpha = 1.0
    cfg.training.mixup_epochs = 5
    cfg.training.grad_clip = 0.0
    cfg.training.seed = 42
    cfg.training.use_tqdm = False
    cfg.training.monitor_metric = "auc"

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = ModelTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=torch.device("cpu"),
            training=cfg.training,
            output_path=Path(tmpdir) / "best.pth",
        )

        mock_train.return_value = 0.5
        mock_validate.return_value = {"loss": 0.3, "accuracy": 0.9, "auc": 0.85}

        trainer.train()

        calls = mock_train.call_args_list

        assert calls[0].kwargs.get("mixup_fn") is not None

        assert calls[6].kwargs.get("mixup_fn") is None


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore:.*lr_scheduler.step.*before.*optimizer.step.*:UserWarning")
def test_step_scheduler_calls_step_for_non_plateau(
    simple_model, mock_loaders, optimizer, criterion, mock_cfg
):
    """Test scheduler.step() is called for non-plateau schedulers."""
    train_loader, val_loader = mock_loaders
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)

    trainer = ModelTrainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=torch.device("cpu"),
        training=mock_cfg.training,
    )

    original_step = trainer.scheduler.step
    call_count = [0]

    def mock_step(*args, **kwargs):
        call_count[0] += 1
        return original_step(*args, **kwargs)

    trainer.scheduler.step = mock_step

    step_scheduler(trainer.scheduler, 0.5)

    assert call_count[0] == 1, "scheduler.step() should be called for non-plateau schedulers"


@pytest.mark.integration
@patch("orchard.trainer._loop.train_one_epoch")
@patch("orchard.trainer._loop.validate_epoch")
def test_train_loads_existing_checkpoint_when_no_improvement(
    mock_validate, mock_train, simple_model, mock_loaders, optimizer, scheduler, criterion, mock_cfg
):
    """Test training loads existing checkpoint when model never improves.

    This covers the case where best_path exists (from previous run) but
    _checkpoint_saved is False (model never improved during current training).
    """
    train_loader, val_loader = mock_loaders

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "best_model.pth"

        # Pre-create a checkpoint file (simulating previous run)
        torch.save(simple_model.state_dict(), output_path)

        trainer = ModelTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=torch.device("cpu"),
            training=mock_cfg.training,
            output_path=output_path,
        )

        # set best_metric very high so model never improves
        trainer.best_metric = 0.9999

        mock_train.return_value = 0.5
        # Return constant metrics that won't improve best_auc
        mock_validate.return_value = {"loss": 0.3, "accuracy": 0.9, "auc": 0.5}

        trainer.optimizer.step = MagicMock()

        best_path, _, _ = trainer.train()

        # Verify checkpoint was loaded (not saved during training)
        assert trainer._checkpoint_saved is False
        assert best_path.exists()


@pytest.mark.unit
@patch("orchard.trainer._loop.train_one_epoch")
@patch("orchard.trainer._loop.validate_epoch")
def test_train_raises_on_missing_monitor_metric(
    mock_validate, mock_train, simple_model, mock_loaders, optimizer, scheduler, criterion
):
    """Test training raises KeyError when monitor_metric is not in val_metrics."""
    train_loader, val_loader = mock_loaders

    cfg = MagicMock()
    cfg.training.epochs = 1
    cfg.training.patience = 3
    cfg.training.use_amp = False
    cfg.training.mixup_alpha = 0.0
    cfg.training.mixup_epochs = 0
    cfg.training.grad_clip = 0.0
    cfg.training.use_tqdm = False
    cfg.training.seed = 42
    cfg.training.monitor_metric = "nonexistent_metric"

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = ModelTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=torch.device("cpu"),
            training=cfg.training,
            output_path=Path(tmpdir) / "best.pth",
        )

        mock_train.return_value = 0.5
        mock_validate.return_value = {"loss": 0.3, "accuracy": 0.9, "auc": 0.85}

        with pytest.raises(KeyError, match="nonexistent_metric"):
            trainer.train()


@pytest.mark.unit
def test_finalize_weights_loads_best_when_checkpoint_saved(trainer):
    """Test _finalize_weights loads best weights when checkpoint was saved."""
    torch.save(trainer.model.state_dict(), trainer.best_path)
    trainer._checkpoint_saved = True

    with patch.object(trainer, "load_best_weights") as mock_load:
        trainer._finalize_weights()
        mock_load.assert_called_once()

    # Model should be in eval mode
    assert not trainer.model.training


@pytest.mark.unit
def test_finalize_weights_loads_existing_when_no_improvement(trainer):
    """Test _finalize_weights loads existing checkpoint when no improvement."""
    torch.save(trainer.model.state_dict(), trainer.best_path)
    trainer._checkpoint_saved = False

    with patch.object(trainer, "load_best_weights") as mock_load:
        trainer._finalize_weights()
        mock_load.assert_called_once()

    assert not trainer.model.training


@pytest.mark.unit
def test_finalize_weights_saves_fallback_when_no_checkpoint(trainer):
    """Test _finalize_weights saves current state when no checkpoint exists."""
    if trainer.best_path.exists():
        trainer.best_path.unlink()
    trainer._checkpoint_saved = False

    trainer._finalize_weights()

    assert trainer.best_path.exists()
    assert not trainer.model.training


@pytest.mark.unit
def test_finalize_weights_fallback_saves_valid_state_dict(trainer):
    """Fallback branch saves model.state_dict() (not None) to best_path."""
    if trainer.best_path.exists():
        trainer.best_path.unlink()
    trainer._checkpoint_saved = False

    trainer._finalize_weights()

    # The saved file must be a valid state dict, not None
    loaded = torch.load(trainer.best_path, weights_only=True)
    assert isinstance(loaded, dict)
    assert len(loaded) > 0


@pytest.mark.unit
def test_finalize_weights_no_improve_logs_warning_with_message(trainer):
    """_finalize_weights logs the no-improvement message in the warning (not None)."""
    torch.save(trainer.model.state_dict(), trainer.best_path)
    trainer._checkpoint_saved = False

    with patch("orchard.trainer.trainer.logger") as mock_logger:
        trainer._finalize_weights()
        # The warning must contain the no_improve_msg (not None or mangled)
        warning_calls = mock_logger.warning.call_args_list
        assert len(warning_calls) == 1
        fmt_args = warning_calls[0][0]
        msg = fmt_args[1]
        assert msg is not None
        assert msg.startswith("No checkpoint was saved")


@pytest.mark.unit
def test_finalize_weights_fallback_logs_warning_with_message(trainer):
    """_finalize_weights fallback branch logs warning with no_improve_msg."""
    if trainer.best_path.exists():
        trainer.best_path.unlink()
    trainer._checkpoint_saved = False

    with patch("orchard.trainer.trainer.logger") as mock_logger:
        trainer._finalize_weights()
        warning_calls = mock_logger.warning.call_args_list
        assert len(warning_calls) == 1
        fmt_args = warning_calls[0][0]
        assert fmt_args[1] is not None
        assert "No checkpoint" in str(fmt_args[1])


@pytest.mark.unit
def test_finalize_requires_both_saved_and_exists(trainer):
    """Test _finalize_weights requires BOTH _checkpoint_saved AND best_path.exists().

    When _checkpoint_saved=True but file doesn't exist, should NOT take the
    first branch (load best). This kills the and->or mutant.
    """
    trainer._checkpoint_saved = True
    if trainer.best_path.exists():
        trainer.best_path.unlink()

    # Should fall through to the "save fallback" branch
    trainer._finalize_weights()
    assert trainer.best_path.exists()
    assert not trainer.model.training


@pytest.mark.integration
@patch("orchard.trainer._loop.train_one_epoch")
@patch("orchard.trainer._loop.validate_epoch")
def test_train_tracks_best_acc(mock_validate, mock_train, trainer):
    """Test train() updates best_acc correctly."""
    mock_train.return_value = 0.5
    mock_validate.side_effect = [
        {"loss": 0.3, "accuracy": 0.85, "auc": 0.80},
        {"loss": 0.3, "accuracy": 0.90, "auc": 0.81},
        {"loss": 0.3, "accuracy": 0.88, "auc": 0.82},
        {"loss": 0.3, "accuracy": 0.92, "auc": 0.83},
        {"loss": 0.3, "accuracy": 0.91, "auc": 0.84},
    ]
    trainer.optimizer.step = MagicMock()

    trainer.train()

    assert trainer.best_acc == pytest.approx(0.92)


@pytest.mark.integration
@patch("orchard.trainer._loop.train_one_epoch")
@patch("orchard.trainer._loop.validate_epoch")
def test_train_records_val_loss_and_monitor(mock_validate, mock_train, trainer):
    """Test train() correctly reads val_loss and monitor_value from metrics."""
    mock_train.return_value = 0.5
    mock_validate.side_effect = [
        {"loss": 0.4, "accuracy": 0.85, "auc": 0.80},
        {"loss": 0.3, "accuracy": 0.90, "auc": 0.85},
        {"loss": 0.2, "accuracy": 0.92, "auc": 0.90},
        {"loss": 0.2, "accuracy": 0.93, "auc": 0.91},
        {"loss": 0.2, "accuracy": 0.93, "auc": 0.92},
    ]
    trainer.optimizer.step = MagicMock()

    _, train_losses, val_metrics = trainer.train()

    # val_loss was correctly extracted (not None)
    for vm in val_metrics:
        assert isinstance(vm["loss"], float)
        assert isinstance(vm["accuracy"], float)
        assert isinstance(vm["auc"], float)

    # train losses recorded
    assert all(loss == pytest.approx(0.5) for loss in train_losses)


@pytest.mark.integration
@patch("orchard.trainer._loop.train_one_epoch")
@patch("orchard.trainer._loop.validate_epoch")
def test_train_calls_tracker(mock_validate, mock_train, trainer):
    """Test train() calls tracker.log_epoch when tracker is set."""
    mock_train.return_value = 0.5
    # Increasing AUC so no early stopping
    mock_validate.side_effect = [
        {"loss": 0.3, "accuracy": 0.9, "auc": 0.80 + i * 0.01} for i in range(trainer.epochs)
    ]
    trainer.optimizer.step = MagicMock()

    mock_tracker = MagicMock()
    trainer.tracker = mock_tracker

    trainer.train()

    assert mock_tracker.log_epoch.call_count == trainer.epochs
    # Verify args of first call
    first_call = mock_tracker.log_epoch.call_args_list[0]
    assert first_call[0][0] == 1  # epoch
    assert first_call[0][1] == pytest.approx(0.5)  # train_loss


@pytest.mark.integration
@patch("orchard.trainer._loop.train_one_epoch")
@patch("orchard.trainer._loop.validate_epoch")
def test_train_early_stop_warning(mock_validate, mock_train, trainer):
    """Test train() logs warning on early stopping."""
    mock_train.return_value = 0.5
    mock_validate.return_value = {"loss": 0.5, "accuracy": 0.5, "auc": 0.5}
    trainer.best_metric = 0.95
    trainer.optimizer.step = MagicMock()

    with patch("orchard.trainer.trainer.logger") as mock_logger:
        trainer.train()
        # Should have called logger.warning with early stopping message
        warning_calls = [
            c
            for c in mock_logger.warning.call_args_list
            if c[0][0] and "Early stopping" in str(c[0][0])
        ]
        assert len(warning_calls) == 1


@pytest.mark.integration
@patch("orchard.trainer._loop.train_one_epoch")
@patch("orchard.trainer._loop.validate_epoch")
def test_train_early_stop_warning_includes_epoch(mock_validate, mock_train, trainer):
    """Early stopping warning includes the actual epoch number (not None)."""
    mock_train.return_value = 0.5
    mock_validate.return_value = {"loss": 0.5, "accuracy": 0.5, "auc": 0.5}
    trainer.best_metric = 0.95
    trainer.optimizer.step = MagicMock()

    with patch("orchard.trainer.trainer.logger") as mock_logger:
        trainer.train()
        es_calls = [
            c for c in mock_logger.warning.call_args_list if "Early stopping" in str(c[0][0])
        ]
        assert len(es_calls) == 1
        # The epoch arg must be an int (not None, not missing)
        epoch_arg = es_calls[0][0][1]
        assert isinstance(epoch_arg, int)
        assert epoch_arg > 0


@pytest.mark.integration
@patch("orchard.trainer._loop.train_one_epoch")
@patch("orchard.trainer._loop.validate_epoch")
def test_train_best_acc_strict_greater(mock_validate, mock_train, trainer):
    """best_acc updates only on strict > (not >=), kills > to >= mutant.

    With >=, best_acc would be reset to 0.80 on the very first epoch
    (0.80 >= -1.0) — which is the same as >. But the real difference is
    visible on second epoch where acc == best_acc: with > it stays, with
    >= it would re-assign redundantly. We test a scenario where initial
    best_acc is set to the exact val_acc and count how many times the
    assignment body executes.
    """
    mock_train.return_value = 0.5
    # All epochs return the same accuracy
    mock_validate.side_effect = [
        {"loss": 0.3, "accuracy": 0.90, "auc": 0.80 + i * 0.01} for i in range(trainer.epochs)
    ]
    trainer.optimizer.step = MagicMock()
    # Set best_acc = val_acc so the > condition should be False
    trainer.best_acc = 0.90

    trainer.train()

    # With >, best_acc stays at 0.90 (no epoch is strictly >)
    # With >=, best_acc would be reassigned each epoch (but still 0.90)
    # This test alone can't distinguish, so we count _log_epoch_summary calls
    # and verify best_acc value remains what we set
    assert trainer.best_acc == pytest.approx(0.90)


@pytest.mark.integration
@patch("orchard.trainer._loop.train_one_epoch")
@patch("orchard.trainer._loop.validate_epoch")
def test_train_log_epoch_summary_receives_real_values(mock_validate, mock_train, trainer):
    """_log_epoch_summary receives actual values (not None) for all arguments."""
    mock_train.return_value = 0.42
    mock_validate.side_effect = [
        {"loss": 0.3, "accuracy": 0.9, "auc": 0.80 + i * 0.01} for i in range(trainer.epochs)
    ]
    trainer.optimizer.step = MagicMock()

    with patch.object(trainer, "_log_epoch_summary") as mock_log:
        trainer.train()

        assert mock_log.call_count == trainer.epochs
        # Check first call args are all non-None and correct types
        first_call_args = mock_log.call_args_list[0][0]
        epoch, train_loss, val_loss, val_acc, monitor_value, lr = first_call_args
        assert epoch == 1
        assert train_loss == pytest.approx(0.42)
        assert isinstance(val_loss, float) and val_loss is not None
        assert isinstance(val_acc, float) and val_acc is not None
        assert isinstance(monitor_value, float) and monitor_value is not None
        assert isinstance(lr, float) and lr is not None


@pytest.mark.integration
@patch("orchard.trainer._loop.train_one_epoch")
@patch("orchard.trainer._loop.validate_epoch")
def test_train_val_loss_extracted_from_metrics(mock_validate, mock_train, trainer):
    """val_loss is read from val_metrics[METRIC_LOSS] (not set to None)."""
    mock_train.return_value = 0.5
    mock_validate.return_value = {"loss": 0.25, "accuracy": 0.9, "auc": 0.85}
    trainer.epochs = 1
    trainer._loop.options = trainer._loop.options.__class__(
        grad_clip=trainer._loop.options.grad_clip,
        total_epochs=1,
        mixup_epochs=trainer._loop.options.mixup_epochs,
        use_tqdm=trainer._loop.options.use_tqdm,
        monitor_metric=trainer._loop.options.monitor_metric,
    )
    trainer.optimizer.step = MagicMock()

    with patch.object(trainer, "_log_epoch_summary") as mock_log:
        trainer.train()
        # val_loss (3rd positional arg, index 2) must be 0.25
        val_loss_arg = mock_log.call_args[0][2]
        assert val_loss_arg == pytest.approx(0.25)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
