"""
Comprehensive Test Suite for ModelTrainer.

Tests cover initialization, training loop, checkpointing,
early stopping, and scheduler interaction.
"""

import tempfile

# Standard Imports
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Third-Party Imports
import torch
import torch.nn as nn

# Internal Imports
from orchard.trainer import ModelTrainer


# FIXTURES
@pytest.fixture
def mock_cfg():
    """Mock Config with training parameters."""
    cfg = MagicMock()
    cfg.training.epochs = 5
    cfg.training.patience = 3
    cfg.training.use_amp = False
    cfg.training.mixup_alpha = 0.0
    cfg.training.cosine_fraction = 0.5
    cfg.training.grad_clip = 1.0
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
    return torch.optim.SGD(simple_model.parameters(), lr=0.01)


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
            cfg=mock_cfg,
            output_path=output_path,
        )

        # Keep tmpdir alive
        trainer._tmpdir = tmpdir

        yield trainer


#                    TESTS: Initialization                                    #
@pytest.mark.unit
def test_trainer_init(trainer):
    """Test ModelTrainer initializes correctly."""
    assert trainer.epochs == 5
    assert trainer.patience == 3
    assert trainer.best_acc == -1.0
    assert trainer.best_auc == -1.0
    assert trainer.epochs_no_improve == 0
    assert len(trainer.train_losses) == 0
    assert len(trainer.val_metrics_history) == 0


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
            cfg=mock_cfg,
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
    cfg.training.cosine_fraction = 0.5
    cfg.training.grad_clip = 0.0

    trainer = ModelTrainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=torch.device("cpu"),
        cfg=cfg,
    )

    assert trainer.scaler is not None


#                    TESTS: Checkpointing                                     #
@pytest.mark.unit
def test_handle_checkpointing_improves(trainer):
    """Test checkpointing saves when AUC improves."""
    val_metrics = {"accuracy": 0.9, "auc": 0.85}

    should_stop = trainer._handle_checkpointing(val_metrics)

    assert trainer.best_auc == 0.85
    assert trainer.epochs_no_improve == 0
    assert should_stop is False
    assert trainer.best_path.exists()


@pytest.mark.unit
def test_handle_checkpointing_no_improve(trainer):
    """Test checkpointing increments patience when no improvement."""
    trainer.best_auc = 0.9

    val_metrics = {"accuracy": 0.8, "auc": 0.85}

    should_stop = trainer._handle_checkpointing(val_metrics)

    assert trainer.best_auc == 0.9
    assert trainer.epochs_no_improve == 1
    assert should_stop is False


@pytest.mark.unit
def test_handle_checkpointing_early_stop(trainer):
    """Test early stopping triggers after patience exhausted."""
    trainer.best_auc = 0.95
    trainer.epochs_no_improve = 2  # patience is 3

    val_metrics = {"accuracy": 0.8, "auc": 0.85}

    should_stop = trainer._handle_checkpointing(val_metrics)

    assert trainer.epochs_no_improve == 3
    assert should_stop is True


#                    TESTS: Scheduler                                         #
@pytest.mark.unit
def test_smart_step_scheduler_reduce_on_plateau(
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
        cfg=mock_cfg,
    )

    trainer._smart_step_scheduler(0.5)


@pytest.mark.unit
def test_smart_step_scheduler_step_lr(trainer):
    """Test scheduler step with StepLR."""
    # Mock optimizer to suppress warning
    trainer.optimizer.step = MagicMock()

    # Call scheduler step
    trainer._smart_step_scheduler(0.5)


#                    TESTS: Load Best Weights                                 #
@pytest.mark.unit
def test_load_best_weights_success(trainer):
    """Test loading best weights from checkpoint."""
    # Save a checkpoint
    torch.save(trainer.model.state_dict(), trainer.best_path)

    # Modify model weights
    with torch.no_grad():
        for param in trainer.model.parameters():
            param.fill_(999.0)

    # Load best weights
    trainer.load_best_weights()

    # Weights should be restored (not all 999)
    first_param = next(trainer.model.parameters())
    assert not torch.all(first_param == 999.0)


@pytest.mark.unit
def test_load_best_weights_file_not_found(trainer):
    """Test load_best_weights raises when file doesn't exist."""
    # Delete checkpoint if exists
    if trainer.best_path.exists():
        trainer.best_path.unlink()

    with pytest.raises(Exception):
        trainer.load_best_weights()


#                    TESTS: Training Loop                                     #
@pytest.mark.integration
@patch("orchard.trainer.trainer.train_one_epoch")
@patch("orchard.trainer.trainer.validate_epoch")
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
@patch("orchard.trainer.trainer.train_one_epoch")
@patch("orchard.trainer.trainer.validate_epoch")
def test_train_early_stopping(mock_validate, mock_train, trainer):
    """Test training stops early when patience is exhausted."""

    # --- 1. Mock training and validation ---
    mock_train.return_value = 0.5
    mock_validate.return_value = {"loss": 0.5, "accuracy": 0.5, "auc": 0.5}

    # --- 2. Force early stopping scenario ---
    trainer.best_auc = 0.95
    trainer.epochs_no_improve = 0

    # --- 3. Mock optimizer step to suppress PyTorch warnings ---
    trainer.optimizer.step = MagicMock()
    trainer.optimizer.step()

    # --- 4. Run trainer ---
    best_path, train_losses, val_metrics = trainer.train()

    # --- 5. Assertions ---
    assert len(train_losses) <= trainer.epochs
    assert trainer.epochs_no_improve >= trainer.cfg.training.patience
    assert best_path.exists()
    for vm in val_metrics:
        assert "loss" in vm
        assert "accuracy" in vm
        assert "auc" in vm


@pytest.mark.integration
@patch("orchard.trainer.trainer.train_one_epoch")
@patch("orchard.trainer.trainer.validate_epoch")
def test_train_mixup_cutoff(
    mock_validate, mock_train, simple_model, mock_loaders, optimizer, scheduler, criterion
):
    """Test MixUp is disabled after cosine_fraction epochs."""
    train_loader, val_loader = mock_loaders

    cfg = MagicMock()
    cfg.training.epochs = 10
    cfg.training.patience = 20
    cfg.training.use_amp = False
    cfg.training.mixup_alpha = 1.0  # Enable MixUp
    cfg.training.cosine_fraction = 0.5  # Disable after epoch 5
    cfg.training.grad_clip = 0.0

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = ModelTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=torch.device("cpu"),
            cfg=cfg,
            output_path=Path(tmpdir) / "best.pth",
        )

        mock_train.return_value = 0.5
        mock_validate.return_value = {"loss": 0.3, "accuracy": 0.9, "auc": 0.85}

        trainer.train()

        calls = mock_train.call_args_list

        # Epoch 1-5: mixup_fn should be set
        assert calls[0].kwargs.get("mixup_fn") is not None

        # Epoch 6-10: mixup_fn should be None
        assert calls[6].kwargs.get("mixup_fn") is None


@pytest.mark.unit
def test_smart_step_scheduler_with_non_gradscaler(
    simple_model, mock_loaders, optimizer, criterion, mock_cfg
):
    """Test scheduler.step() is called when scaler is not GradScaler."""
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
        cfg=mock_cfg,
    )

    # Replace scaler with a non-GradScaler object to trigger the branch
    trainer.scaler = MagicMock()  # Not a torch.amp.GradScaler

    # Mock scheduler.step to track if it's called
    original_step = trainer.scheduler.step
    call_count = [0]

    def mock_step(*args, **kwargs):
        call_count[0] += 1
        return original_step(*args, **kwargs)

    trainer.scheduler.step = mock_step

    # Call the method - should trigger scheduler.step() since scaler is not GradScaler
    trainer._smart_step_scheduler(0.5)

    # Verify scheduler.step was called
    assert call_count[0] == 1, "scheduler.step() should be called when scaler is not GradScaler"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
