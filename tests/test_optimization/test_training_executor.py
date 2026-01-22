"""
Comprehensive Test Suite for TrialTrainingExecutor.

Tests cover initialization, Optuna integration (reporting/pruning),
scheduler stepping, and error handling during validation.
"""

# =========================================================================== #
#                         Standard Imports                                    #
# =========================================================================== #
from unittest.mock import MagicMock, patch

# =========================================================================== #
#                         Third-Party Imports                                 #
# =========================================================================== #
import optuna
import pytest
import torch
import torch.nn as nn

# =========================================================================== #
#                         Internal Imports                                    #
# =========================================================================== #
from orchard.optimization import MetricExtractor, TrialTrainingExecutor

# =========================================================================== #
#                    FIXTURES                                                 #
# =========================================================================== #


@pytest.fixture
def mock_cfg():
    """Mock Config specific for Optuna trials."""
    cfg = MagicMock()
    cfg.training.epochs = 5
    cfg.training.use_amp = False
    cfg.training.grad_clip = 1.0
    cfg.training.scheduler_type = "step"

    cfg.optuna.enable_pruning = True
    cfg.optuna.pruning_warmup_epochs = 2
    return cfg


@pytest.fixture
def mock_trial():
    """Mock Optuna trial."""
    trial = MagicMock(spec=optuna.Trial)
    trial.number = 42
    trial.should_prune.return_value = False
    return trial


@pytest.fixture
def executor(mock_cfg):
    """TrialTrainingExecutor instance with mocked components."""
    model = nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    metric_extractor = MetricExtractor(metric_name="auc")

    return TrialTrainingExecutor(
        model=model,
        train_loader=MagicMock(),
        val_loader=MagicMock(),
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=nn.CrossEntropyLoss(),
        cfg=mock_cfg,
        device=torch.device("cpu"),
        metric_extractor=metric_extractor,
    )


# =========================================================================== #
#                    TESTS: Initialization                                    #
# =========================================================================== #


@pytest.mark.unit
def test_executor_init(executor, mock_cfg):
    """Test correctly mapping config to executor attributes."""
    assert executor.epochs == 5
    assert executor.enable_pruning is True
    assert executor.warmup_epochs == 2
    assert executor.scaler is None


# =========================================================================== #
#                    TESTS: Optuna Integration                                #
# =========================================================================== #


@pytest.mark.unit
def test_should_prune_respects_warmup(executor, mock_trial):
    """Ensure pruning is never triggered before warmup_epochs."""
    executor.enable_pruning = True
    executor.warmup_epochs = 3
    mock_trial.should_prune.return_value = True

    assert executor._should_prune(mock_trial, epoch=1) is False
    # Epoch 3: At/Above warmup, SHOULD prune
    assert executor._should_prune(mock_trial, epoch=3) is True


@pytest.mark.unit
def test_should_prune_respects_flag(executor, mock_trial):
    """Ensure pruning is disabled if enable_pruning is False."""
    executor.enable_pruning = False
    executor.warmup_epochs = 0
    mock_trial.should_prune.return_value = True

    assert executor._should_prune(mock_trial, epoch=1) is False


# =========================================================================== #
#                    TESTS: Scheduler Logic                                   #
# =========================================================================== #


@pytest.mark.unit
def test_step_scheduler_plateau(executor):
    """Test plateau scheduler receives val_loss."""
    executor.cfg.training.scheduler_type = "plateau"
    executor.scheduler = MagicMock()

    executor._step_scheduler(val_loss=0.5)
    executor.scheduler.step.assert_called_once_with(0.5)


@pytest.mark.unit
def test_step_scheduler_standard(executor):
    """Test standard scheduler (StepLR) is called without arguments."""
    executor.cfg.training.scheduler_type = "step"
    executor.scheduler = MagicMock()

    executor._step_scheduler(val_loss=0.5)
    executor.scheduler.step.assert_called_once_with()


# =========================================================================== #
#                    TESTS: Full Execution Loop                               #
# =========================================================================== #


@pytest.mark.integration
@patch("orchard.optimization.objective.TrialTrainingExecutor._train_epoch")
@patch("orchard.optimization.objective.TrialTrainingExecutor._validate_epoch")
def test_execute_full_loop(mock_val, mock_train, executor, mock_trial):
    """Test a complete successful execution of the trial."""
    mock_train.return_value = 0.4
    mock_val.return_value = {"loss": 0.3, "accuracy": 0.8, "auc": 0.85}

    if hasattr(executor, "scheduler") and executor.scheduler is not None:
        with patch.object(executor.scheduler, "step"):
            executor.epochs = 2
            best_metric = executor.execute(mock_trial)
    else:
        executor.epochs = 2
        best_metric = executor.execute(mock_trial)

    assert best_metric == 0.85
    assert mock_trial.report.call_count == 2
    mock_trial.report.assert_any_call(0.85, 1)


@pytest.mark.integration
@patch("orchard.optimization.objective.TrialTrainingExecutor._train_epoch")
@patch("orchard.optimization.objective.TrialTrainingExecutor._validate_epoch")
def test_execute_pruning_raises(mock_val, mock_train, executor, mock_trial):
    """Test that TrialPruned is raised and execution stops."""
    mock_train.return_value = 0.4
    mock_val.return_value = {"loss": 0.3, "auc": 0.5}

    executor.warmup_epochs = 1
    executor.enable_pruning = True

    mock_trial.should_prune.return_value = True

    with pytest.raises(optuna.TrialPruned):
        executor.execute(mock_trial)

    assert mock_train.call_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
