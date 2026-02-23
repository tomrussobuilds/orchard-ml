"""
Comprehensive Test Suite for TrialTrainingExecutor.

Tests cover initialization, Optuna integration (reporting/pruning),
scheduler stepping, and error handling during validation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import optuna
import pytest
import torch
import torch.nn as nn

from orchard.optimization import MetricExtractor, TrialTrainingExecutor
from orchard.optimization.objective.training_executor import _FALLBACK_METRICS
from orchard.trainer._scheduling import step_scheduler


# FIXTURES
@pytest.fixture
def mock_cfg():
    """Mock Config specific for Optuna trials."""
    cfg = MagicMock()
    cfg.training.epochs = 5
    cfg.training.use_amp = False
    cfg.training.grad_clip = 1.0
    cfg.training.mixup_alpha = 0
    cfg.training.mixup_epochs = 0
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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0)
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


# TESTS: INITIALIZATION
@pytest.mark.unit
def test_executor_init(executor):
    """Test correctly mapping config to executor attributes."""
    assert executor.epochs == 5
    assert executor.enable_pruning is True
    assert executor.warmup_epochs == 2
    assert executor.scaler is None
    assert executor.mixup_fn is None


@pytest.mark.unit
def test_executor_init_with_mixup():
    """Test executor initializes MixUp when mixup_alpha > 0."""
    cfg = MagicMock()
    cfg.training.epochs = 5
    cfg.training.use_amp = False
    cfg.training.grad_clip = 1.0
    cfg.training.mixup_alpha = 0.2
    cfg.training.mixup_epochs = 3
    cfg.training.seed = 42
    cfg.training.scheduler_type = "step"
    cfg.optuna.enable_pruning = False
    cfg.optuna.pruning_warmup_epochs = 0

    executor = TrialTrainingExecutor(
        model=nn.Linear(10, 2),
        train_loader=MagicMock(),
        val_loader=MagicMock(),
        optimizer=torch.optim.SGD(
            nn.Linear(10, 2).parameters(), lr=0.01, momentum=0.0, weight_decay=0.0
        ),
        scheduler=MagicMock(),
        criterion=nn.CrossEntropyLoss(),
        cfg=cfg,
        device=torch.device("cpu"),
        metric_extractor=MetricExtractor("auc"),
    )

    assert executor.mixup_fn is not None


# TESTS: OPTUNA INTEGRATION
@pytest.mark.unit
def test_should_prune_respects_warmup(executor, mock_trial):
    """Ensure pruning is never triggered before warmup_epochs."""
    executor.enable_pruning = True
    executor.warmup_epochs = 3
    mock_trial.should_prune.return_value = True

    assert executor._should_prune(mock_trial, epoch=1) is False
    assert executor._should_prune(mock_trial, epoch=3) is True


@pytest.mark.unit
def test_should_prune_respects_flag(executor, mock_trial):
    """Ensure pruning is disabled if enable_pruning is False."""
    executor.enable_pruning = False
    executor.warmup_epochs = 0
    mock_trial.should_prune.return_value = True

    assert executor._should_prune(mock_trial, epoch=1) is False


# TESTS: SCHEDULER LOGIC
@pytest.mark.unit
def test_step_scheduler_plateau(executor):
    """Test plateau scheduler receives val_loss."""
    executor.scheduler = MagicMock(spec=torch.optim.lr_scheduler.ReduceLROnPlateau)

    step_scheduler(executor.scheduler, val_loss=0.5)
    executor.scheduler.step.assert_called_once_with(0.5)


@pytest.mark.unit
def test_step_scheduler_standard(executor):
    """Test standard scheduler (StepLR) is called without arguments."""
    executor.scheduler = MagicMock(spec=torch.optim.lr_scheduler.StepLR)

    step_scheduler(executor.scheduler, val_loss=0.5)
    executor.scheduler.step.assert_called_once_with()


@pytest.mark.unit
def test_return_if_scheduler_is_none(executor):
    """Ensures the function exits early when the scheduler is not initialized."""
    result = step_scheduler(None, val_loss=0.5)
    assert result is None


# TESTS: VALIDATION ERROR HANDLING
@pytest.mark.unit
def test_validate_epoch_returns_fallback_on_exception():
    """Test _validate_epoch returns fallback metrics when exception occurs."""
    executor = TrialTrainingExecutor(
        model=nn.Linear(10, 2),
        train_loader=MagicMock(),
        val_loader=MagicMock(),
        optimizer=torch.optim.SGD(
            nn.Linear(10, 2).parameters(), lr=0.01, momentum=0.0, weight_decay=0.0
        ),
        scheduler=MagicMock(),
        criterion=nn.CrossEntropyLoss(),
        cfg=MagicMock(
            training=MagicMock(use_amp=False, epochs=5, mixup_alpha=0),
            optuna=MagicMock(enable_pruning=False, pruning_warmup_epochs=0),
        ),
        device=torch.device("cpu"),
        metric_extractor=MetricExtractor("auc"),
    )

    with patch("orchard.optimization.objective.training_executor.validate_epoch") as mock_validate:
        mock_validate.side_effect = RuntimeError("Validation error")
        result = executor._validate_epoch()

        assert result == _FALLBACK_METRICS


@pytest.mark.unit
def test_validate_epoch_returns_fallback_on_invalid_type():
    """Test _validate_epoch returns fallback when validate_epoch returns invalid type."""
    executor = TrialTrainingExecutor(
        model=nn.Linear(10, 2),
        train_loader=MagicMock(),
        val_loader=MagicMock(),
        optimizer=torch.optim.SGD(
            nn.Linear(10, 2).parameters(), lr=0.01, momentum=0.0, weight_decay=0.0
        ),
        scheduler=MagicMock(),
        criterion=nn.CrossEntropyLoss(),
        cfg=MagicMock(
            training=MagicMock(use_amp=False, epochs=5, mixup_alpha=0),
            optuna=MagicMock(enable_pruning=False, pruning_warmup_epochs=0),
        ),
        device=torch.device("cpu"),
        metric_extractor=MetricExtractor("auc"),
    )

    with patch("orchard.optimization.objective.training_executor.validate_epoch") as mock_validate:
        mock_validate.return_value = "not_a_dict"
        result = executor._validate_epoch()

        assert result == _FALLBACK_METRICS


@pytest.mark.unit
def test_validate_epoch_returns_fallback_on_none():
    """Test _validate_epoch returns fallback when validate_epoch returns None."""
    executor = TrialTrainingExecutor(
        model=nn.Linear(10, 2),
        train_loader=MagicMock(),
        val_loader=MagicMock(),
        optimizer=torch.optim.SGD(
            nn.Linear(10, 2).parameters(), lr=0.01, momentum=0.0, weight_decay=0.0
        ),
        scheduler=MagicMock(),
        criterion=nn.CrossEntropyLoss(),
        cfg=MagicMock(
            training=MagicMock(use_amp=False, epochs=5, mixup_alpha=0),
            optuna=MagicMock(enable_pruning=False, pruning_warmup_epochs=0),
        ),
        device=torch.device("cpu"),
        metric_extractor=MetricExtractor("auc"),
    )

    with patch("orchard.optimization.objective.training_executor.validate_epoch") as mock_validate:
        mock_validate.return_value = None
        result = executor._validate_epoch()

        assert result == _FALLBACK_METRICS


# TESTS: EXECUTE ERROR HANDLING
@pytest.mark.unit
def test_execute_handles_none_validation_result():
    """Test execute returns 0.0 when validation returns None."""
    mock_cfg = MagicMock()
    mock_cfg.training.epochs = 1
    mock_cfg.training.use_amp = False
    mock_cfg.training.grad_clip = 0.0
    mock_cfg.training.mixup_alpha = 0
    mock_cfg.training.scheduler_type = "step"
    mock_cfg.optuna.enable_pruning = False
    mock_cfg.optuna.pruning_warmup_epochs = 0

    executor = TrialTrainingExecutor(
        model=nn.Linear(10, 2),
        train_loader=MagicMock(),
        val_loader=MagicMock(),
        optimizer=torch.optim.SGD(
            nn.Linear(10, 2).parameters(), lr=0.01, momentum=0.0, weight_decay=0.0
        ),
        scheduler=MagicMock(),
        criterion=nn.CrossEntropyLoss(),
        cfg=mock_cfg,
        device=torch.device("cpu"),
        metric_extractor=MetricExtractor("auc"),
    )

    mock_trial = MagicMock()
    mock_trial.number = 1

    with patch.object(executor, "_train_epoch", return_value=0.5):
        with patch.object(executor, "_validate_epoch", return_value=None):
            result = executor.execute(mock_trial)
            assert result == pytest.approx(0.0)


@pytest.mark.unit
def test_execute_handles_invalid_validation_type():
    """Test execute returns 0.0 when validation returns non-dict."""
    mock_cfg = MagicMock()
    mock_cfg.training.epochs = 1
    mock_cfg.training.use_amp = False
    mock_cfg.training.grad_clip = 0.0
    mock_cfg.training.mixup_alpha = 0
    mock_cfg.training.scheduler_type = "step"
    mock_cfg.optuna.enable_pruning = False
    mock_cfg.optuna.pruning_warmup_epochs = 0

    executor = TrialTrainingExecutor(
        model=nn.Linear(10, 2),
        train_loader=MagicMock(),
        val_loader=MagicMock(),
        optimizer=torch.optim.SGD(
            nn.Linear(10, 2).parameters(),
            lr=0.01,
            momentum=0.0,
            weight_decay=0.0,
        ),
        scheduler=MagicMock(),
        criterion=nn.CrossEntropyLoss(),
        cfg=mock_cfg,
        device=torch.device("cpu"),
        metric_extractor=MetricExtractor("auc"),
    )

    mock_trial = MagicMock()
    mock_trial.number = 1

    with patch.object(executor, "_train_epoch", return_value=0.5):
        with patch.object(executor, "_validate_epoch", return_value="invalid"):
            result = executor.execute(mock_trial)
            assert result == pytest.approx(0.0)


# TESTS: FULL EXECUTION LOOP
@pytest.mark.integration
@patch("orchard.optimization.objective.training_executor.train_one_epoch")
@patch("orchard.optimization.objective.training_executor.validate_epoch")
def test_execute_full_loop(mock_val, mock_train, executor, mock_trial):
    """Test a complete successful execution of the trial."""
    mock_train.return_value = 0.4
    mock_val.return_value = {"loss": 0.3, "accuracy": 0.8, "auc": 0.85}

    with patch.object(executor.scheduler, "step"):
        executor.epochs = 2
        best_metric = executor.execute(mock_trial)

    assert best_metric == pytest.approx(0.85)
    assert mock_trial.report.call_count == 2
    mock_trial.report.assert_any_call(0.85, 1)


@pytest.mark.integration
@patch("orchard.optimization.objective.training_executor.train_one_epoch")
@patch("orchard.optimization.objective.training_executor.validate_epoch")
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


@pytest.mark.integration
@patch("orchard.optimization.objective.training_executor.train_one_epoch")
@patch("orchard.optimization.objective.training_executor.validate_epoch")
def test_execute_logs_completion(mock_val, mock_train, executor, mock_trial):
    """Test that trial completion is logged correctly."""
    mock_train.return_value = 0.35
    mock_val.return_value = {"loss": 0.25, "accuracy": 0.9, "auc": 0.92}
    executor.epochs = 1

    def train_side_effect(*args, **kwargs):
        executor.optimizer.step()
        return 0.35

    mock_train.side_effect = train_side_effect

    with patch.object(executor, "_log_trial_complete") as mock_log:
        best_metric = executor.execute(mock_trial)

    mock_log.assert_called_once_with(mock_trial, 0.92, 0.35)
    assert best_metric == pytest.approx(0.92)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
