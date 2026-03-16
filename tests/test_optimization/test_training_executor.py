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
from orchard.optimization.objective.training_executor import (
    _FALLBACK_METRICS,
    _MAX_CONSECUTIVE_VAL_FAILURES,
)
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
    cfg.training.monitor_metric = "auc"

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
        training=mock_cfg.training,
        optuna=mock_cfg.optuna,
        log_interval=5,
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
        training=cfg.training,
        optuna=cfg.optuna,
        log_interval=5,
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
    """Test plateau scheduler receives monitor_value."""
    executor.scheduler = MagicMock(spec=torch.optim.lr_scheduler.ReduceLROnPlateau)

    step_scheduler(executor.scheduler, monitor_value=0.5)
    executor.scheduler.step.assert_called_once_with(0.5)


@pytest.mark.unit
def test_step_scheduler_standard(executor):
    """Test standard scheduler (StepLR) is called without arguments."""
    executor.scheduler = MagicMock(spec=torch.optim.lr_scheduler.StepLR)

    step_scheduler(executor.scheduler, monitor_value=0.5)
    executor.scheduler.step.assert_called_once_with()


@pytest.mark.unit
def test_return_if_scheduler_is_none(executor):
    """Ensures the function exits early when the scheduler is not initialized."""
    result = step_scheduler(None, monitor_value=0.5)
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
        training=MagicMock(use_amp=False, epochs=5, mixup_alpha=0),
        optuna=MagicMock(enable_pruning=False, pruning_warmup_epochs=0),
        log_interval=5,
        device=torch.device("cpu"),
        metric_extractor=MetricExtractor("auc"),
    )

    with patch("orchard.optimization.objective.training_executor.validate_epoch") as mock_validate:
        mock_validate.side_effect = RuntimeError("Validation error")
        result = executor._validate_epoch()

        assert result == _FALLBACK_METRICS


# TESTS: FULL EXECUTION LOOP
@pytest.mark.integration
@patch("orchard.trainer._loop.train_one_epoch")
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
@patch("orchard.trainer._loop.train_one_epoch")
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
@patch("orchard.trainer._loop.train_one_epoch")
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


@pytest.mark.unit
def test_validate_epoch_reraises_after_consecutive_failures():
    """Test _validate_epoch raises RuntimeError after 3 consecutive failures."""
    executor = TrialTrainingExecutor(
        model=nn.Linear(10, 2),
        train_loader=MagicMock(),
        val_loader=MagicMock(),
        optimizer=torch.optim.SGD(
            nn.Linear(10, 2).parameters(), lr=0.01, momentum=0.0, weight_decay=0.0
        ),
        scheduler=MagicMock(),
        criterion=nn.CrossEntropyLoss(),
        training=MagicMock(use_amp=False, epochs=5, mixup_alpha=0),
        optuna=MagicMock(enable_pruning=False, pruning_warmup_epochs=0),
        log_interval=5,
        device=torch.device("cpu"),
        metric_extractor=MetricExtractor("auc"),
    )

    with patch("orchard.optimization.objective.training_executor.validate_epoch") as mock_validate:
        mock_validate.side_effect = RuntimeError("Validation error")

        # First two failures: returns fallback
        executor._validate_epoch()
        executor._validate_epoch()

        # Third failure: re-raises
        with pytest.raises(RuntimeError, match="3 consecutive times"):
            executor._validate_epoch()


# ---------------------------------------------------------------------------
# Mutation-killing tests: init attrs, constants, NaN, val failures reset
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_max_consecutive_val_failures_constant():
    """Assert _MAX_CONSECUTIVE_VAL_FAILURES is exactly 3."""
    assert _MAX_CONSECUTIVE_VAL_FAILURES == 3


@pytest.mark.unit
def test_fallback_metrics_exact_values():
    """Assert _FALLBACK_METRICS has exact expected values."""
    assert _FALLBACK_METRICS["loss"] == 999.0
    assert _FALLBACK_METRICS["accuracy"] == 0.0
    assert _FALLBACK_METRICS["auc"] == 0.0
    assert _FALLBACK_METRICS["f1"] == 0.0


@pytest.mark.unit
def test_executor_init_all_attributes(executor, mock_cfg):
    """Assert all init attributes are stored correctly."""
    assert executor.epochs == 5
    assert executor.enable_pruning is True
    assert executor.warmup_epochs == 2
    assert executor.monitor_metric == "auc"
    assert executor.log_interval == 5
    assert executor._consecutive_val_failures == 0
    assert executor.device == torch.device("cpu")
    assert executor.scaler is None
    assert executor.mixup_fn is None


@pytest.mark.unit
def test_executor_loop_use_tqdm_false(executor):
    """Assert the _loop options have use_tqdm=False."""
    assert executor._loop.options.use_tqdm is False


@pytest.mark.unit
@patch("orchard.trainer._loop.train_one_epoch")
@patch("orchard.optimization.objective.training_executor.validate_epoch")
def test_nan_metric_skips_trial_report(mock_val, mock_train, executor, mock_trial):
    """Verify trial.report is NOT called when metric is NaN."""
    mock_train.return_value = 0.4
    mock_val.return_value = {"loss": 0.3, "accuracy": 0.8, "auc": float("nan")}

    executor.epochs = 1
    executor.enable_pruning = False

    with patch.object(executor.scheduler, "step"):
        executor.execute(mock_trial)

    mock_trial.report.assert_not_called()


@pytest.mark.unit
@patch("orchard.trainer._loop.train_one_epoch")
@patch("orchard.optimization.objective.training_executor.validate_epoch")
def test_non_nan_metric_calls_trial_report(mock_val, mock_train, executor, mock_trial):
    """Verify trial.report IS called when metric is valid."""
    mock_train.return_value = 0.4
    mock_val.return_value = {"loss": 0.3, "accuracy": 0.8, "auc": 0.85}

    executor.epochs = 1
    executor.enable_pruning = False

    with patch.object(executor.scheduler, "step"):
        executor.execute(mock_trial)

    mock_trial.report.assert_called_once_with(0.85, 1)


@pytest.mark.unit
def test_consecutive_val_failures_reset_on_success():
    """Assert _consecutive_val_failures resets to 0 after a successful validation."""
    executor = TrialTrainingExecutor(
        model=nn.Linear(10, 2),
        train_loader=MagicMock(),
        val_loader=MagicMock(),
        optimizer=torch.optim.SGD(
            nn.Linear(10, 2).parameters(), lr=0.01, momentum=0.0, weight_decay=0.0
        ),
        scheduler=MagicMock(),
        criterion=nn.CrossEntropyLoss(),
        training=MagicMock(use_amp=False, epochs=5, mixup_alpha=0),
        optuna=MagicMock(enable_pruning=False, pruning_warmup_epochs=0),
        log_interval=5,
        device=torch.device("cpu"),
        metric_extractor=MetricExtractor("auc"),
    )

    # Simulate prior failures
    executor._consecutive_val_failures = 2

    with patch("orchard.optimization.objective.training_executor.validate_epoch") as mock_val:
        mock_val.return_value = {"loss": 0.3, "accuracy": 0.8, "auc": 0.85, "f1": 0.82}
        executor._validate_epoch()

    assert executor._consecutive_val_failures == 0


@pytest.mark.integration
@patch("orchard.trainer._loop.train_one_epoch")
@patch("orchard.optimization.objective.training_executor.validate_epoch")
def test_execute_returns_best_metric_with_increasing(mock_val, mock_train, executor, mock_trial):
    """Verify execute returns the best (highest) metric across epochs."""
    call_count = [0]

    def val_side_effect(*args, **kwargs):
        call_count[0] += 1
        auc = 0.7 + call_count[0] * 0.05  # 0.75, 0.80, 0.85
        return {"loss": 0.3, "accuracy": 0.8, "auc": auc}

    mock_train.return_value = 0.4
    mock_val.side_effect = val_side_effect
    executor.epochs = 3
    executor.enable_pruning = False

    with patch.object(executor.scheduler, "step"):
        best = executor.execute(mock_trial)

    assert best == pytest.approx(0.85)


@pytest.mark.integration
@patch("orchard.trainer._loop.train_one_epoch")
@patch("orchard.optimization.objective.training_executor.validate_epoch")
def test_scheduler_receives_monitor_metric_value(mock_val, mock_train, executor, mock_trial):
    """Verify step_scheduler receives val_metrics[monitor_metric]."""
    mock_train.return_value = 0.4
    mock_val.return_value = {"loss": 0.3, "accuracy": 0.8, "auc": 0.85}
    executor.epochs = 1
    executor.enable_pruning = False

    with patch("orchard.optimization.objective.training_executor.step_scheduler") as mock_step:
        executor.execute(mock_trial)

    mock_step.assert_called_once_with(executor.scheduler, 0.85)


# ---------------------------------------------------------------------------
# Mutation-killing: attribute identity, _validate_epoch kwargs, log interval
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_executor_stores_model_identity(executor):
    """Assert model attribute is the exact object passed, not None."""
    assert executor.model is not None
    assert isinstance(executor.model, nn.Linear)


@pytest.mark.unit
def test_executor_stores_loader_identities():
    """Assert train_loader and val_loader are stored as-is (not None)."""
    train_loader = MagicMock()
    val_loader = MagicMock()
    model = nn.Linear(10, 2)

    executor = TrialTrainingExecutor(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0),
        scheduler=MagicMock(),
        criterion=nn.CrossEntropyLoss(),
        training=MagicMock(
            use_amp=False,
            epochs=5,
            mixup_alpha=0,
            grad_clip=1.0,
            mixup_epochs=0,
            scheduler_type="step",
            monitor_metric="auc",
        ),
        optuna=MagicMock(enable_pruning=False, pruning_warmup_epochs=0),
        log_interval=5,
        device=torch.device("cpu"),
        metric_extractor=MetricExtractor("auc"),
    )

    assert executor.train_loader is train_loader
    assert executor.val_loader is val_loader
    assert executor.criterion is not None
    assert executor.model is model


@pytest.mark.unit
def test_validate_epoch_passes_correct_kwargs():
    """Assert _validate_epoch passes self.model, self.val_loader etc. to validate_epoch."""
    model = nn.Linear(10, 2)
    val_loader = MagicMock()
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    executor = TrialTrainingExecutor(
        model=model,
        train_loader=MagicMock(),
        val_loader=val_loader,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0),
        scheduler=MagicMock(),
        criterion=criterion,
        training=MagicMock(
            use_amp=False,
            epochs=5,
            mixup_alpha=0,
            grad_clip=1.0,
            mixup_epochs=0,
            scheduler_type="step",
            monitor_metric="auc",
        ),
        optuna=MagicMock(enable_pruning=False, pruning_warmup_epochs=0),
        log_interval=5,
        device=device,
        metric_extractor=MetricExtractor("auc"),
    )

    with patch("orchard.optimization.objective.training_executor.validate_epoch") as mock_val:
        mock_val.return_value = {"loss": 0.3, "accuracy": 0.8, "auc": 0.85, "f1": 0.82}
        executor._validate_epoch()

    mock_val.assert_called_once_with(
        model=model,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
    )


@pytest.mark.unit
def test_validate_epoch_uses_injected_adapter():
    """Assert _validate_epoch delegates to injected TaskValidationMetrics adapter."""
    model = nn.Linear(10, 2)
    val_loader = MagicMock()
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    mock_adapter = MagicMock()
    mock_adapter.compute_validation_metrics.return_value = {
        "loss": 0.2,
        "accuracy": 0.95,
        "auc": 0.98,
        "f1": 0.93,
    }

    executor = TrialTrainingExecutor(
        model=model,
        train_loader=MagicMock(),
        val_loader=val_loader,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0),
        scheduler=MagicMock(),
        criterion=criterion,
        training=MagicMock(
            use_amp=False,
            epochs=5,
            mixup_alpha=0,
            grad_clip=1.0,
            mixup_epochs=0,
            scheduler_type="step",
            monitor_metric="auc",
        ),
        optuna=MagicMock(enable_pruning=False, pruning_warmup_epochs=0),
        log_interval=5,
        device=device,
        metric_extractor=MetricExtractor("auc"),
        validation_metrics=mock_adapter,
    )

    result = executor._validate_epoch()

    mock_adapter.compute_validation_metrics.assert_called_once_with(
        model=model,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
    )
    assert result["auc"] == pytest.approx(0.98)


@pytest.mark.unit
def test_validate_epoch_skips_validate_epoch_when_adapter_injected():
    """Assert _validate_epoch does NOT call validate_epoch when adapter is set."""
    model = nn.Linear(10, 2)
    mock_adapter = MagicMock()
    mock_adapter.compute_validation_metrics.return_value = {
        "loss": 0.2,
        "accuracy": 0.95,
        "auc": 0.98,
        "f1": 0.93,
    }

    executor = TrialTrainingExecutor(
        model=model,
        train_loader=MagicMock(),
        val_loader=MagicMock(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0),
        scheduler=MagicMock(),
        criterion=nn.CrossEntropyLoss(),
        training=MagicMock(
            use_amp=False,
            epochs=5,
            mixup_alpha=0,
            grad_clip=1.0,
            mixup_epochs=0,
            scheduler_type="step",
            monitor_metric="auc",
        ),
        optuna=MagicMock(enable_pruning=False, pruning_warmup_epochs=0),
        log_interval=5,
        device=torch.device("cpu"),
        metric_extractor=MetricExtractor("auc"),
        validation_metrics=mock_adapter,
    )

    with patch("orchard.optimization.objective.training_executor.validate_epoch") as mock_val:
        executor._validate_epoch()
        mock_val.assert_not_called()


@pytest.mark.unit
def test_executor_stores_validation_metrics():
    """Assert TrialTrainingExecutor.__init__ stores validation_metrics."""
    model = nn.Linear(10, 2)
    mock_adapter = MagicMock()

    executor = TrialTrainingExecutor(
        model=model,
        train_loader=MagicMock(),
        val_loader=MagicMock(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0),
        scheduler=MagicMock(),
        criterion=nn.CrossEntropyLoss(),
        training=MagicMock(
            use_amp=False,
            epochs=5,
            mixup_alpha=0,
            grad_clip=1.0,
            mixup_epochs=0,
            scheduler_type="step",
            monitor_metric="auc",
        ),
        optuna=MagicMock(enable_pruning=False, pruning_warmup_epochs=0),
        log_interval=5,
        device=torch.device("cpu"),
        metric_extractor=MetricExtractor("auc"),
        validation_metrics=mock_adapter,
    )

    assert executor._validation_metrics is mock_adapter


@pytest.mark.unit
@patch("orchard.trainer._loop.train_one_epoch")
@patch("orchard.optimization.objective.training_executor.validate_epoch")
def test_log_interval_controls_logging(mock_val, mock_train, mock_trial):
    """Verify log_interval logic: epoch % log_interval == 0 OR epoch == epochs."""
    model = nn.Linear(10, 2)
    executor = TrialTrainingExecutor(
        model=model,
        train_loader=MagicMock(),
        val_loader=MagicMock(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0),
        scheduler=MagicMock(),
        criterion=nn.CrossEntropyLoss(),
        training=MagicMock(
            use_amp=False,
            epochs=5,
            mixup_alpha=0,
            grad_clip=1.0,
            mixup_epochs=0,
            scheduler_type="step",
            monitor_metric="auc",
        ),
        optuna=MagicMock(enable_pruning=False, pruning_warmup_epochs=0),
        log_interval=2,
        device=torch.device("cpu"),
        metric_extractor=MetricExtractor("auc"),
    )

    mock_train.return_value = 0.4
    mock_val.return_value = {"loss": 0.3, "accuracy": 0.8, "auc": 0.85}
    mock_trial.should_prune.return_value = False

    with patch("orchard.optimization.objective.training_executor.logger") as mock_logger:
        with patch.object(executor.scheduler, "step"):
            executor.execute(mock_trial)

    # Epochs 1-5 with log_interval=2:
    # epoch 2: 2%2==0 → log
    # epoch 4: 4%2==0 → log
    # epoch 5: 5==5 → log
    # Total: 3 epoch-progress calls
    info_calls = mock_logger.info.call_args_list
    progress_calls = [
        c for c in info_calls if c[0] and isinstance(c[0][0], str) and "E%d/%d" in c[0][0]
    ]
    assert len(progress_calls) == 3


@pytest.mark.unit
@patch("orchard.trainer._loop.train_one_epoch")
@patch("orchard.optimization.objective.training_executor.validate_epoch")
def test_log_interval_only_last_epoch(mock_val, mock_train, mock_trial):
    """With log_interval=10 and epochs=3, only last epoch logs (epoch==epochs)."""
    model = nn.Linear(10, 2)
    executor = TrialTrainingExecutor(
        model=model,
        train_loader=MagicMock(),
        val_loader=MagicMock(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0),
        scheduler=MagicMock(),
        criterion=nn.CrossEntropyLoss(),
        training=MagicMock(
            use_amp=False,
            epochs=3,
            mixup_alpha=0,
            grad_clip=1.0,
            mixup_epochs=0,
            scheduler_type="step",
            monitor_metric="auc",
        ),
        optuna=MagicMock(enable_pruning=False, pruning_warmup_epochs=0),
        log_interval=10,
        device=torch.device("cpu"),
        metric_extractor=MetricExtractor("auc"),
    )

    mock_train.return_value = 0.4
    mock_val.return_value = {"loss": 0.3, "accuracy": 0.8, "auc": 0.85}
    mock_trial.should_prune.return_value = False

    with patch("orchard.optimization.objective.training_executor.logger") as mock_logger:
        with patch.object(executor.scheduler, "step"):
            executor.execute(mock_trial)

    info_calls = mock_logger.info.call_args_list
    progress_calls = [
        c for c in info_calls if c[0] and isinstance(c[0][0], str) and "E%d/%d" in c[0][0]
    ]
    assert len(progress_calls) == 1


@pytest.mark.unit
def test_loop_options_grad_clip(executor):
    """Assert LoopOptions.grad_clip matches training config."""
    assert executor._loop.options.grad_clip == 1.0


@pytest.mark.unit
def test_loop_options_monitor_metric(executor):
    """Assert LoopOptions.monitor_metric matches training config."""
    assert executor._loop.options.monitor_metric == "auc"


@pytest.mark.unit
def test_loop_options_total_epochs(executor):
    """Assert LoopOptions.total_epochs matches executor.epochs."""
    assert executor._loop.options.total_epochs == executor.epochs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
