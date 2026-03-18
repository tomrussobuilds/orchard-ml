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

from orchard.core import OptunaConfig, TrainingConfig
from orchard.optimization import MetricExtractor, TrialTrainingExecutor
from orchard.optimization.objective.training_executor import (
    _GENERIC_FALLBACK,
    _MAX_CONSECUTIVE_VAL_FAILURES,
    TaskAdapters,
)
from orchard.trainer._scheduling import step_scheduler
from tests.conftest import (
    TrainingBundle,
    make_optuna_config,
    make_training_config,
)


# FIXTURES
@pytest.fixture
def training_cfg() -> TrainingConfig:
    """Real TrainingConfig for Optuna trial tests."""
    return make_training_config(epochs=5, scheduler_type="step", monitor_metric="auc")


@pytest.fixture
def optuna_cfg() -> OptunaConfig:
    """Real OptunaConfig for Optuna trial tests."""
    return make_optuna_config(pruning_warmup_epochs=2)


@pytest.fixture
def mock_trial() -> MagicMock:
    """Mock Optuna trial."""
    trial = MagicMock(spec=optuna.Trial)
    trial.number = 42
    trial.should_prune.return_value = False
    return trial


@pytest.fixture
def bundle() -> TrainingBundle:
    """Real PyTorch objects for executor construction."""
    return TrainingBundle()


@pytest.fixture
def executor(
    training_cfg: TrainingConfig, optuna_cfg: OptunaConfig, bundle: TrainingBundle
) -> TrialTrainingExecutor:
    """TrialTrainingExecutor instance with real components."""
    return TrialTrainingExecutor(
        model=bundle.model,
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        optimizer=bundle.optimizer,
        scheduler=bundle.scheduler,
        criterion=bundle.criterion,
        training=training_cfg,
        optuna=optuna_cfg,
        log_interval=5,
        device=bundle.device,
        metric_extractor=MetricExtractor(metric_name="auc"),
    )


# TESTS: INITIALIZATION
@pytest.mark.unit
def test_executor_init(executor: TrialTrainingExecutor) -> None:
    """Test correctly mapping config to executor attributes."""
    assert executor.epochs == 5
    assert executor.enable_pruning is True
    assert executor.warmup_epochs == 2
    assert executor.scaler is None
    assert executor.mixup_fn is None


@pytest.mark.unit
def test_executor_init_with_mixup() -> None:
    """Test executor initializes MixUp when mixup_alpha > 0."""
    training = make_training_config(epochs=5, mixup_alpha=0.2, mixup_epochs=3)
    optuna_cfg = make_optuna_config(enable_pruning=False, pruning_warmup_epochs=0)
    b = TrainingBundle()

    executor = TrialTrainingExecutor(
        model=b.model,
        train_loader=b.train_loader,
        val_loader=b.val_loader,
        optimizer=b.optimizer,
        scheduler=b.scheduler,
        criterion=b.criterion,
        training=training,
        optuna=optuna_cfg,
        log_interval=5,
        device=b.device,
        metric_extractor=MetricExtractor("auc"),
    )

    assert executor.mixup_fn is not None


# TESTS: OPTUNA INTEGRATION
@pytest.mark.unit
def test_should_prune_respects_warmup(
    executor: TrialTrainingExecutor, mock_trial: MagicMock
) -> None:
    """Ensure pruning is never triggered before warmup_epochs."""
    executor.enable_pruning = True
    executor.warmup_epochs = 3
    mock_trial.should_prune.return_value = True

    assert executor._should_prune(mock_trial, epoch=1) is False
    assert executor._should_prune(mock_trial, epoch=3) is True


@pytest.mark.unit
def test_should_prune_respects_flag(executor: TrialTrainingExecutor, mock_trial: MagicMock) -> None:
    """Ensure pruning is disabled if enable_pruning is False."""
    executor.enable_pruning = False
    executor.warmup_epochs = 0
    mock_trial.should_prune.return_value = True

    assert executor._should_prune(mock_trial, epoch=1) is False


# TESTS: SCHEDULER LOGIC
@pytest.mark.unit
def test_step_scheduler_plateau(executor: TrialTrainingExecutor) -> None:
    """Test plateau scheduler receives monitor_value."""
    executor.scheduler = MagicMock(spec=torch.optim.lr_scheduler.ReduceLROnPlateau)

    step_scheduler(executor.scheduler, monitor_value=0.5)
    executor.scheduler.step.assert_called_once_with(0.5)


@pytest.mark.unit
def test_step_scheduler_standard(executor: TrialTrainingExecutor) -> None:
    """Test standard scheduler (StepLR) is called without arguments."""
    executor.scheduler = MagicMock(spec=torch.optim.lr_scheduler.StepLR)

    step_scheduler(executor.scheduler, monitor_value=0.5)
    executor.scheduler.step.assert_called_once_with()


@pytest.mark.unit
def test_return_if_scheduler_is_none(executor: TrialTrainingExecutor) -> None:
    """Ensures the function exits early when the scheduler is not initialized."""
    step_scheduler(None, monitor_value=0.5)  # should be a no-op


# TESTS: VALIDATION ERROR HANDLING
@pytest.mark.unit
def test_validate_epoch_returns_fallback_on_exception() -> None:
    """Test _validate_epoch returns fallback metrics when exception occurs."""
    training = make_training_config(epochs=5)
    optuna_cfg = make_optuna_config(enable_pruning=False, pruning_warmup_epochs=0)
    b = TrainingBundle()

    executor = TrialTrainingExecutor(
        model=b.model,
        train_loader=b.train_loader,
        val_loader=b.val_loader,
        optimizer=b.optimizer,
        scheduler=b.scheduler,
        criterion=b.criterion,
        training=training,
        optuna=optuna_cfg,
        log_interval=5,
        device=b.device,
        metric_extractor=MetricExtractor("auc"),
    )

    with patch("orchard.optimization.objective.training_executor.validate_epoch") as mock_validate:
        mock_validate.side_effect = RuntimeError("Validation error")
        result = executor._validate_epoch()

        # Generic fallback (loss=999.0) augmented with monitor_metric (auc=0.0)
        assert result == {"loss": 999.0, "auc": 0.0}


@pytest.mark.unit
def test_validate_epoch_fallback_with_explicit_metrics() -> None:
    """Fallback uses explicit fallback_metrics when monitor_metric is present.

    Kills mutant ``self._fallback_metrics = resolved_fallback`` → ``None``.
    """
    training = make_training_config(epochs=5, monitor_metric="auc")
    optuna_cfg = make_optuna_config(enable_pruning=False, pruning_warmup_epochs=0)
    b = TrainingBundle()

    explicit_fallback = {"loss": 0.0, "auc": -1.0}

    executor = TrialTrainingExecutor(
        model=b.model,
        train_loader=b.train_loader,
        val_loader=b.val_loader,
        optimizer=b.optimizer,
        scheduler=b.scheduler,
        criterion=b.criterion,
        training=training,
        optuna=optuna_cfg,
        log_interval=5,
        device=b.device,
        metric_extractor=MetricExtractor("auc"),
        task_adapters=TaskAdapters(fallback_metrics=explicit_fallback),
    )

    with patch("orchard.optimization.objective.training_executor.validate_epoch") as mock_validate:
        mock_validate.side_effect = RuntimeError("Validation error")
        result = executor._validate_epoch()

        assert result == {"loss": 0.0, "auc": -1.0}


# TESTS: FULL EXECUTION LOOP
@pytest.mark.integration
@patch("orchard.trainer._loop.train_one_epoch")
@patch("orchard.optimization.objective.training_executor.validate_epoch")
def test_execute_full_loop(
    mock_val: MagicMock,
    mock_train: MagicMock,
    executor: TrialTrainingExecutor,
    mock_trial: MagicMock,
) -> None:
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
def test_execute_pruning_raises(
    mock_val: MagicMock,
    mock_train: MagicMock,
    executor: TrialTrainingExecutor,
    mock_trial: MagicMock,
) -> None:
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
def test_execute_logs_completion(
    mock_val: MagicMock,
    mock_train: MagicMock,
    executor: TrialTrainingExecutor,
    mock_trial: MagicMock,
) -> None:
    """Test that trial completion is logged correctly."""
    mock_train.return_value = 0.35
    mock_val.return_value = {"loss": 0.25, "accuracy": 0.9, "auc": 0.92}
    executor.epochs = 1

    def train_side_effect(*args: object, **kwargs: object) -> float:
        executor.optimizer.step()
        return 0.35

    mock_train.side_effect = train_side_effect

    with patch.object(executor, "_log_trial_complete") as mock_log:
        best_metric = executor.execute(mock_trial)

    mock_log.assert_called_once_with(mock_trial, 0.92, 0.35)
    assert best_metric == pytest.approx(0.92)


@pytest.mark.unit
def test_validate_epoch_reraises_after_consecutive_failures() -> None:
    """Test _validate_epoch raises RuntimeError after 3 consecutive failures."""
    training = make_training_config(epochs=5)
    optuna_cfg = make_optuna_config(enable_pruning=False, pruning_warmup_epochs=0)
    b = TrainingBundle()

    executor = TrialTrainingExecutor(
        model=b.model,
        train_loader=b.train_loader,
        val_loader=b.val_loader,
        optimizer=b.optimizer,
        scheduler=b.scheduler,
        criterion=b.criterion,
        training=training,
        optuna=optuna_cfg,
        log_interval=5,
        device=b.device,
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
def test_max_consecutive_val_failures_constant() -> None:
    """Assert _MAX_CONSECUTIVE_VAL_FAILURES is exactly 3."""
    assert _MAX_CONSECUTIVE_VAL_FAILURES == 3


@pytest.mark.unit
def test_generic_fallback_exact_values() -> None:
    """Assert _GENERIC_FALLBACK has exact expected values."""
    assert _GENERIC_FALLBACK["loss"] == 999.0
    assert len(_GENERIC_FALLBACK) == 1


@pytest.mark.unit
def test_executor_init_all_attributes(executor: TrialTrainingExecutor) -> None:
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
def test_executor_loop_use_tqdm_false(executor: TrialTrainingExecutor) -> None:
    """Assert the _loop options have use_tqdm=False."""
    assert executor._loop.options.use_tqdm is False


@pytest.mark.unit
@patch("orchard.trainer._loop.train_one_epoch")
@patch("orchard.optimization.objective.training_executor.validate_epoch")
def test_nan_metric_skips_trial_report(
    mock_val: MagicMock,
    mock_train: MagicMock,
    executor: TrialTrainingExecutor,
    mock_trial: MagicMock,
) -> None:
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
def test_non_nan_metric_calls_trial_report(
    mock_val: MagicMock,
    mock_train: MagicMock,
    executor: TrialTrainingExecutor,
    mock_trial: MagicMock,
) -> None:
    """Verify trial.report IS called when metric is valid."""
    mock_train.return_value = 0.4
    mock_val.return_value = {"loss": 0.3, "accuracy": 0.8, "auc": 0.85}

    executor.epochs = 1
    executor.enable_pruning = False

    with patch.object(executor.scheduler, "step"):
        executor.execute(mock_trial)

    mock_trial.report.assert_called_once_with(0.85, 1)


@pytest.mark.unit
def test_consecutive_val_failures_reset_on_success() -> None:
    """Assert _consecutive_val_failures resets to 0 after a successful validation."""
    training = make_training_config(epochs=5)
    optuna_cfg = make_optuna_config(enable_pruning=False, pruning_warmup_epochs=0)
    b = TrainingBundle()

    executor = TrialTrainingExecutor(
        model=b.model,
        train_loader=b.train_loader,
        val_loader=b.val_loader,
        optimizer=b.optimizer,
        scheduler=b.scheduler,
        criterion=b.criterion,
        training=training,
        optuna=optuna_cfg,
        log_interval=5,
        device=b.device,
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
def test_execute_returns_best_metric_with_increasing(
    mock_val: MagicMock,
    mock_train: MagicMock,
    executor: TrialTrainingExecutor,
    mock_trial: MagicMock,
) -> None:
    """Verify execute returns the best (highest) metric across epochs."""
    call_count = [0]

    def val_side_effect(*args: object, **kwargs: object) -> dict[str, float]:
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
def test_scheduler_receives_monitor_metric_value(
    mock_val: MagicMock,
    mock_train: MagicMock,
    executor: TrialTrainingExecutor,
    mock_trial: MagicMock,
) -> None:
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
def test_executor_stores_model_identity(executor: TrialTrainingExecutor) -> None:
    """Assert model attribute is the exact object passed, not None."""
    assert executor.model is not None
    assert isinstance(executor.model, nn.Linear)


@pytest.mark.unit
def test_executor_stores_loader_identities() -> None:
    """Assert train_loader and val_loader are stored as-is (not None)."""
    b = TrainingBundle()
    training = make_training_config(epochs=5, scheduler_type="step", monitor_metric="auc")
    optuna_cfg = make_optuna_config(enable_pruning=False, pruning_warmup_epochs=0)

    executor = TrialTrainingExecutor(
        model=b.model,
        train_loader=b.train_loader,
        val_loader=b.val_loader,
        optimizer=b.optimizer,
        scheduler=b.scheduler,
        criterion=b.criterion,
        training=training,
        optuna=optuna_cfg,
        log_interval=5,
        device=b.device,
        metric_extractor=MetricExtractor("auc"),
    )

    assert executor.train_loader is b.train_loader
    assert executor.val_loader is b.val_loader
    assert executor.criterion is not None
    assert executor.model is b.model


@pytest.mark.unit
def test_validate_epoch_passes_correct_kwargs() -> None:
    """Assert _validate_epoch passes self.model, self.val_loader etc. to validate_epoch."""
    b = TrainingBundle()
    training = make_training_config(epochs=5, scheduler_type="step", monitor_metric="auc")
    optuna_cfg = make_optuna_config(enable_pruning=False, pruning_warmup_epochs=0)

    executor = TrialTrainingExecutor(
        model=b.model,
        train_loader=b.train_loader,
        val_loader=b.val_loader,
        optimizer=b.optimizer,
        scheduler=b.scheduler,
        criterion=b.criterion,
        training=training,
        optuna=optuna_cfg,
        log_interval=5,
        device=b.device,
        metric_extractor=MetricExtractor("auc"),
    )

    with patch("orchard.optimization.objective.training_executor.validate_epoch") as mock_val:
        mock_val.return_value = {"loss": 0.3, "accuracy": 0.8, "auc": 0.85, "f1": 0.82}
        executor._validate_epoch()

    mock_val.assert_called_once_with(
        model=b.model,
        val_loader=b.val_loader,
        criterion=b.criterion,
        device=b.device,
    )


@pytest.mark.unit
def test_validate_epoch_uses_injected_adapter() -> None:
    """Assert _validate_epoch delegates to injected TaskValidationMetrics adapter."""
    b = TrainingBundle()
    training = make_training_config(epochs=5, scheduler_type="step", monitor_metric="auc")
    optuna_cfg = make_optuna_config(enable_pruning=False, pruning_warmup_epochs=0)

    mock_adapter = MagicMock()
    mock_adapter.compute_validation_metrics.return_value = {
        "loss": 0.2,
        "accuracy": 0.95,
        "auc": 0.98,
        "f1": 0.93,
    }

    executor = TrialTrainingExecutor(
        model=b.model,
        train_loader=b.train_loader,
        val_loader=b.val_loader,
        optimizer=b.optimizer,
        scheduler=b.scheduler,
        criterion=b.criterion,
        training=training,
        optuna=optuna_cfg,
        log_interval=5,
        device=b.device,
        metric_extractor=MetricExtractor("auc"),
        task_adapters=TaskAdapters(validation_metrics=mock_adapter),
    )

    result = executor._validate_epoch()

    mock_adapter.compute_validation_metrics.assert_called_once_with(
        model=b.model,
        val_loader=b.val_loader,
        criterion=b.criterion,
        device=b.device,
    )
    assert result["auc"] == pytest.approx(0.98)


@pytest.mark.unit
def test_validate_epoch_skips_validate_epoch_when_adapter_injected() -> None:
    """Assert _validate_epoch does NOT call validate_epoch when adapter is set."""
    b = TrainingBundle()
    training = make_training_config(epochs=5, scheduler_type="step", monitor_metric="auc")
    optuna_cfg = make_optuna_config(enable_pruning=False, pruning_warmup_epochs=0)

    mock_adapter = MagicMock()
    mock_adapter.compute_validation_metrics.return_value = {
        "loss": 0.2,
        "accuracy": 0.95,
        "auc": 0.98,
        "f1": 0.93,
    }

    executor = TrialTrainingExecutor(
        model=b.model,
        train_loader=b.train_loader,
        val_loader=b.val_loader,
        optimizer=b.optimizer,
        scheduler=b.scheduler,
        criterion=b.criterion,
        training=training,
        optuna=optuna_cfg,
        log_interval=5,
        device=b.device,
        metric_extractor=MetricExtractor("auc"),
        task_adapters=TaskAdapters(validation_metrics=mock_adapter),
    )

    with patch("orchard.optimization.objective.training_executor.validate_epoch") as mock_val:
        executor._validate_epoch()
        mock_val.assert_not_called()


@pytest.mark.unit
def test_executor_stores_validation_metrics() -> None:
    """Assert TrialTrainingExecutor.__init__ stores validation_metrics."""
    b = TrainingBundle()
    training = make_training_config(epochs=5, scheduler_type="step", monitor_metric="auc")
    optuna_cfg = make_optuna_config(enable_pruning=False, pruning_warmup_epochs=0)
    mock_adapter = MagicMock()

    executor = TrialTrainingExecutor(
        model=b.model,
        train_loader=b.train_loader,
        val_loader=b.val_loader,
        optimizer=b.optimizer,
        scheduler=b.scheduler,
        criterion=b.criterion,
        training=training,
        optuna=optuna_cfg,
        log_interval=5,
        device=b.device,
        metric_extractor=MetricExtractor("auc"),
        task_adapters=TaskAdapters(validation_metrics=mock_adapter),
    )

    assert executor._validation_metrics is mock_adapter


@pytest.mark.unit
@patch("orchard.trainer._loop.train_one_epoch")
@patch("orchard.optimization.objective.training_executor.validate_epoch")
def test_log_interval_controls_logging(
    mock_val: MagicMock, mock_train: MagicMock, mock_trial: MagicMock
) -> None:
    """Verify log_interval logic: epoch % log_interval == 0 OR epoch == epochs."""
    b = TrainingBundle()
    training = make_training_config(epochs=5, scheduler_type="step", monitor_metric="auc")
    optuna_cfg = make_optuna_config(enable_pruning=False, pruning_warmup_epochs=0)

    executor = TrialTrainingExecutor(
        model=b.model,
        train_loader=b.train_loader,
        val_loader=b.val_loader,
        optimizer=b.optimizer,
        scheduler=b.scheduler,
        criterion=b.criterion,
        training=training,
        optuna=optuna_cfg,
        log_interval=2,
        device=b.device,
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
def test_log_interval_only_last_epoch(
    mock_val: MagicMock, mock_train: MagicMock, mock_trial: MagicMock
) -> None:
    """With log_interval=10 and epochs=3, only last epoch logs (epoch==epochs)."""
    b = TrainingBundle()
    training = make_training_config(epochs=3, scheduler_type="step", monitor_metric="auc")
    optuna_cfg = make_optuna_config(enable_pruning=False, pruning_warmup_epochs=0)

    executor = TrialTrainingExecutor(
        model=b.model,
        train_loader=b.train_loader,
        val_loader=b.val_loader,
        optimizer=b.optimizer,
        scheduler=b.scheduler,
        criterion=b.criterion,
        training=training,
        optuna=optuna_cfg,
        log_interval=10,
        device=b.device,
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
def test_loop_options_grad_clip(executor: TrialTrainingExecutor) -> None:
    """Assert LoopOptions.grad_clip matches training config."""
    assert executor._loop.options.grad_clip == 1.0


@pytest.mark.unit
def test_loop_options_monitor_metric(executor: TrialTrainingExecutor) -> None:
    """Assert LoopOptions.monitor_metric matches training config."""
    assert executor._loop.options.monitor_metric == "auc"


@pytest.mark.unit
def test_loop_options_total_epochs(executor: TrialTrainingExecutor) -> None:
    """Assert LoopOptions.total_epochs matches executor.epochs."""
    assert executor._loop.options.total_epochs == executor.epochs


@pytest.mark.unit
def test_executor_forwards_training_step_to_loop(
    training_cfg: TrainingConfig, optuna_cfg: OptunaConfig, bundle: TrainingBundle
) -> None:
    """TrialTrainingExecutor passes training_step to the inner TrainingLoop."""
    mock_step = MagicMock()
    executor = TrialTrainingExecutor(
        model=bundle.model,
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        optimizer=bundle.optimizer,
        scheduler=bundle.scheduler,
        criterion=bundle.criterion,
        training=training_cfg,
        optuna=optuna_cfg,
        log_interval=5,
        device=bundle.device,
        metric_extractor=MetricExtractor(metric_name="auc"),
        task_adapters=TaskAdapters(training_step=mock_step),
    )
    assert executor._loop._training_step is mock_step


@pytest.mark.unit
def test_executor_training_step_defaults_to_none(executor: TrialTrainingExecutor) -> None:
    """TrialTrainingExecutor._loop._training_step is None when not provided."""
    assert executor._loop._training_step is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
