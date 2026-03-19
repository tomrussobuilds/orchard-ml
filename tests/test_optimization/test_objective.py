"""
Unit tests for the refactored OptunaObjective.

These tests validate the behavior of the Optuna objective and its
supporting components using dependency injection, enabling high
coverage through isolated and deterministic unit tests.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import optuna
import pytest
import torch

from orchard.optimization.objective import (
    MetricExtractor,
    OptunaObjective,
    TrialConfigBuilder,
    TrialTrainingExecutor,
)
from tests.conftest import make_optuna_config, make_training_config


# TRIAL CONFIG BUILDER TESTS
@pytest.mark.unit
def test_config_builder_preserves_metadata() -> None:
    """Test TrialConfigBuilder preserves dataset metadata."""
    mock_cfg = MagicMock()
    mock_cfg.model_dump.return_value = {
        "dataset": {"resolution": 28},
        "training": {},
        "architecture": {},
        "augmentation": {},
    }
    mock_cfg.dataset.resolution = 28
    mock_cfg.dataset._ensure_metadata = MagicMock()
    mock_cfg.dataset._ensure_metadata.name = "bloodmnist"
    mock_cfg.optuna.epochs = 20

    builder = TrialConfigBuilder(mock_cfg)

    assert builder.base_metadata == mock_cfg.dataset._ensure_metadata
    assert builder.optuna_epochs == 20


@pytest.mark.unit
def test_config_builder_applies_param_overrides() -> None:
    """Test TrialConfigBuilder applies parameter overrides correctly."""
    mock_cfg = MagicMock()
    config_dict: dict[str, Any] = {
        "dataset": {"resolution": 28, "metadata": None},
        "training": {"learning_rate": 0.001, "epochs": 60},
        "architecture": {"dropout": 0.0},
        "augmentation": {"rotation_angle": 0},
    }
    mock_cfg.model_dump.return_value = config_dict.copy()
    mock_cfg.dataset.resolution = 28
    mock_cfg.dataset._ensure_metadata = MagicMock()
    mock_cfg.optuna.epochs = 20

    builder = TrialConfigBuilder(mock_cfg)

    trial_params = {
        "learning_rate": 0.0001,
        "dropout": 0.3,
        "rotation_angle": 15,
    }

    test_dict: dict[str, Any] = config_dict.copy()
    test_dict["dataset"]["metadata"] = builder.base_metadata
    test_dict["training"]["epochs"] = builder.optuna_epochs
    builder._apply_param_overrides(test_dict, trial_params)

    assert test_dict["training"]["learning_rate"] == pytest.approx(0.0001)
    assert test_dict["architecture"]["dropout"] == pytest.approx(0.3)
    assert test_dict["augmentation"]["rotation_angle"] == 15
    assert test_dict["training"]["epochs"] == 20


@pytest.mark.unit
def test_config_builder_handles_model_name() -> None:
    """Test TrialConfigBuilder maps model_name to architecture.name."""
    mock_cfg = MagicMock()
    config_dict: dict[str, Any] = {
        "dataset": {"resolution": 224, "metadata": None},
        "training": {"epochs": 60},
        "architecture": {"name": "efficientnet_b0", "dropout": 0.3},
        "augmentation": {},
    }
    mock_cfg.model_dump.return_value = config_dict.copy()
    mock_cfg.dataset.resolution = 224
    mock_cfg.dataset._ensure_metadata = MagicMock()
    mock_cfg.optuna.epochs = 15

    builder = TrialConfigBuilder(mock_cfg)

    trial_params = {"model_name": "vit_tiny"}

    test_dict: dict[str, Any] = config_dict.copy()
    test_dict["dataset"]["metadata"] = builder.base_metadata
    test_dict["training"]["epochs"] = builder.optuna_epochs
    builder._apply_param_overrides(test_dict, trial_params)

    assert test_dict["architecture"]["name"] == "vit_tiny"


@pytest.mark.unit
def test_config_builder_handles_weight_variant() -> None:
    """Test TrialConfigBuilder applies weight_variant."""
    mock_cfg = MagicMock()
    config_dict: dict[str, Any] = {
        "dataset": {"resolution": 224, "metadata": None},
        "training": {"epochs": 60},
        "architecture": {"name": "vit_tiny", "weight_variant": None},
        "augmentation": {},
    }
    mock_cfg.model_dump.return_value = config_dict.copy()
    mock_cfg.dataset.resolution = 224
    mock_cfg.dataset._ensure_metadata = MagicMock()
    mock_cfg.optuna.epochs = 15

    builder = TrialConfigBuilder(mock_cfg)

    trial_params = {"weight_variant": "vit_tiny_patch16_224.augreg_in21k_ft_in1k"}

    test_dict: dict[str, Any] = config_dict.copy()
    test_dict["dataset"]["metadata"] = builder.base_metadata
    test_dict["training"]["epochs"] = builder.optuna_epochs
    builder._apply_param_overrides(test_dict, trial_params)

    assert (
        test_dict["architecture"]["weight_variant"] == "vit_tiny_patch16_224.augreg_in21k_ft_in1k"
    )


@pytest.mark.unit
def test_config_builder_maps_criterion_and_focal_gamma() -> None:
    """Test TrialConfigBuilder maps criterion_type and focal_gamma to training."""
    mock_cfg = MagicMock()
    config_dict: dict[str, Any] = {
        "dataset": {"resolution": 28, "metadata": None},
        "training": {"epochs": 60, "criterion_type": "cross_entropy", "focal_gamma": 2.0},
        "architecture": {"name": "mini_cnn", "dropout": 0.3},
        "augmentation": {},
    }
    mock_cfg.model_dump.return_value = config_dict.copy()
    mock_cfg.dataset.resolution = 28
    mock_cfg.dataset._ensure_metadata = MagicMock()
    mock_cfg.optuna.epochs = 15

    builder = TrialConfigBuilder(mock_cfg)

    trial_params = {"criterion_type": "focal", "focal_gamma": 3.5}

    test_dict: dict[str, Any] = config_dict.copy()
    test_dict["dataset"]["metadata"] = builder.base_metadata
    test_dict["training"]["epochs"] = builder.optuna_epochs
    builder._apply_param_overrides(test_dict, trial_params)

    assert test_dict["training"]["criterion_type"] == "focal"
    assert test_dict["training"]["focal_gamma"] == pytest.approx(3.5)


@pytest.mark.unit
def test_config_builder_skips_none_weight_variant() -> None:
    """Test TrialConfigBuilder skips None weight_variant (for non-ViT models)."""
    mock_cfg = MagicMock()
    config_dict: dict[str, Any] = {
        "dataset": {"resolution": 224, "metadata": None},
        "training": {"epochs": 60},
        "architecture": {"name": "efficientnet_b0", "weight_variant": "original_value"},
        "augmentation": {},
    }
    mock_cfg.model_dump.return_value = config_dict.copy()
    mock_cfg.dataset.resolution = 224
    mock_cfg.dataset._ensure_metadata = MagicMock()
    mock_cfg.optuna.epochs = 15

    builder = TrialConfigBuilder(mock_cfg)

    # Simulate None from search space for non-ViT model
    trial_params = {"weight_variant": None}

    test_dict: dict[str, Any] = config_dict.copy()
    test_dict["dataset"]["metadata"] = builder.base_metadata
    test_dict["training"]["epochs"] = builder.optuna_epochs
    builder._apply_param_overrides(test_dict, trial_params)

    # Should preserve original value and not overwrite with None
    assert test_dict["architecture"]["weight_variant"] == "original_value"


# METRIC EXTRACTOR TESTS
@pytest.mark.unit
def test_metric_extractor_extracts_correct_metric() -> None:
    """Test MetricExtractor extracts specified metric."""
    extractor = MetricExtractor(metric_name="auc")

    val_metrics = {"loss": 0.5, "accuracy": 0.85, "auc": 0.92}

    result = extractor.extract(val_metrics)

    assert result == pytest.approx(0.92)


@pytest.mark.unit
def test_metric_extractor_raises_on_missing_metric() -> None:
    """Test MetricExtractor raises KeyError for missing metric."""
    extractor = MetricExtractor(metric_name="f1")

    val_metrics = {"loss": 0.5, "accuracy": 0.85, "auc": 0.92}

    with pytest.raises(KeyError, match="f1"):
        extractor.extract(val_metrics)


@pytest.mark.unit
def test_metric_extractor_tracks_best() -> None:
    """Test MetricExtractor tracks best metric."""
    extractor = MetricExtractor(metric_name="auc")

    best1 = extractor.update_best(0.80)
    assert best1 == pytest.approx(0.80)
    assert extractor.best_metric == pytest.approx(0.80)

    best2 = extractor.update_best(0.90)
    assert best2 == pytest.approx(0.90)
    assert extractor.best_metric == pytest.approx(0.90)

    best3 = extractor.update_best(0.85)
    assert best3 == pytest.approx(0.90)
    assert extractor.best_metric == pytest.approx(0.90)


@pytest.mark.unit
def test_metric_extractor_ignores_nan() -> None:
    """NaN values should not poison the best-metric state."""
    extractor = MetricExtractor(metric_name="auc")

    extractor.update_best(0.85)
    result = extractor.update_best(float("nan"))

    assert result == pytest.approx(0.85)
    assert extractor.best_metric == pytest.approx(0.85)


@pytest.mark.unit
def test_metric_extractor_reset() -> None:
    """Test MetricExtractor.reset() clears best metric between trials."""
    extractor = MetricExtractor(metric_name="auc")

    extractor.update_best(0.95)
    assert extractor.best_metric == pytest.approx(0.95)

    extractor.reset()
    assert extractor.best_metric == -float("inf")

    best = extractor.update_best(0.70)
    assert best == pytest.approx(0.70)


# TRAINING EXECUTOR TESTS
@pytest.mark.unit
def test_training_executor_should_prune_warmup() -> None:
    """Test TrialTrainingExecutor respects warmup period."""
    mock_trial = MagicMock()
    mock_trial.should_prune.return_value = True

    training = make_training_config(epochs=30)
    optuna_cfg = make_optuna_config(enable_pruning=True, pruning_warmup_epochs=10)

    executor = TrialTrainingExecutor(
        model=MagicMock(),
        train_loader=MagicMock(),
        val_loader=MagicMock(),
        optimizer=MagicMock(),
        scheduler=MagicMock(),
        criterion=MagicMock(),
        training=training,
        optuna=optuna_cfg,
        log_interval=5,
        device=torch.device("cpu"),
        metric_extractor=MetricExtractor("auc"),
    )

    assert executor._should_prune(mock_trial, epoch=5) is False

    assert executor._should_prune(mock_trial, epoch=15) is True


@pytest.mark.unit
def test_training_executor_disabled_pruning() -> None:
    """Test TrialTrainingExecutor with pruning disabled."""
    mock_trial = MagicMock()
    mock_trial.should_prune.return_value = True

    training = make_training_config(epochs=30, grad_clip=0.0)
    optuna_cfg = make_optuna_config(enable_pruning=False, pruning_warmup_epochs=10)

    assert optuna_cfg.enable_pruning is False

    executor = TrialTrainingExecutor(
        model=MagicMock(),
        train_loader=MagicMock(),
        val_loader=MagicMock(),
        optimizer=MagicMock(),
        scheduler=MagicMock(),
        criterion=MagicMock(),
        training=training,
        optuna=optuna_cfg,
        log_interval=5,
        device=torch.device("cpu"),
        metric_extractor=MetricExtractor("auc"),
    )

    result1 = executor._should_prune(mock_trial, epoch=5)
    result2 = executor._should_prune(mock_trial, epoch=15)

    assert result1 is False, f"Expected False but got {result1}"
    assert result2 is False, f"Expected False but got {result2}"


@pytest.mark.unit
def test_training_executor_validate_epoch_error_handling() -> None:
    """Test TrialTrainingExecutor handles validation errors."""
    training = make_training_config(epochs=30)
    optuna_cfg = make_optuna_config(enable_pruning=True, pruning_warmup_epochs=5)

    executor = TrialTrainingExecutor(
        model=MagicMock(),
        train_loader=MagicMock(),
        val_loader=MagicMock(),
        optimizer=MagicMock(),
        scheduler=MagicMock(),
        criterion=MagicMock(),
        training=training,
        optuna=optuna_cfg,
        log_interval=5,
        device=torch.device("cpu"),
        metric_extractor=MetricExtractor("auc"),
    )

    # Patch where validate_epoch is used, not where it's defined
    with patch(
        "orchard.optimization.objective.training_executor.validate_epoch",
        side_effect=RuntimeError("Validation failed"),
    ):
        result = executor._validate_epoch()

    # Generic fallback (loss=999.0) augmented with monitor_metric (auc=0.0)
    assert result == {"loss": 999.0, "auc": 0.0}


# OPTUNA OBJECTIVE TESTS
@pytest.mark.unit
def test_optuna_objective_init_with_defaults() -> None:
    """Test OptunaObjective initializes with dependency injection."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 20
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.dataset._ensure_metadata = MagicMock()
    mock_cfg.dataset._ensure_metadata.name = "bloodmnist"

    search_space = {"learning_rate": MagicMock()}
    device = torch.device("cpu")

    mock_dataset = MagicMock()
    mock_dataset.path = "/fake/path"
    mock_dataset_loader = MagicMock(return_value=mock_dataset)

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space=search_space,
        device=device,
        dataset_loader=mock_dataset_loader,
    )

    mock_dataset_loader.assert_called_once_with(mock_cfg.dataset._ensure_metadata)
    assert objective.dataset_data == mock_dataset
    assert objective._dataset_loader == mock_dataset_loader


@pytest.mark.unit
def test_optuna_objective_uses_injected_dependencies() -> None:
    """Test OptunaObjective uses all injected dependencies."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 20
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    search_space: dict[str, Any] = {}
    device = torch.device("cpu")

    mock_dataset_loader = MagicMock(return_value=MagicMock())
    mock_dataloader_factory = MagicMock()
    mock_model_factory = MagicMock()

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space=search_space,
        device=device,
        dataset_loader=mock_dataset_loader,
        dataloader_factory=mock_dataloader_factory,
        model_factory=mock_model_factory,
    )

    assert objective._dataset_loader == mock_dataset_loader
    assert objective._dataloader_factory == mock_dataloader_factory
    assert objective._model_factory == mock_model_factory

    mock_dataset_loader.assert_called_once_with(mock_cfg.dataset._ensure_metadata)


@pytest.mark.unit
def test_optuna_objective_sample_params_dict() -> None:
    """Test OptunaObjective samples params from dict search space."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 20
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.dataset._ensure_metadata = MagicMock()
    mock_cfg.training.momentum = 0.9
    mock_cfg.training.mixup_alpha = 0.0
    mock_suggest = MagicMock(return_value=0.001)
    search_space = {"learning_rate": mock_suggest}

    mock_loader = MagicMock(return_value=MagicMock())

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space=search_space,
        device=torch.device("cpu"),
        dataset_loader=mock_loader,
    )

    mock_trial = MagicMock()
    params = objective._sample_params(mock_trial)

    mock_suggest.assert_called_once_with(mock_trial)
    assert params == {"learning_rate": 0.001}


@pytest.mark.unit
def test_optuna_objective_sample_params_object() -> None:
    """Test OptunaObjective samples params from object with sample_params."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 20
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    mock_search_space = MagicMock()
    mock_search_space.sample_params.return_value = {"lr": 0.01, "dropout": 0.3}

    mock_dataset_loader = MagicMock(return_value=MagicMock())

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space=mock_search_space,
        device=torch.device("cpu"),
        dataset_loader=mock_dataset_loader,
    )

    mock_trial = MagicMock()
    params = objective._sample_params(mock_trial)

    mock_search_space.sample_params.assert_called_once_with(mock_trial)
    assert params == {"lr": 0.01, "dropout": 0.3}

    mock_dataset_loader.assert_called_once_with(mock_cfg.dataset._ensure_metadata)


# OPTUNA OBJECTIVE: __CALL__ METHOD
@pytest.mark.unit
def test_optuna_objective_call_with_pruning() -> None:
    """Test OptunaObjective.__call__ handles pruning correctly."""

    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    search_space: dict[str, Any] = {}
    mock_dataset_loader = MagicMock(return_value=MagicMock())
    mock_dataloader_factory = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock()))
    mock_model_factory = MagicMock(return_value=MagicMock())

    mock_trial = MagicMock()
    mock_trial.number = 3

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space=search_space,
        device=torch.device("cpu"),
        dataset_loader=mock_dataset_loader,
        dataloader_factory=mock_dataloader_factory,
        model_factory=mock_model_factory,
    )

    _mock_trial_cfg = MagicMock()
    _mock_trial_cfg.training.weighted_loss = False
    objective.config_builder.build = MagicMock(return_value=_mock_trial_cfg)  # type: ignore

    with (
        patch("orchard.optimization.objective.objective.get_optimizer"),
        patch("orchard.optimization.objective.objective.get_scheduler"),
        patch("orchard.optimization.objective.objective.get_task"),
        patch("orchard.optimization.objective.objective.log_trial_start"),
    ):
        with patch(
            "orchard.optimization.objective.objective.TrialTrainingExecutor"
        ) as mock_executor_cls:
            mock_executor_instance = MagicMock()
            mock_executor_instance.execute.side_effect = optuna.TrialPruned()
            mock_executor_cls.return_value = mock_executor_instance

            with pytest.raises(optuna.TrialPruned):
                objective(mock_trial)


@pytest.mark.unit
def test_optuna_objective_call_cleanup_on_success() -> None:
    """Test OptunaObjective.__call__ calls cleanup after successful trial."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    search_space: dict[str, Any] = {}
    mock_dataset_loader = MagicMock(return_value=MagicMock())
    mock_dataloader_factory = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock()))
    mock_model_factory = MagicMock(return_value=MagicMock())

    mock_trial = MagicMock()
    mock_trial.number = 1

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space=search_space,
        device=torch.device("cpu"),
        dataset_loader=mock_dataset_loader,
        dataloader_factory=mock_dataloader_factory,
        model_factory=mock_model_factory,
    )

    objective._cleanup = MagicMock()  # type: ignore

    _mock_trial_cfg = MagicMock()
    _mock_trial_cfg.training.weighted_loss = False
    objective.config_builder.build = MagicMock(return_value=_mock_trial_cfg)  # type: ignore

    with (
        patch("orchard.optimization.objective.objective.get_optimizer"),
        patch("orchard.optimization.objective.objective.get_scheduler"),
        patch("orchard.optimization.objective.objective.get_task"),
        patch("orchard.optimization.objective.objective.log_trial_start"),
    ):
        with patch(
            "orchard.optimization.objective.objective.TrialTrainingExecutor"
        ) as mock_executor_cls:
            mock_executor_instance = MagicMock()
            mock_executor_instance.execute.return_value = 0.88
            mock_executor_cls.return_value = mock_executor_instance

            result = objective(mock_trial)

            objective._cleanup.assert_called_once()
            assert result == pytest.approx(0.88)


@pytest.mark.unit
def test_optuna_objective_call_returns_worst_metric_on_failure() -> None:
    """Test OptunaObjective.__call__ returns worst metric on exception (maximize)."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.training.monitor_direction = "maximize"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    search_space: dict[str, Any] = {}
    mock_dataset_loader = MagicMock(return_value=MagicMock())
    mock_dataloader_factory = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock()))
    mock_model_factory = MagicMock(return_value=MagicMock())

    mock_trial = MagicMock()
    mock_trial.number = 2

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space=search_space,
        device=torch.device("cpu"),
        dataset_loader=mock_dataset_loader,
        dataloader_factory=mock_dataloader_factory,
        model_factory=mock_model_factory,
    )

    objective._cleanup = MagicMock()  # type: ignore

    _mock_trial_cfg = MagicMock()
    _mock_trial_cfg.training.weighted_loss = False
    objective.config_builder.build = MagicMock(return_value=_mock_trial_cfg)  # type: ignore

    with (
        patch("orchard.optimization.objective.objective.get_optimizer"),
        patch("orchard.optimization.objective.objective.get_scheduler"),
        patch("orchard.optimization.objective.objective.get_task"),
        patch("orchard.optimization.objective.objective.log_trial_start"),
    ):
        with patch(
            "orchard.optimization.objective.objective.TrialTrainingExecutor"
        ) as mock_executor_cls:
            mock_executor_instance = MagicMock()
            mock_executor_instance.execute.side_effect = RuntimeError("Training failed")
            mock_executor_cls.return_value = mock_executor_instance

            result = objective(mock_trial)

            assert result == -float("inf")
            objective._cleanup.assert_called_once()


@pytest.mark.unit
def test_optuna_objective_call_returns_inf_on_failure_minimize() -> None:
    """Test OptunaObjective.__call__ returns inf on exception (minimize)."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.training.monitor_direction = "minimize"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    search_space: dict[str, Any] = {}
    mock_dataset_loader = MagicMock(return_value=MagicMock())
    mock_dataloader_factory = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock()))
    mock_model_factory = MagicMock(return_value=MagicMock())

    mock_trial = MagicMock()
    mock_trial.number = 3

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space=search_space,
        device=torch.device("cpu"),
        dataset_loader=mock_dataset_loader,
        dataloader_factory=mock_dataloader_factory,
        model_factory=mock_model_factory,
    )

    objective._cleanup = MagicMock()  # type: ignore

    _mock_trial_cfg = MagicMock()
    _mock_trial_cfg.training.weighted_loss = False
    objective.config_builder.build = MagicMock(return_value=_mock_trial_cfg)  # type: ignore

    with (
        patch("orchard.optimization.objective.objective.get_optimizer"),
        patch("orchard.optimization.objective.objective.get_scheduler"),
        patch("orchard.optimization.objective.objective.get_task"),
        patch("orchard.optimization.objective.objective.log_trial_start"),
    ):
        with patch(
            "orchard.optimization.objective.objective.TrialTrainingExecutor"
        ) as mock_executor_cls:
            mock_executor_instance = MagicMock()
            mock_executor_instance.execute.side_effect = RuntimeError("OOM")
            mock_executor_cls.return_value = mock_executor_instance

            result = objective(mock_trial)

            assert result == float("inf")
            objective._cleanup.assert_called_once()


@pytest.mark.unit
def test_optuna_objective_failed_trial_logs_worst_metric_to_tracker() -> None:
    """Test tracker receives worst_metric when trial fails before validation."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.training.monitor_direction = "maximize"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    search_space: dict[str, Any] = {}
    mock_tracker = MagicMock()

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space=search_space,
        device=torch.device("cpu"),
        dataset_loader=MagicMock(return_value=MagicMock()),
        dataloader_factory=MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock())),
        model_factory=MagicMock(return_value=MagicMock()),
        tracker=mock_tracker,
    )
    objective._cleanup = MagicMock()  # type: ignore

    _mock_trial_cfg = MagicMock()
    _mock_trial_cfg.training.weighted_loss = False
    objective.config_builder.build = MagicMock(return_value=_mock_trial_cfg)  # type: ignore

    mock_trial = MagicMock()
    mock_trial.number = 5

    with (
        patch("orchard.optimization.objective.objective.get_optimizer"),
        patch("orchard.optimization.objective.objective.get_scheduler"),
        patch("orchard.optimization.objective.objective.get_task"),
        patch("orchard.optimization.objective.objective.log_trial_start"),
    ):
        with patch(
            "orchard.optimization.objective.objective.TrialTrainingExecutor"
        ) as mock_executor_cls:
            mock_executor_cls.return_value.execute.side_effect = RuntimeError("OOM")

            result = objective(mock_trial)

    assert result == -float("inf")
    mock_tracker.start_optuna_trial.assert_called_once()
    mock_tracker.end_optuna_trial.assert_called_once_with(-float("inf"))


@pytest.mark.unit
def test_optuna_objective_call_reraises_trial_pruned() -> None:
    """Test OptunaObjective.__call__ re-raises TrialPruned (not caught)."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.training.monitor_direction = "maximize"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    search_space: dict[str, Any] = {}
    mock_dataset_loader = MagicMock(return_value=MagicMock())
    mock_dataloader_factory = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock()))
    mock_model_factory = MagicMock(return_value=MagicMock())

    mock_trial = MagicMock()
    mock_trial.number = 4

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space=search_space,
        device=torch.device("cpu"),
        dataset_loader=mock_dataset_loader,
        dataloader_factory=mock_dataloader_factory,
        model_factory=mock_model_factory,
    )

    objective._cleanup = MagicMock()  # type: ignore

    _mock_trial_cfg = MagicMock()
    _mock_trial_cfg.training.weighted_loss = False
    objective.config_builder.build = MagicMock(return_value=_mock_trial_cfg)  # type: ignore

    with (
        patch("orchard.optimization.objective.objective.get_optimizer"),
        patch("orchard.optimization.objective.objective.get_scheduler"),
        patch("orchard.optimization.objective.objective.get_task"),
        patch("orchard.optimization.objective.objective.log_trial_start"),
    ):
        with patch(
            "orchard.optimization.objective.objective.TrialTrainingExecutor"
        ) as mock_executor_cls:
            mock_executor_instance = MagicMock()
            mock_executor_instance.execute.side_effect = optuna.TrialPruned()
            mock_executor_cls.return_value = mock_executor_instance

            with pytest.raises(optuna.TrialPruned):
                objective(mock_trial)

            objective._cleanup.assert_called_once()


@pytest.mark.unit
def test_optuna_objective_call_builds_trial_config() -> None:
    """Test OptunaObjective.__call__ builds trial-specific config."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    mock_suggest_lr = MagicMock(return_value=0.005)
    mock_suggest_dropout = MagicMock(return_value=0.4)
    search_space = {
        "learning_rate": mock_suggest_lr,
        "dropout": mock_suggest_dropout,
    }

    mock_dataset_loader = MagicMock(return_value=MagicMock())
    mock_dataloader_factory = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock()))
    mock_model_factory = MagicMock(return_value=MagicMock())

    mock_trial = MagicMock()
    mock_trial.number = 7

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space=search_space,
        device=torch.device("cpu"),
        dataset_loader=mock_dataset_loader,
        dataloader_factory=mock_dataloader_factory,
        model_factory=mock_model_factory,
    )

    mock_trial_cfg = MagicMock()
    objective.config_builder.build = MagicMock(return_value=mock_trial_cfg)  # type: ignore

    with (
        patch("orchard.optimization.objective.objective.get_optimizer"),
        patch("orchard.optimization.objective.objective.get_scheduler"),
        patch("orchard.optimization.objective.objective.get_task"),
        patch("orchard.optimization.objective.objective.log_trial_start"),
    ):
        with patch(
            "orchard.optimization.objective.objective.TrialTrainingExecutor"
        ) as mock_executor_cls:
            mock_executor_instance = MagicMock()
            mock_executor_instance.execute.return_value = 0.92
            mock_executor_cls.return_value = mock_executor_instance

            objective(mock_trial)

            objective.config_builder.build.assert_called_once_with(
                {"learning_rate": 0.005, "dropout": 0.4}
            )

            mock_dataloader_factory.assert_called_once_with(
                objective.dataset_data,
                mock_trial_cfg.dataset,
                mock_trial_cfg.training,
                mock_trial_cfg.augmentation,
                mock_trial_cfg.num_workers,
                is_optuna=True,
                task_type=mock_trial_cfg.task_type,
            )


# CLEANUP METHOD TESTS
@pytest.mark.unit
def test_cleanup_with_cuda_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _cleanup clears CUDA cache when available."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    search_space: dict[str, Any] = {}
    mock_dataset_loader = MagicMock(return_value=MagicMock())

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space=search_space,
        device=torch.device("cpu"),
        dataset_loader=mock_dataset_loader,
    )

    cuda_cache_cleared = False

    def mock_empty_cache() -> None:
        nonlocal cuda_cache_cleared
        cuda_cache_cleared = True

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", mock_empty_cache)

    objective._cleanup()

    assert cuda_cache_cleared


@pytest.mark.unit
def test_cleanup_with_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _cleanup does nothing when CUDA unavailable."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    search_space: dict[str, Any] = {}
    mock_dataset_loader = MagicMock(return_value=MagicMock())

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space=search_space,
        device=torch.device("cpu"),
        dataset_loader=mock_dataset_loader,
    )

    cuda_cache_cleared = False

    def mock_empty_cache() -> None:
        nonlocal cuda_cache_cleared
        cuda_cache_cleared = True

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", mock_empty_cache)

    objective._cleanup()

    assert not cuda_cache_cleared


@pytest.mark.unit
def test_cleanup_with_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _cleanup clears MPS cache when MPS is available."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    search_space: dict[str, Any] = {}
    mock_dataset_loader = MagicMock(return_value=MagicMock())

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space=search_space,
        device=torch.device("cpu"),
        dataset_loader=mock_dataset_loader,
    )

    mps_cache_cleared = False

    def mock_mps_empty_cache() -> None:
        nonlocal mps_cache_cleared
        mps_cache_cleared = True

    mock_mps_backend = MagicMock()
    mock_mps_backend.is_available.return_value = True
    mock_mps = MagicMock()
    mock_mps.empty_cache = mock_mps_empty_cache

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "mps", mock_mps_backend)
    monkeypatch.setattr(torch, "mps", mock_mps)

    objective._cleanup()

    assert mps_cache_cleared


@pytest.mark.unit
def test_trial_config_builder_preserves_resolution_when_none() -> None:
    """Test TrialConfigBuilder sets resolution when missing from model_dump."""
    from orchard.optimization.objective.config_builder import TrialConfigBuilder

    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 15
    mock_cfg.dataset._ensure_metadata = {"num_classes": 10}
    mock_cfg.dataset.resolution = 224

    mock_cfg.model_dump.return_value = {
        "dataset": {"resolution": None},
        "training": {"epochs": 60, "mixup_epochs": 20},
        "architecture": {},
        "augmentation": {},
    }

    builder = TrialConfigBuilder(mock_cfg)

    with patch("orchard.optimization.objective.config_builder.Config") as MockConfig:
        trial_params = {"learning_rate": 0.001}
        builder.build(trial_params)

        call_args = MockConfig.call_args[1]
        assert call_args["dataset"]["resolution"] == 224


@pytest.mark.unit
def test_trial_config_builder_keeps_existing_resolution() -> None:
    """Test TrialConfigBuilder doesn't override existing resolution."""
    from orchard.optimization.objective.config_builder import TrialConfigBuilder

    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 15
    mock_cfg.dataset._ensure_metadata = {"num_classes": 10}
    mock_cfg.dataset.resolution = 224

    mock_cfg.model_dump.return_value = {
        "dataset": {"resolution": 28},
        "training": {"epochs": 60, "mixup_epochs": 20},
        "architecture": {},
        "augmentation": {},
    }

    builder = TrialConfigBuilder(mock_cfg)

    with patch("orchard.optimization.objective.config_builder.Config") as MockConfig:
        trial_params = {"learning_rate": 0.001}
        builder.build(trial_params)

        call_args = MockConfig.call_args[1]
        assert call_args["dataset"]["resolution"] == 28


# ---------------------------------------------------------------------------
# Mutation-killing tests: init attrs, tracker flow, log_params, weighted_loss
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_objective_init_stores_all_attributes() -> None:
    """Assert all __init__ attributes are stored correctly."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_direction = "maximize"
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    search_space = {"lr": MagicMock()}
    device = torch.device("cpu")
    mock_tracker = MagicMock()
    mock_loader = MagicMock(return_value=MagicMock())

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space=search_space,
        device=device,
        dataset_loader=mock_loader,
        tracker=mock_tracker,
    )

    assert objective.cfg is mock_cfg
    assert objective.search_space is search_space
    assert objective.device == device
    assert objective.tracker is mock_tracker


@pytest.mark.unit
def test_objective_init_tracker_default_none() -> None:
    """Assert tracker defaults to None."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_direction = "maximize"
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space={},
        device=torch.device("cpu"),
        dataset_loader=MagicMock(return_value=MagicMock()),
    )

    assert objective.tracker is None


@pytest.mark.unit
def test_objective_init_metric_extractor_config() -> None:
    """Assert metric_extractor uses correct metric_name and direction."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_direction = "minimize"
    mock_cfg.training.monitor_metric = "loss"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space={},
        device=torch.device("cpu"),
        dataset_loader=MagicMock(return_value=MagicMock()),
    )

    assert objective.metric_extractor.metric_name == "loss"
    assert objective.metric_extractor.direction == "minimize"


@pytest.mark.unit
def test_objective_call_resets_metric_extractor() -> None:
    """Verify metric_extractor.reset() is called at start of __call__."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.training.monitor_direction = "maximize"
    mock_cfg.dataset._ensure_metadata = MagicMock()
    mock_cfg.architecture.pretrained = True

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space={},
        device=torch.device("cpu"),
        dataset_loader=MagicMock(return_value=MagicMock()),
        dataloader_factory=MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock())),
        model_factory=MagicMock(return_value=MagicMock()),
    )
    objective._cleanup = MagicMock()  # type: ignore
    objective.metric_extractor = MagicMock()
    _mock_trial_cfg = MagicMock()
    _mock_trial_cfg.training.weighted_loss = False
    objective.config_builder.build = MagicMock(return_value=_mock_trial_cfg)  # type: ignore

    mock_trial = MagicMock()
    mock_trial.number = 0

    with (
        patch("orchard.optimization.objective.objective.get_optimizer"),
        patch("orchard.optimization.objective.objective.get_scheduler"),
        patch("orchard.optimization.objective.objective.get_task"),
        patch("orchard.optimization.objective.objective.log_trial_start"),
        patch("orchard.optimization.objective.objective.TrialTrainingExecutor") as mock_exec_cls,
    ):
        mock_exec_cls.return_value.execute.return_value = 0.9
        objective(mock_trial)

    objective.metric_extractor.reset.assert_called_once()


@pytest.mark.unit
def test_objective_call_log_params_includes_pretrained() -> None:
    """Assert log_trial_start receives params with 'pretrained' key."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.training.monitor_direction = "maximize"
    mock_cfg.dataset._ensure_metadata = MagicMock()
    mock_cfg.architecture.pretrained = True

    mock_suggest = MagicMock(return_value=0.001)
    search_space = {"learning_rate": mock_suggest}

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space=search_space,
        device=torch.device("cpu"),
        dataset_loader=MagicMock(return_value=MagicMock()),
        dataloader_factory=MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock())),
        model_factory=MagicMock(return_value=MagicMock()),
    )
    objective._cleanup = MagicMock()  # type: ignore
    _mock_trial_cfg = MagicMock()
    _mock_trial_cfg.training.weighted_loss = False
    objective.config_builder.build = MagicMock(return_value=_mock_trial_cfg)  # type: ignore

    mock_trial = MagicMock()
    mock_trial.number = 1

    with (
        patch("orchard.optimization.objective.objective.get_optimizer"),
        patch("orchard.optimization.objective.objective.get_scheduler"),
        patch("orchard.optimization.objective.objective.get_task"),
        patch("orchard.optimization.objective.objective.log_trial_start") as mock_log,
        patch("orchard.optimization.objective.objective.TrialTrainingExecutor") as mock_exec,
    ):
        mock_exec.return_value.execute.return_value = 0.9
        objective(mock_trial)

    log_params = mock_log.call_args[0][1]
    assert "pretrained" in log_params
    assert log_params["pretrained"] is True


@pytest.mark.unit
def test_objective_tracker_end_with_best_metric_on_success() -> None:
    """Verify tracker.end_optuna_trial receives best_metric on success."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.training.monitor_direction = "maximize"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    mock_tracker = MagicMock()

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space={},
        device=torch.device("cpu"),
        dataset_loader=MagicMock(return_value=MagicMock()),
        dataloader_factory=MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock())),
        model_factory=MagicMock(return_value=MagicMock()),
        tracker=mock_tracker,
    )
    objective._cleanup = MagicMock()  # type: ignore
    _mock_trial_cfg = MagicMock()
    _mock_trial_cfg.training.weighted_loss = False
    objective.config_builder.build = MagicMock(return_value=_mock_trial_cfg)  # type: ignore
    mock_trial = MagicMock()
    mock_trial.number = 0

    def _fake_execute(trial: MagicMock) -> float:
        # Simulate what real executor does: update best_metric on the shared extractor
        objective.metric_extractor.update_best(0.95)
        return 0.95

    with (
        patch("orchard.optimization.objective.objective.get_optimizer"),
        patch("orchard.optimization.objective.objective.get_scheduler"),
        patch("orchard.optimization.objective.objective.get_task"),
        patch("orchard.optimization.objective.objective.log_trial_start"),
        patch("orchard.optimization.objective.objective.TrialTrainingExecutor") as mock_exec,
    ):
        mock_exec.return_value.execute.side_effect = _fake_execute
        objective(mock_trial)

    mock_tracker.end_optuna_trial.assert_called_once_with(0.95)


@pytest.mark.unit
def test_objective_worst_metric_maximize() -> None:
    """Assert _worst_metric returns -inf for maximize direction."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_direction = "maximize"
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space={},
        device=torch.device("cpu"),
        dataset_loader=MagicMock(return_value=MagicMock()),
    )

    assert objective._worst_metric() == -float("inf")


@pytest.mark.unit
def test_objective_worst_metric_minimize() -> None:
    """Assert _worst_metric returns +inf for minimize direction."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_direction = "minimize"
    mock_cfg.training.monitor_metric = "loss"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space={},
        device=torch.device("cpu"),
        dataset_loader=MagicMock(return_value=MagicMock()),
    )

    assert objective._worst_metric() == float("inf")


@pytest.mark.unit
def test_objective_weighted_loss_calls_compute_class_weights() -> None:
    """Verify compute_class_weights is called when weighted_loss=True."""
    mock_cfg = MagicMock()
    mock_cfg.task_type = "classification"
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.training.monitor_direction = "maximize"
    mock_cfg.dataset._ensure_metadata = MagicMock()
    mock_cfg.dataset._ensure_metadata.num_classes = 5

    import numpy as np

    mock_train_loader = MagicMock()
    mock_train_loader.dataset.labels.flatten.return_value = np.array([0, 1, 2, 3, 4])

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space={},
        device=torch.device("cpu"),
        dataset_loader=MagicMock(return_value=MagicMock()),
        dataloader_factory=MagicMock(return_value=(mock_train_loader, MagicMock(), MagicMock())),
        model_factory=MagicMock(return_value=MagicMock()),
    )
    objective._cleanup = MagicMock()  # type: ignore

    _mock_trial_cfg = MagicMock()
    _mock_trial_cfg.training.weighted_loss = True
    objective.config_builder.build = MagicMock(return_value=_mock_trial_cfg)  # type: ignore

    mock_trial = MagicMock()
    mock_trial.number = 0

    with (
        patch("orchard.optimization.objective.objective.get_optimizer"),
        patch("orchard.optimization.objective.objective.get_scheduler"),
        patch("orchard.optimization.objective.objective.get_task"),
        patch("orchard.optimization.objective.objective.log_trial_start"),
        patch("orchard.optimization.objective.objective.compute_class_weights") as mock_cw,
        patch("orchard.optimization.objective.objective.TrialTrainingExecutor") as mock_exec,
    ):
        mock_exec.return_value.execute.return_value = 0.9
        objective(mock_trial)

    mock_cw.assert_called_once()


@pytest.mark.unit
def test_objective_non_classification_skips_class_weights() -> None:
    """Verify compute_class_weights is NOT called for non-classification tasks.

    Kills ``and`` → ``or`` mutant on the task_type guard.
    """
    mock_cfg = MagicMock()
    mock_cfg.task_type = "detection"
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.training.monitor_direction = "maximize"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space={},
        device=torch.device("cpu"),
        dataset_loader=MagicMock(return_value=MagicMock()),
        dataloader_factory=MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock())),
        model_factory=MagicMock(return_value=MagicMock()),
    )
    objective._cleanup = MagicMock()  # type: ignore

    _mock_trial_cfg = MagicMock()
    _mock_trial_cfg.training.weighted_loss = True
    objective.config_builder.build = MagicMock(return_value=_mock_trial_cfg)  # type: ignore

    mock_trial = MagicMock()
    mock_trial.number = 0

    with (
        patch("orchard.optimization.objective.objective.get_optimizer"),
        patch("orchard.optimization.objective.objective.get_scheduler"),
        patch("orchard.optimization.objective.objective.get_task"),
        patch("orchard.optimization.objective.objective.log_trial_start"),
        patch("orchard.optimization.objective.objective.compute_class_weights") as mock_cw,
        patch("orchard.optimization.objective.objective.TrialTrainingExecutor") as mock_exec,
    ):
        mock_exec.return_value.execute.return_value = 0.9
        objective(mock_trial)

    mock_cw.assert_not_called()


# ---------------------------------------------------------------------------
# Mutation-killing: _sample_params dispatch, TrialTrainingExecutor wiring,
# trial_succeeded state, log_trial_start args
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sample_params_uses_sample_params_attr() -> None:
    """Verify _sample_params dispatches to obj.sample_params when attr exists."""
    mock_cfg = MagicMock()
    mock_cfg.training.monitor_direction = "maximize"
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    search_space = MagicMock()
    search_space.sample_params.return_value = {"lr": 0.01}

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space=search_space,
        device=torch.device("cpu"),
        dataset_loader=MagicMock(return_value=MagicMock()),
    )

    trial = MagicMock()
    result = objective._sample_params(trial)

    search_space.sample_params.assert_called_once_with(trial)
    assert result == {"lr": 0.01}


@pytest.mark.unit
def test_sample_params_dict_fallback() -> None:
    """Verify _sample_params iterates dict items when no sample_params attr."""
    mock_cfg = MagicMock()
    mock_cfg.training.monitor_direction = "maximize"
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    trial = MagicMock()
    fn = MagicMock(return_value=0.001)
    search_space = {"lr": fn}

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space=search_space,
        device=torch.device("cpu"),
        dataset_loader=MagicMock(return_value=MagicMock()),
    )

    result = objective._sample_params(trial)
    fn.assert_called_once_with(trial)
    assert result == {"lr": 0.001}


@pytest.mark.unit
def test_call_passes_correct_kwargs_to_executor() -> None:
    """Verify TrialTrainingExecutor receives correct wired kwargs."""
    mock_cfg = MagicMock()
    mock_cfg.training.monitor_direction = "maximize"
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    mock_model = MagicMock()
    mock_train_loader = MagicMock()
    mock_val_loader = MagicMock()
    mock_optimizer = MagicMock()
    mock_scheduler = MagicMock()
    mock_criterion = MagicMock()

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space={},
        device=torch.device("cpu"),
        dataset_loader=MagicMock(return_value=MagicMock()),
        dataloader_factory=MagicMock(
            return_value=(mock_train_loader, mock_val_loader, MagicMock())
        ),
        model_factory=MagicMock(return_value=mock_model),
    )
    objective._cleanup = MagicMock()  # type: ignore

    _mock_trial_cfg = MagicMock()
    _mock_trial_cfg.training.weighted_loss = False
    objective.config_builder.build = MagicMock(return_value=_mock_trial_cfg)  # type: ignore

    mock_trial = MagicMock()
    mock_trial.number = 0

    with (
        patch(
            "orchard.optimization.objective.objective.get_optimizer", return_value=mock_optimizer
        ),
        patch(
            "orchard.optimization.objective.objective.get_scheduler", return_value=mock_scheduler
        ),
        patch(
            "orchard.optimization.objective.objective.get_task",
            return_value=MagicMock(
                criterion_factory=MagicMock(get_criterion=MagicMock(return_value=mock_criterion))
            ),
        ),
        patch("orchard.optimization.objective.objective.log_trial_start"),
        patch("orchard.optimization.objective.objective.TrialTrainingExecutor") as mock_exec_cls,
    ):
        mock_exec_cls.return_value.execute.return_value = 0.9
        objective(mock_trial)

    # Verify TrialTrainingExecutor was constructed with correct kwargs
    call_kwargs = mock_exec_cls.call_args[1]
    assert call_kwargs["model"] is mock_model
    assert call_kwargs["train_loader"] is mock_train_loader
    assert call_kwargs["val_loader"] is mock_val_loader
    assert call_kwargs["optimizer"] is mock_optimizer
    assert call_kwargs["scheduler"] is mock_scheduler
    assert call_kwargs["criterion"] is mock_criterion
    assert call_kwargs["device"] == torch.device("cpu")
    assert call_kwargs["metric_extractor"] is objective.metric_extractor
    assert call_kwargs["training"] is _mock_trial_cfg.training
    assert call_kwargs["optuna"] is _mock_trial_cfg.optuna
    adapters = call_kwargs["task_adapters"]
    assert adapters.training_step is not None
    assert adapters.validation_metrics is not None


@pytest.mark.unit
def test_call_uses_task_registry_for_criterion() -> None:
    """Verify __call__ uses get_task() for criterion and validation_metrics."""
    mock_cfg = MagicMock()
    mock_cfg.training.monitor_direction = "maximize"
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.training.weighted_loss = False
    mock_cfg.dataset._ensure_metadata = MagicMock()
    mock_cfg.task_type = "classification"

    mock_task = MagicMock()
    mock_criterion = MagicMock()
    mock_task.criterion_factory.get_criterion.return_value = mock_criterion

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space={},
        device=torch.device("cpu"),
        dataset_loader=MagicMock(return_value=MagicMock()),
        dataloader_factory=MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock())),
        model_factory=MagicMock(return_value=MagicMock()),
    )
    objective._cleanup = MagicMock()  # type: ignore
    objective.config_builder.build = MagicMock(return_value=MagicMock(training=mock_cfg.training))  # type: ignore

    mock_trial = MagicMock()
    mock_trial.number = 0

    with (
        patch("orchard.optimization.objective.objective.get_optimizer"),
        patch("orchard.optimization.objective.objective.get_scheduler"),
        patch(
            "orchard.optimization.objective.objective.get_task", return_value=mock_task
        ) as mock_get_task,
        patch("orchard.optimization.objective.objective.log_trial_start"),
        patch("orchard.optimization.objective.objective.TrialTrainingExecutor") as mock_exec_cls,
    ):
        mock_exec_cls.return_value.execute.return_value = 0.9
        objective(mock_trial)

    mock_get_task.assert_called_once_with(mock_cfg.task_type)
    mock_task.criterion_factory.get_criterion.assert_called_once()
    call_kwargs = mock_exec_cls.call_args[1]
    assert call_kwargs["criterion"] is mock_criterion
    adapters = call_kwargs["task_adapters"]
    assert adapters.training_step is mock_task.training_step
    assert adapters.validation_metrics is mock_task.validation_metrics


@pytest.mark.unit
def test_call_log_trial_start_receives_correct_args() -> None:
    """Verify log_trial_start is called with trial.number and log_params including pretrained."""
    mock_cfg = MagicMock()
    mock_cfg.training.monitor_direction = "maximize"
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.architecture.pretrained = True
    mock_cfg.dataset._ensure_metadata = MagicMock()

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space={"lr": MagicMock(return_value=0.01)},
        device=torch.device("cpu"),
        dataset_loader=MagicMock(return_value=MagicMock()),
        dataloader_factory=MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock())),
        model_factory=MagicMock(return_value=MagicMock()),
    )
    objective._cleanup = MagicMock()  # type: ignore

    _mock_trial_cfg = MagicMock()
    _mock_trial_cfg.training.weighted_loss = False
    objective.config_builder.build = MagicMock(return_value=_mock_trial_cfg)  # type: ignore

    mock_trial = MagicMock()
    mock_trial.number = 7

    with (
        patch("orchard.optimization.objective.objective.get_optimizer"),
        patch("orchard.optimization.objective.objective.get_scheduler"),
        patch("orchard.optimization.objective.objective.get_task"),
        patch("orchard.optimization.objective.objective.log_trial_start") as mock_log,
        patch("orchard.optimization.objective.objective.TrialTrainingExecutor") as mock_exec,
    ):
        mock_exec.return_value.execute.return_value = 0.9
        objective(mock_trial)

    mock_log.assert_called_once()
    call_args = mock_log.call_args[0]
    assert call_args[0] == 7  # trial.number
    assert call_args[1]["pretrained"] is True
    assert call_args[1]["lr"] == 0.01


@pytest.mark.unit
def test_call_tracker_not_called_on_failure() -> None:
    """Verify tracker.end_optuna_trial gets worst metric when trial fails."""
    mock_cfg = MagicMock()
    mock_cfg.training.monitor_direction = "maximize"
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    mock_tracker = MagicMock()

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space={},
        device=torch.device("cpu"),
        dataset_loader=MagicMock(return_value=MagicMock()),
        dataloader_factory=MagicMock(side_effect=RuntimeError("boom")),
        model_factory=MagicMock(),
        tracker=mock_tracker,
    )
    objective._cleanup = MagicMock()  # type: ignore

    _mock_trial_cfg = MagicMock()
    _mock_trial_cfg.training.weighted_loss = False
    objective.config_builder.build = MagicMock(return_value=_mock_trial_cfg)  # type: ignore

    mock_trial = MagicMock()
    mock_trial.number = 0

    with patch("orchard.optimization.objective.objective.log_trial_start"):
        result = objective(mock_trial)

    # Failed trial returns worst metric
    assert result == -float("inf")
    # Tracker gets worst metric (not best)
    mock_tracker.end_optuna_trial.assert_called_once_with(-float("inf"))


@pytest.mark.unit
def test_call_dataloader_factory_receives_is_optuna() -> None:
    """Verify dataloader_factory is called with is_optuna=True."""
    mock_cfg = MagicMock()
    mock_cfg.training.monitor_direction = "maximize"
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    mock_factory = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock()))

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space={},
        device=torch.device("cpu"),
        dataset_loader=MagicMock(return_value=MagicMock()),
        dataloader_factory=mock_factory,
        model_factory=MagicMock(return_value=MagicMock()),
    )
    objective._cleanup = MagicMock()  # type: ignore

    _mock_trial_cfg = MagicMock()
    _mock_trial_cfg.training.weighted_loss = False
    objective.config_builder.build = MagicMock(return_value=_mock_trial_cfg)  # type: ignore

    mock_trial = MagicMock()
    mock_trial.number = 0

    with (
        patch("orchard.optimization.objective.objective.get_optimizer"),
        patch("orchard.optimization.objective.objective.get_scheduler"),
        patch("orchard.optimization.objective.objective.get_task"),
        patch("orchard.optimization.objective.objective.log_trial_start"),
        patch("orchard.optimization.objective.objective.TrialTrainingExecutor") as mock_exec,
    ):
        mock_exec.return_value.execute.return_value = 0.9
        objective(mock_trial)

    # Verify is_optuna=True was passed
    call_kwargs = mock_factory.call_args
    assert (
        call_kwargs[0][4] == _mock_trial_cfg.num_workers or call_kwargs[1].get("is_optuna") is True
    )
    # More direct check: 6th positional arg or keyword
    args, kwargs = call_kwargs
    if "is_optuna" in kwargs:
        assert kwargs["is_optuna"] is True
    else:
        assert args[5] is True  # is_optuna is the 6th positional arg


# ---------------------------------------------------------------------------
# MUTANT KILLERS: __call__ exact args, trial_succeeded, _sample_params
# ---------------------------------------------------------------------------


def _make_objective_with_tracker() -> tuple[OptunaObjective, MagicMock, MagicMock]:
    """Helper: build OptunaObjective with mock tracker + mock trial cfg."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.training.monitor_direction = "maximize"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    mock_tracker = MagicMock()

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space={},
        device=torch.device("cpu"),
        dataset_loader=MagicMock(return_value=MagicMock()),
        dataloader_factory=MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock())),
        model_factory=MagicMock(return_value=MagicMock()),
        tracker=mock_tracker,
    )
    objective._cleanup = MagicMock()  # type: ignore

    mock_trial_cfg = MagicMock()
    mock_trial_cfg.training.weighted_loss = False
    objective.config_builder.build = MagicMock(return_value=mock_trial_cfg)  # type: ignore

    mock_trial = MagicMock()
    mock_trial.number = 1

    return objective, mock_tracker, mock_trial


@pytest.mark.unit
def test_successful_trial_sends_best_metric_to_tracker() -> None:
    """Kill mutmut_17/18: trial_succeeded=True → end_optuna_trial(best_metric)."""
    objective, mock_tracker, mock_trial = _make_objective_with_tracker()

    with (
        patch("orchard.optimization.objective.objective.get_optimizer"),
        patch("orchard.optimization.objective.objective.get_scheduler"),
        patch("orchard.optimization.objective.objective.get_task"),
        patch("orchard.optimization.objective.objective.log_trial_start"),
        patch(
            "orchard.optimization.objective.objective.TrialTrainingExecutor"
        ) as mock_executor_cls,
    ):
        mock_executor_cls.return_value.execute.return_value = 0.92
        objective(mock_trial)

    # trial_succeeded=True → end_optuna_trial(best_metric), NOT worst_metric
    mock_tracker.end_optuna_trial.assert_called_once_with(objective.metric_extractor.best_metric)


@pytest.mark.unit
def test_pruned_trial_sends_best_metric_to_tracker() -> None:
    """Kill mutmut_100/101: trial_succeeded=True after pruning → end_optuna_trial(best_metric)."""
    objective, mock_tracker, mock_trial = _make_objective_with_tracker()

    with (
        patch("orchard.optimization.objective.objective.get_optimizer"),
        patch("orchard.optimization.objective.objective.get_scheduler"),
        patch("orchard.optimization.objective.objective.get_task"),
        patch("orchard.optimization.objective.objective.log_trial_start"),
        patch(
            "orchard.optimization.objective.objective.TrialTrainingExecutor"
        ) as mock_executor_cls,
    ):
        mock_executor_cls.return_value.execute.side_effect = optuna.TrialPruned()

        with pytest.raises(optuna.TrialPruned):
            objective(mock_trial)

    # Pruned but trial_succeeded=True → best_metric, not worst
    mock_tracker.end_optuna_trial.assert_called_once_with(objective.metric_extractor.best_metric)


@pytest.mark.unit
def test_call_passes_exact_args_to_model_factory() -> None:
    """Kill mutmut_34/35: verify model_factory receives (device, dataset, arch)."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    device = torch.device("cpu")
    mock_model_factory = MagicMock(return_value=MagicMock())

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space={},
        device=device,
        dataset_loader=MagicMock(return_value=MagicMock()),
        dataloader_factory=MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock())),
        model_factory=mock_model_factory,
    )
    objective._cleanup = MagicMock()  # type: ignore

    mock_trial_cfg = MagicMock()
    mock_trial_cfg.training.weighted_loss = False
    objective.config_builder.build = MagicMock(return_value=mock_trial_cfg)  # type: ignore

    mock_trial = MagicMock()
    mock_trial.number = 1

    with (
        patch("orchard.optimization.objective.objective.get_optimizer"),
        patch("orchard.optimization.objective.objective.get_scheduler"),
        patch("orchard.optimization.objective.objective.get_task"),
        patch("orchard.optimization.objective.objective.log_trial_start"),
        patch(
            "orchard.optimization.objective.objective.TrialTrainingExecutor"
        ) as mock_executor_cls,
    ):
        mock_executor_cls.return_value.execute.return_value = 0.9
        objective(mock_trial)

    mock_model_factory.assert_called_once_with(
        device, mock_trial_cfg.dataset, mock_trial_cfg.architecture
    )


@pytest.mark.unit
def test_call_passes_trial_to_sample_params() -> None:
    """Kill mutmut_2: _sample_params receives the actual trial, not None."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    mock_suggest = MagicMock(return_value=0.01)
    search_space = {"lr": mock_suggest}

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space=search_space,
        device=torch.device("cpu"),
        dataset_loader=MagicMock(return_value=MagicMock()),
        dataloader_factory=MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock())),
        model_factory=MagicMock(return_value=MagicMock()),
    )
    objective._cleanup = MagicMock()  # type: ignore

    mock_trial_cfg = MagicMock()
    mock_trial_cfg.training.weighted_loss = False
    objective.config_builder.build = MagicMock(return_value=mock_trial_cfg)  # type: ignore

    mock_trial = MagicMock()
    mock_trial.number = 1

    with (
        patch("orchard.optimization.objective.objective.get_optimizer"),
        patch("orchard.optimization.objective.objective.get_scheduler"),
        patch("orchard.optimization.objective.objective.get_task"),
        patch("orchard.optimization.objective.objective.log_trial_start"),
        patch(
            "orchard.optimization.objective.objective.TrialTrainingExecutor"
        ) as mock_executor_cls,
    ):
        mock_executor_cls.return_value.execute.return_value = 0.9
        objective(mock_trial)

    # The suggest function should have been called with the actual trial
    mock_suggest.assert_called_once_with(mock_trial)


@pytest.mark.unit
def test_sample_params_hasattr_exact_string() -> None:
    """Kill mutmut_5 in _sample_params: hasattr checks exact 'sample_params'."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    # Object WITHOUT sample_params → falls back to dict iteration
    search_space_no_method: dict[str, Any] = {"lr": MagicMock(return_value=0.01)}

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space=search_space_no_method,
        device=torch.device("cpu"),
        dataset_loader=MagicMock(return_value=MagicMock()),
    )

    mock_trial = MagicMock()
    result = objective._sample_params(mock_trial)
    assert "lr" in result

    # Object WITH sample_params → uses it
    mock_search_space = MagicMock()
    mock_search_space.sample_params.return_value = {"lr": 0.001}
    objective.search_space = mock_search_space

    result2 = objective._sample_params(mock_trial)
    assert result2 == {"lr": 0.001}
    mock_search_space.sample_params.assert_called_once_with(mock_trial)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
