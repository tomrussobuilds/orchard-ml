"""
Unit tests for the refactored OptunaObjective.

These tests validate the behavior of the Optuna objective and its
supporting components using dependency injection, enabling high
coverage through isolated and deterministic unit tests.
"""

from __future__ import annotations

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


# TRIAL CONFIG BUILDER TESTS
@pytest.mark.unit
def test_config_builder_preserves_metadata():
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
def test_config_builder_applies_param_overrides():
    """Test TrialConfigBuilder applies parameter overrides correctly."""
    mock_cfg = MagicMock()
    config_dict = {
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

    test_dict = config_dict.copy()
    test_dict["dataset"]["metadata"] = builder.base_metadata
    test_dict["training"]["epochs"] = builder.optuna_epochs
    builder._apply_param_overrides(test_dict, trial_params)

    assert test_dict["training"]["learning_rate"] == pytest.approx(0.0001)
    assert test_dict["architecture"]["dropout"] == pytest.approx(0.3)
    assert test_dict["augmentation"]["rotation_angle"] == 15
    assert test_dict["training"]["epochs"] == 20


@pytest.mark.unit
def test_config_builder_handles_model_name():
    """Test TrialConfigBuilder maps model_name to architecture.name."""
    mock_cfg = MagicMock()
    config_dict = {
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

    test_dict = config_dict.copy()
    test_dict["dataset"]["metadata"] = builder.base_metadata
    test_dict["training"]["epochs"] = builder.optuna_epochs
    builder._apply_param_overrides(test_dict, trial_params)

    assert test_dict["architecture"]["name"] == "vit_tiny"


@pytest.mark.unit
def test_config_builder_handles_weight_variant():
    """Test TrialConfigBuilder applies weight_variant."""
    mock_cfg = MagicMock()
    config_dict = {
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

    test_dict = config_dict.copy()
    test_dict["dataset"]["metadata"] = builder.base_metadata
    test_dict["training"]["epochs"] = builder.optuna_epochs
    builder._apply_param_overrides(test_dict, trial_params)

    assert (
        test_dict["architecture"]["weight_variant"] == "vit_tiny_patch16_224.augreg_in21k_ft_in1k"
    )


@pytest.mark.unit
def test_config_builder_maps_criterion_and_focal_gamma():
    """Test TrialConfigBuilder maps criterion_type and focal_gamma to training."""
    mock_cfg = MagicMock()
    config_dict = {
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

    test_dict = config_dict.copy()
    test_dict["dataset"]["metadata"] = builder.base_metadata
    test_dict["training"]["epochs"] = builder.optuna_epochs
    builder._apply_param_overrides(test_dict, trial_params)

    assert test_dict["training"]["criterion_type"] == "focal"
    assert test_dict["training"]["focal_gamma"] == pytest.approx(3.5)


@pytest.mark.unit
def test_config_builder_skips_none_weight_variant():
    """Test TrialConfigBuilder skips None weight_variant (for non-ViT models)."""
    mock_cfg = MagicMock()
    config_dict = {
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

    test_dict = config_dict.copy()
    test_dict["dataset"]["metadata"] = builder.base_metadata
    test_dict["training"]["epochs"] = builder.optuna_epochs
    builder._apply_param_overrides(test_dict, trial_params)

    # Should preserve original value and not overwrite with None
    assert test_dict["architecture"]["weight_variant"] == "original_value"


# METRIC EXTRACTOR TESTS
@pytest.mark.unit
def test_metric_extractor_extracts_correct_metric():
    """Test MetricExtractor extracts specified metric."""
    extractor = MetricExtractor(metric_name="auc")

    val_metrics = {"loss": 0.5, "accuracy": 0.85, "auc": 0.92}

    result = extractor.extract(val_metrics)

    assert result == pytest.approx(0.92)


@pytest.mark.unit
def test_metric_extractor_raises_on_missing_metric():
    """Test MetricExtractor raises KeyError for missing metric."""
    extractor = MetricExtractor(metric_name="f1")

    val_metrics = {"loss": 0.5, "accuracy": 0.85, "auc": 0.92}

    with pytest.raises(KeyError, match="f1"):
        extractor.extract(val_metrics)


@pytest.mark.unit
def test_metric_extractor_tracks_best():
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
def test_metric_extractor_reset():
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
def test_training_executor_should_prune_warmup():
    """Test TrialTrainingExecutor respects warmup period."""
    mock_trial = MagicMock()
    mock_trial.should_prune.return_value = True

    mock_cfg = MagicMock()
    mock_cfg.training.use_amp = False
    mock_cfg.training.epochs = 30
    mock_cfg.training.mixup_alpha = 0
    mock_cfg.training.scheduler_type = "cosine"
    mock_cfg.optuna.enable_pruning = True
    mock_cfg.optuna.pruning_warmup_epochs = 10

    executor = TrialTrainingExecutor(
        model=MagicMock(),
        train_loader=MagicMock(),
        val_loader=MagicMock(),
        optimizer=MagicMock(),
        scheduler=MagicMock(),
        criterion=MagicMock(),
        training=mock_cfg.training,
        optuna=mock_cfg.optuna,
        log_interval=5,
        device=torch.device("cpu"),
        metric_extractor=MetricExtractor("auc"),
    )

    assert executor._should_prune(mock_trial, epoch=5) is False

    assert executor._should_prune(mock_trial, epoch=15) is True


@pytest.mark.unit
def test_training_executor_disabled_pruning():
    """Test TrialTrainingExecutor with pruning disabled."""
    mock_trial = MagicMock()
    mock_trial.should_prune.return_value = True

    mock_cfg = MagicMock()
    mock_cfg.training.use_amp = False
    mock_cfg.training.epochs = 30
    mock_cfg.training.grad_clip = 0.0
    mock_cfg.training.mixup_alpha = 0
    mock_cfg.training.scheduler_type = "cosine"
    mock_cfg.training.mixup_epochs = 0
    mock_cfg.optuna.enable_pruning = False
    mock_cfg.optuna.pruning_warmup_epochs = 10

    assert mock_cfg.optuna.enable_pruning is False

    executor = TrialTrainingExecutor(
        model=MagicMock(),
        train_loader=MagicMock(),
        val_loader=MagicMock(),
        optimizer=MagicMock(),
        scheduler=MagicMock(),
        criterion=MagicMock(),
        training=mock_cfg.training,
        optuna=mock_cfg.optuna,
        log_interval=5,
        device=torch.device("cpu"),
        metric_extractor=MetricExtractor("auc"),
    )

    result1 = executor._should_prune(mock_trial, epoch=5)
    result2 = executor._should_prune(mock_trial, epoch=15)

    assert result1 is False, f"Expected False but got {result1}"
    assert result2 is False, f"Expected False but got {result2}"


@pytest.mark.unit
def test_training_executor_validate_epoch_error_handling():
    """Test TrialTrainingExecutor handles validation errors."""
    mock_cfg = MagicMock()
    mock_cfg.training.use_amp = False
    mock_cfg.training.epochs = 30
    mock_cfg.training.mixup_alpha = 0
    mock_cfg.optuna.enable_pruning = True
    mock_cfg.optuna.pruning_warmup_epochs = 5

    executor = TrialTrainingExecutor(
        model=MagicMock(),
        train_loader=MagicMock(),
        val_loader=MagicMock(),
        optimizer=MagicMock(),
        scheduler=MagicMock(),
        criterion=MagicMock(),
        training=mock_cfg.training,
        optuna=mock_cfg.optuna,
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

    assert result == {"loss": 999.0, "accuracy": 0.0, "auc": 0.0, "f1": 0.0}


# OPTUNA OBJECTIVE TESTS
@pytest.mark.unit
def test_optuna_objective_init_with_defaults():
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
def test_optuna_objective_uses_injected_dependencies():
    """Test OptunaObjective uses all injected dependencies."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 20
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    search_space = {}
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
def test_optuna_objective_sample_params_dict():
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
def test_optuna_objective_sample_params_object():
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
def test_optuna_objective_call_with_pruning():
    """Test OptunaObjective.__call__ handles pruning correctly."""

    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    search_space = {}
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
    objective.config_builder.build = MagicMock(return_value=_mock_trial_cfg)

    with (
        patch("orchard.optimization.objective.objective.get_optimizer"),
        patch("orchard.optimization.objective.objective.get_scheduler"),
        patch("orchard.optimization.objective.objective.get_criterion"),
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
def test_optuna_objective_call_cleanup_on_success():
    """Test OptunaObjective.__call__ calls cleanup after successful trial."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    search_space = {}
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

    objective._cleanup = MagicMock()

    _mock_trial_cfg = MagicMock()
    _mock_trial_cfg.training.weighted_loss = False
    objective.config_builder.build = MagicMock(return_value=_mock_trial_cfg)

    with (
        patch("orchard.optimization.objective.objective.get_optimizer"),
        patch("orchard.optimization.objective.objective.get_scheduler"),
        patch("orchard.optimization.objective.objective.get_criterion"),
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
def test_optuna_objective_call_returns_worst_metric_on_failure():
    """Test OptunaObjective.__call__ returns worst metric on exception (maximize)."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.optuna.direction = "maximize"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    search_space = {}
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

    objective._cleanup = MagicMock()

    _mock_trial_cfg = MagicMock()
    _mock_trial_cfg.training.weighted_loss = False
    objective.config_builder.build = MagicMock(return_value=_mock_trial_cfg)

    with (
        patch("orchard.optimization.objective.objective.get_optimizer"),
        patch("orchard.optimization.objective.objective.get_scheduler"),
        patch("orchard.optimization.objective.objective.get_criterion"),
        patch("orchard.optimization.objective.objective.log_trial_start"),
    ):

        with patch(
            "orchard.optimization.objective.objective.TrialTrainingExecutor"
        ) as mock_executor_cls:
            mock_executor_instance = MagicMock()
            mock_executor_instance.execute.side_effect = RuntimeError("Training failed")
            mock_executor_cls.return_value = mock_executor_instance

            result = objective(mock_trial)

            assert result == pytest.approx(0.0)
            objective._cleanup.assert_called_once()


@pytest.mark.unit
def test_optuna_objective_call_returns_inf_on_failure_minimize():
    """Test OptunaObjective.__call__ returns inf on exception (minimize)."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.optuna.direction = "minimize"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    search_space = {}
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

    objective._cleanup = MagicMock()

    _mock_trial_cfg = MagicMock()
    _mock_trial_cfg.training.weighted_loss = False
    objective.config_builder.build = MagicMock(return_value=_mock_trial_cfg)

    with (
        patch("orchard.optimization.objective.objective.get_optimizer"),
        patch("orchard.optimization.objective.objective.get_scheduler"),
        patch("orchard.optimization.objective.objective.get_criterion"),
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
def test_optuna_objective_failed_trial_logs_worst_metric_to_tracker():
    """Test tracker receives worst_metric when trial fails before validation."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.optuna.direction = "maximize"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    search_space = {}
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
    objective._cleanup = MagicMock()

    _mock_trial_cfg = MagicMock()
    _mock_trial_cfg.training.weighted_loss = False
    objective.config_builder.build = MagicMock(return_value=_mock_trial_cfg)

    mock_trial = MagicMock()
    mock_trial.number = 5

    with (
        patch("orchard.optimization.objective.objective.get_optimizer"),
        patch("orchard.optimization.objective.objective.get_scheduler"),
        patch("orchard.optimization.objective.objective.get_criterion"),
        patch("orchard.optimization.objective.objective.log_trial_start"),
    ):
        with patch(
            "orchard.optimization.objective.objective.TrialTrainingExecutor"
        ) as mock_executor_cls:
            mock_executor_cls.return_value.execute.side_effect = RuntimeError("OOM")

            result = objective(mock_trial)

    assert result == pytest.approx(0.0)
    mock_tracker.start_optuna_trial.assert_called_once()
    # Failed trial: tracker receives worst_metric (0.0 for maximize), not -inf
    mock_tracker.end_optuna_trial.assert_called_once_with(0.0)


@pytest.mark.unit
def test_optuna_objective_call_reraises_trial_pruned():
    """Test OptunaObjective.__call__ re-raises TrialPruned (not caught)."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.optuna.direction = "maximize"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    search_space = {}
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

    objective._cleanup = MagicMock()

    _mock_trial_cfg = MagicMock()
    _mock_trial_cfg.training.weighted_loss = False
    objective.config_builder.build = MagicMock(return_value=_mock_trial_cfg)

    with (
        patch("orchard.optimization.objective.objective.get_optimizer"),
        patch("orchard.optimization.objective.objective.get_scheduler"),
        patch("orchard.optimization.objective.objective.get_criterion"),
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
def test_optuna_objective_call_builds_trial_config():
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
    objective.config_builder.build = MagicMock(return_value=mock_trial_cfg)

    with (
        patch("orchard.optimization.objective.objective.get_optimizer"),
        patch("orchard.optimization.objective.objective.get_scheduler"),
        patch("orchard.optimization.objective.objective.get_criterion"),
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
            )


# CLEANUP METHOD TESTS
@pytest.mark.unit
def test_cleanup_with_cuda_available(monkeypatch):
    """Test _cleanup clears CUDA cache when available."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    search_space = {}
    mock_dataset_loader = MagicMock(return_value=MagicMock())

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space=search_space,
        device=torch.device("cpu"),
        dataset_loader=mock_dataset_loader,
    )

    cuda_cache_cleared = False

    def mock_empty_cache():
        nonlocal cuda_cache_cleared
        cuda_cache_cleared = True

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", mock_empty_cache)

    objective._cleanup()

    assert cuda_cache_cleared


@pytest.mark.unit
def test_cleanup_with_cuda_unavailable(monkeypatch):
    """Test _cleanup does nothing when CUDA unavailable."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    search_space = {}
    mock_dataset_loader = MagicMock(return_value=MagicMock())

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space=search_space,
        device=torch.device("cpu"),
        dataset_loader=mock_dataset_loader,
    )

    cuda_cache_cleared = False

    def mock_empty_cache():
        nonlocal cuda_cache_cleared
        cuda_cache_cleared = True

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", mock_empty_cache)

    objective._cleanup()

    assert not cuda_cache_cleared


@pytest.mark.unit
def test_cleanup_with_mps(monkeypatch):
    """Test _cleanup clears MPS cache when MPS is available."""
    mock_cfg = MagicMock()
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.dataset._ensure_metadata = MagicMock()

    search_space = {}
    mock_dataset_loader = MagicMock(return_value=MagicMock())

    objective = OptunaObjective(
        cfg=mock_cfg,
        search_space=search_space,
        device=torch.device("cpu"),
        dataset_loader=mock_dataset_loader,
    )

    mps_cache_cleared = False

    def mock_mps_empty_cache():
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
def test_trial_config_builder_preserves_resolution_when_none():
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
def test_trial_config_builder_keeps_existing_resolution():
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
