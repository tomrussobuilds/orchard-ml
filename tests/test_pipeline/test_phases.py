"""
Test Suite for Pipeline Phase Functions.

Tests for run_optimization_phase, run_training_phase, and run_export_phase.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from orchard.pipeline.phases import (
    TrainingResult,
    run_export_phase,
    run_optimization_phase,
    run_training_phase,
)


@pytest.fixture
def mock_orchestrator():
    """Create a mock RootOrchestrator."""
    orch = MagicMock()
    orch.cfg = MagicMock()
    orch.cfg.dataset.resolution = 28
    orch.cfg.dataset.force_rgb = True
    orch.cfg.dataset.effective_in_channels = 3
    orch.cfg.dataset.metadata.name = "organcmnist"
    orch.cfg.dataset.dataset_name = "organcmnist"
    orch.cfg.architecture.name = "mini_cnn"
    orch.cfg.evaluation.n_samples = 16

    orch.paths = MagicMock()
    orch.paths.exports = Path("/mock/test_exports")
    orch.paths.best_model_path = Path("/mock/best_model.pth")
    orch.paths.logs = Path("/mock/logs")

    orch.get_device.return_value = "cpu"
    orch.run_logger = MagicMock()

    return orch


# OPTIMIZATION PHASE TESTS
@pytest.mark.unit
@patch("orchard.pipeline.phases.run_optimization")
@patch("orchard.pipeline.phases.log_optimization_summary")
def test_run_optimization_phase_returns_study_and_path(
    _mock_log_summary, mock_run_opt, mock_orchestrator, tmp_path
):
    """Test run_optimization_phase returns (study, config_path)."""
    mock_study = MagicMock()
    mock_run_opt.return_value = mock_study

    # Simulate best_config.yaml exists (created by orchestrator)
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    best_config = reports_dir / "best_config.yaml"
    best_config.write_text("test: config")
    mock_orchestrator.paths.reports = reports_dir

    study, config_path = run_optimization_phase(mock_orchestrator)

    assert study is mock_study
    assert config_path == best_config
    mock_run_opt.assert_called_once()


@pytest.mark.unit
@patch("orchard.pipeline.phases.run_optimization")
@patch("orchard.pipeline.phases.log_optimization_summary")
def test_run_optimization_phase_with_custom_config(
    _mock_log_summary, mock_run_opt, mock_orchestrator, tmp_path
):
    """Test run_optimization_phase uses provided config override."""
    custom_cfg = MagicMock()
    mock_study = MagicMock()
    mock_run_opt.return_value = mock_study

    # No best_config.yaml (not exported)
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    mock_orchestrator.paths.reports = reports_dir

    _study, _config_path = run_optimization_phase(mock_orchestrator, cfg=custom_cfg)

    call_args = mock_run_opt.call_args
    assert call_args.kwargs["cfg"] is custom_cfg


@pytest.mark.unit
@patch("orchard.pipeline.phases.run_optimization")
@patch("orchard.pipeline.phases.log_optimization_summary")
def test_run_optimization_phase_logs_summary(
    mock_log_summary, mock_run_opt, mock_orchestrator, tmp_path
):
    """Test run_optimization_phase calls log_optimization_summary."""
    mock_run_opt.return_value = MagicMock()

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    mock_orchestrator.paths.reports = reports_dir

    run_optimization_phase(mock_orchestrator)

    mock_log_summary.assert_called_once()


@pytest.mark.unit
@patch("orchard.pipeline.phases.run_optimization")
@patch("orchard.pipeline.phases.log_optimization_summary")
def test_run_optimization_phase_handles_none_config_path(
    _mock_log_summary, mock_run_opt, mock_orchestrator, tmp_path
):
    """Test run_optimization_phase returns None when best_config.yaml doesn't exist."""
    mock_run_opt.return_value = MagicMock()

    # No best_config.yaml exists
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    mock_orchestrator.paths.reports = reports_dir

    _study, config_path = run_optimization_phase(mock_orchestrator)

    assert config_path is None


# TRAINING PHASE TESTS
@pytest.mark.unit
@patch("orchard.pipeline.phases.DatasetRegistryWrapper")
@patch("orchard.pipeline.phases.load_dataset")
@patch("orchard.pipeline.phases.get_dataloaders")
@patch("orchard.pipeline.phases.show_samples_for_dataset")
@patch("orchard.pipeline.phases.get_model")
@patch("orchard.pipeline.phases.get_criterion")
@patch("orchard.pipeline.phases.get_optimizer")
@patch("orchard.pipeline.phases.get_scheduler")
@patch("orchard.pipeline.phases.ModelTrainer")
@patch("orchard.pipeline.phases.run_final_evaluation")
@patch("orchard.pipeline.phases.get_augmentations_description")
def test_run_training_phase_returns_expected_tuple(
    mock_aug_desc,
    mock_final_eval,
    mock_trainer_cls,
    _mock_get_scheduler,
    _mock_get_optimizer,
    _mock_get_criterion,
    mock_get_model,
    _mock_show_samples,
    mock_get_loaders,
    _mock_load_dataset,
    mock_registry,
    mock_orchestrator,
):
    """Test run_training_phase returns all expected components."""
    mock_registry.return_value.get_dataset.return_value = MagicMock(classes=["a", "b"])
    mock_get_loaders.return_value = (MagicMock(), MagicMock(), MagicMock())
    mock_model = MagicMock()
    mock_get_model.return_value = mock_model

    mock_trainer = MagicMock()
    mock_trainer.train.return_value = (Path("/mock/best.pth"), [0.5, 0.4], [{"accuracy": 0.9}])
    mock_trainer_cls.return_value = mock_trainer

    mock_final_eval.return_value = (0.85, 0.90, 0.92)
    mock_aug_desc.return_value = "test_aug"

    result = run_training_phase(mock_orchestrator)

    assert isinstance(result, TrainingResult)
    assert len(result) == 7
    assert result.best_model_path == Path("/mock/best.pth")
    assert result.train_losses == [0.5, 0.4]
    assert result.macro_f1 == pytest.approx(0.85)
    assert result.test_acc == pytest.approx(0.90)
    assert result.test_auc == pytest.approx(0.92)


@pytest.mark.unit
@patch("orchard.pipeline.phases.DatasetRegistryWrapper")
@patch("orchard.pipeline.phases.load_dataset")
@patch("orchard.pipeline.phases.get_dataloaders")
@patch("orchard.pipeline.phases.show_samples_for_dataset")
@patch("orchard.pipeline.phases.get_model")
@patch("orchard.pipeline.phases.get_criterion")
@patch("orchard.pipeline.phases.get_optimizer")
@patch("orchard.pipeline.phases.get_scheduler")
@patch("orchard.pipeline.phases.ModelTrainer")
@patch("orchard.pipeline.phases.run_final_evaluation")
@patch("orchard.pipeline.phases.get_augmentations_description")
def test_run_training_phase_with_custom_config(
    mock_aug_desc,
    mock_final_eval,
    mock_trainer_cls,
    _mock_get_scheduler,
    _mock_get_optimizer,
    _mock_get_criterion,
    mock_get_model,
    _mock_show_samples,
    mock_get_loaders,
    _mock_load_dataset,
    mock_registry,
    mock_orchestrator,
):
    """Test run_training_phase uses provided config override."""
    custom_cfg = MagicMock()
    custom_cfg.dataset.resolution = 224
    custom_cfg.dataset.metadata.name = "bloodmnist"
    custom_cfg.dataset.dataset_name = "bloodmnist"
    custom_cfg.architecture.name = "resnet"
    custom_cfg.evaluation.n_samples = 8

    mock_registry.return_value.get_dataset.return_value = MagicMock(classes=["a", "b"])
    mock_get_loaders.return_value = (MagicMock(), MagicMock(), MagicMock())
    mock_get_model.return_value = MagicMock()

    mock_trainer = MagicMock()
    mock_trainer.train.return_value = (Path("/mock/best.pth"), [], [])
    mock_trainer_cls.return_value = mock_trainer

    mock_final_eval.return_value = (0.8, 0.85, 0.90)
    mock_aug_desc.return_value = ""

    run_training_phase(mock_orchestrator, cfg=custom_cfg)

    # Verify custom config was used
    mock_registry.assert_called_with(resolution=224)


# EXPORT PHASE TESTS
@pytest.mark.unit
def test_run_export_phase_returns_none_when_no_export_config(mock_orchestrator):
    """Test run_export_phase returns None when export config is absent."""
    mock_orchestrator.cfg.export = None
    result = run_export_phase(
        mock_orchestrator,
        checkpoint_path=Path("/mock/model.pth"),
    )

    assert result is None


@pytest.mark.unit
@patch("orchard.pipeline.phases.validate_export")
@patch("orchard.pipeline.phases.get_model")
@patch("orchard.pipeline.phases.export_to_onnx")
def test_run_export_phase_exports_onnx(
    mock_export_onnx, mock_get_model, mock_validate, mock_orchestrator
):
    """Test run_export_phase calls export_to_onnx with correct parameters."""
    mock_model = MagicMock()
    mock_get_model.return_value = mock_model

    checkpoint_path = Path("/mock/model.pth")

    result = run_export_phase(
        mock_orchestrator,
        checkpoint_path=checkpoint_path,
    )

    assert result == mock_orchestrator.paths.exports / "model.onnx"
    mock_export_onnx.assert_called_once()
    call_kwargs = mock_export_onnx.call_args.kwargs
    assert call_kwargs["checkpoint_path"] == checkpoint_path
    assert call_kwargs["input_shape"] == (3, 28, 28)
    mock_validate.assert_called_once()


@pytest.mark.unit
@patch("orchard.pipeline.phases.validate_export")
@patch("orchard.pipeline.phases.get_model")
@patch("orchard.pipeline.phases.export_to_onnx")
def test_run_export_phase_grayscale_input(
    mock_export_onnx, _mock_get_model, _mock_validate, mock_orchestrator
):
    """Test run_export_phase determines input channels from config."""
    mock_orchestrator.cfg.dataset.force_rgb = False
    mock_orchestrator.cfg.dataset.effective_in_channels = 1
    mock_orchestrator.cfg.dataset.resolution = 224

    run_export_phase(
        mock_orchestrator,
        checkpoint_path=Path("/mock/model.pth"),
    )

    call_kwargs = mock_export_onnx.call_args.kwargs
    assert call_kwargs["input_shape"] == (1, 224, 224)


@pytest.mark.unit
@patch("orchard.pipeline.phases.validate_export")
@patch("orchard.pipeline.phases.get_model")
@patch("orchard.pipeline.phases.export_to_onnx")
def test_run_export_phase_with_custom_config(
    mock_export_onnx, _mock_get_model, _mock_validate, mock_orchestrator
):
    """Test run_export_phase uses provided config override."""
    custom_cfg = MagicMock()
    custom_cfg.dataset.resolution = 64
    custom_cfg.dataset.effective_in_channels = 3

    run_export_phase(
        mock_orchestrator,
        checkpoint_path=Path("/mock/model.pth"),
        cfg=custom_cfg,
    )

    call_kwargs = mock_export_onnx.call_args.kwargs
    assert call_kwargs["input_shape"] == (3, 64, 64)


@pytest.mark.unit
@patch("orchard.pipeline.phases.validate_export")
@patch("orchard.pipeline.phases.get_model")
@patch("orchard.pipeline.phases.export_to_onnx")
def test_run_export_phase_logs_output_path(
    _mock_export_onnx, _mock_get_model, _mock_validate, mock_orchestrator
):
    """Test run_export_phase logs the export path."""
    run_export_phase(
        mock_orchestrator,
        checkpoint_path=Path("/mock/model.pth"),
    )

    mock_orchestrator.run_logger.info.assert_called()


@pytest.mark.unit
@patch("orchard.pipeline.phases.benchmark_onnx_inference")
@patch("orchard.pipeline.phases.validate_export")
@patch("orchard.pipeline.phases.get_model")
@patch("orchard.pipeline.phases.export_to_onnx")
def test_run_export_phase_benchmark(
    _mock_export, _mock_get_model, _mock_validate, mock_benchmark, mock_orchestrator
):
    """Test run_export_phase calls benchmark when enabled."""
    mock_orchestrator.cfg.export.benchmark = True
    mock_orchestrator.cfg.training.seed = 42

    run_export_phase(
        mock_orchestrator,
        checkpoint_path=Path("/mock/model.pth"),
    )

    mock_benchmark.assert_called_once()
    call_kwargs = mock_benchmark.call_args.kwargs
    assert call_kwargs["seed"] == 42


@pytest.mark.unit
@patch("orchard.pipeline.phases.benchmark_onnx_inference")
@patch("orchard.pipeline.phases.validate_export")
@patch("orchard.pipeline.phases.get_model")
@patch("orchard.pipeline.phases.export_to_onnx")
def test_run_export_phase_no_benchmark_by_default(
    _mock_export, _mock_get_model, _mock_validate, mock_benchmark, mock_orchestrator
):
    """Test run_export_phase skips benchmark when disabled."""
    mock_orchestrator.cfg.export.benchmark = False

    run_export_phase(
        mock_orchestrator,
        checkpoint_path=Path("/mock/model.pth"),
    )

    mock_benchmark.assert_not_called()


@pytest.mark.unit
@patch("orchard.pipeline.phases.benchmark_onnx_inference")
@patch("orchard.pipeline.phases.quantize_model")
@patch("orchard.pipeline.phases.validate_export")
@patch("orchard.pipeline.phases.get_model")
@patch("orchard.pipeline.phases.export_to_onnx")
def test_run_export_phase_quantize_with_benchmark(
    _mock_export,
    _mock_get_model,
    _mock_validate,
    mock_quantize,
    mock_benchmark,
    mock_orchestrator,
):
    """Test run_export_phase benchmarks quantized model when both enabled."""
    mock_orchestrator.cfg.export.quantize = True
    mock_orchestrator.cfg.export.quantization_backend = "fbgemm"
    mock_orchestrator.cfg.export.benchmark = True
    mock_orchestrator.cfg.training.seed = 42
    mock_quantize.return_value = Path("/mock/test_exports/model_quantized.onnx")

    run_export_phase(
        mock_orchestrator,
        checkpoint_path=Path("/mock/model.pth"),
    )

    mock_quantize.assert_called_once()
    # Two benchmark calls: original + quantized
    assert mock_benchmark.call_count == 2


@pytest.mark.unit
@patch("orchard.pipeline.phases.quantize_model")
@patch("orchard.pipeline.phases.validate_export")
@patch("orchard.pipeline.phases.get_model")
@patch("orchard.pipeline.phases.export_to_onnx")
def test_run_export_phase_quantize_logs_output(
    _mock_export,
    _mock_get_model,
    _mock_validate,
    mock_quantize,
    mock_orchestrator,
):
    """Test run_export_phase logs quantized model path."""
    mock_orchestrator.cfg.export.quantize = True
    mock_orchestrator.cfg.export.quantization_backend = "qnnpack"
    mock_orchestrator.cfg.export.benchmark = False
    mock_quantize.return_value = Path("/mock/test_exports/model_quantized.onnx")

    run_export_phase(
        mock_orchestrator,
        checkpoint_path=Path("/mock/model.pth"),
    )

    mock_quantize.assert_called_once()
    assert mock_quantize.call_args.kwargs["backend"] == "qnnpack"


@pytest.mark.unit
@patch("orchard.pipeline.phases.logger")
@patch("orchard.pipeline.phases.validate_export", return_value=False)
@patch("orchard.pipeline.phases.get_model")
@patch("orchard.pipeline.phases.export_to_onnx")
def test_run_export_phase_validation_failure_logs_warning(
    _mock_export, _mock_get_model, mock_validate, mock_logger, mock_orchestrator
):
    """Test run_export_phase logs warning when numerical validation fails."""
    mock_orchestrator.cfg.export.benchmark = False
    mock_orchestrator.cfg.export.quantize = False

    run_export_phase(
        mock_orchestrator,
        checkpoint_path=Path("/mock/model.pth"),
    )

    mock_validate.assert_called_once()
    warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
    assert any("Numerical validation failed" in w for w in warning_calls)


# OPTIMIZATION PHASE: KWARGS AND GUARDS
@pytest.mark.unit
@patch("orchard.pipeline.phases.run_optimization")
@patch("orchard.pipeline.phases.log_optimization_summary")
def test_run_optimization_phase_asserts_run_logger(
    _mock_log_summary, mock_run_opt, mock_orchestrator, tmp_path
):
    """Test run_optimization_phase raises when run_logger is None."""
    mock_orchestrator.run_logger = None
    mock_run_opt.return_value = MagicMock()
    mock_orchestrator.paths.reports = tmp_path

    with pytest.raises(AssertionError):
        run_optimization_phase(mock_orchestrator)


@pytest.mark.unit
@patch("orchard.pipeline.phases.run_optimization")
@patch("orchard.pipeline.phases.log_optimization_summary")
def test_run_optimization_phase_asserts_paths(
    _mock_log_summary, mock_run_opt, mock_orchestrator, tmp_path
):
    """Test run_optimization_phase raises when paths is None."""
    mock_orchestrator.paths = None
    mock_run_opt.return_value = MagicMock()

    with pytest.raises(AssertionError):
        run_optimization_phase(mock_orchestrator)


@pytest.mark.unit
@patch("orchard.pipeline.phases.run_optimization")
@patch("orchard.pipeline.phases.log_optimization_summary")
def test_run_optimization_phase_forwards_all_kwargs(
    mock_log_summary, mock_run_opt, mock_orchestrator, tmp_path
):
    """Test run_optimization_phase passes device, paths, tracker to run_optimization."""
    mock_study = MagicMock()
    mock_run_opt.return_value = mock_study
    mock_tracker = MagicMock()

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    mock_orchestrator.paths.reports = reports_dir

    run_optimization_phase(mock_orchestrator, tracker=mock_tracker)

    # Verify run_optimization kwargs
    opt_kwargs = mock_run_opt.call_args.kwargs
    assert opt_kwargs["cfg"] is mock_orchestrator.cfg
    assert opt_kwargs["device"] == mock_orchestrator.get_device()
    assert opt_kwargs["paths"] is mock_orchestrator.paths
    assert opt_kwargs["tracker"] is mock_tracker

    # Verify log_optimization_summary kwargs
    log_kwargs = mock_log_summary.call_args.kwargs
    assert log_kwargs["study"] is mock_study
    assert log_kwargs["cfg"] is mock_orchestrator.cfg
    assert log_kwargs["device"] == mock_orchestrator.get_device()
    assert log_kwargs["paths"] is mock_orchestrator.paths


# TRAINING PHASE: KWARGS AND GUARDS
_TRAINING_PATCHES = [
    "orchard.pipeline.phases.get_augmentations_description",
    "orchard.pipeline.phases.run_final_evaluation",
    "orchard.pipeline.phases.ModelTrainer",
    "orchard.pipeline.phases.get_scheduler",
    "orchard.pipeline.phases.get_optimizer",
    "orchard.pipeline.phases.get_criterion",
    "orchard.pipeline.phases.get_model",
    "orchard.pipeline.phases.show_samples_for_dataset",
    "orchard.pipeline.phases.get_dataloaders",
    "orchard.pipeline.phases.load_dataset",
    "orchard.pipeline.phases.DatasetRegistryWrapper",
]


def _setup_training_mocks(
    mock_registry,
    mock_load_dataset,
    mock_get_loaders,
    mock_show_samples,
    mock_get_model,
    mock_get_criterion,
    mock_get_optimizer,
    mock_get_scheduler,
    mock_trainer_cls,
    mock_final_eval,
    mock_aug_desc,
):
    """Configure mocks for run_training_phase tests."""
    ds_meta = MagicMock(classes=["a", "b"], num_classes=2)
    mock_registry.return_value.get_dataset.return_value = ds_meta

    train_loader = MagicMock()
    val_loader = MagicMock()
    test_loader = MagicMock()
    mock_get_loaders.return_value = (train_loader, val_loader, test_loader)

    mock_model = MagicMock()
    mock_get_model.return_value = mock_model

    mock_criterion = MagicMock()
    mock_get_criterion.return_value = mock_criterion

    mock_optimizer = MagicMock()
    mock_get_optimizer.return_value = mock_optimizer

    mock_scheduler_obj = MagicMock()
    mock_get_scheduler.return_value = mock_scheduler_obj

    mock_trainer = MagicMock()
    mock_trainer.train.return_value = (Path("/mock/best.pth"), [0.5], [{"acc": 0.9}])
    mock_trainer_cls.return_value = mock_trainer

    mock_final_eval.return_value = (0.85, 0.90, 0.92)
    mock_aug_desc.return_value = "test_aug"

    return {
        "ds_meta": ds_meta,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "model": mock_model,
        "criterion": mock_criterion,
        "optimizer": mock_optimizer,
        "scheduler": mock_scheduler_obj,
        "trainer": mock_trainer,
    }


@pytest.mark.unit
@patch("orchard.pipeline.phases.DatasetRegistryWrapper")
@patch("orchard.pipeline.phases.load_dataset")
@patch("orchard.pipeline.phases.get_dataloaders")
@patch("orchard.pipeline.phases.show_samples_for_dataset")
@patch("orchard.pipeline.phases.get_model")
@patch("orchard.pipeline.phases.get_criterion")
@patch("orchard.pipeline.phases.get_optimizer")
@patch("orchard.pipeline.phases.get_scheduler")
@patch("orchard.pipeline.phases.ModelTrainer")
@patch("orchard.pipeline.phases.run_final_evaluation")
@patch("orchard.pipeline.phases.get_augmentations_description")
def test_run_training_phase_asserts_run_logger(
    _m1,
    _m2,
    _m3,
    _m4,
    _m5,
    _m6,
    _m7,
    _m8,
    _m9,
    _m10,
    _m11,
    mock_orchestrator,
):
    """Test run_training_phase raises when run_logger is None."""
    mock_orchestrator.run_logger = None

    with pytest.raises(AssertionError):
        run_training_phase(mock_orchestrator)


@pytest.mark.unit
@patch("orchard.pipeline.phases.DatasetRegistryWrapper")
@patch("orchard.pipeline.phases.load_dataset")
@patch("orchard.pipeline.phases.get_dataloaders")
@patch("orchard.pipeline.phases.show_samples_for_dataset")
@patch("orchard.pipeline.phases.get_model")
@patch("orchard.pipeline.phases.get_criterion")
@patch("orchard.pipeline.phases.get_optimizer")
@patch("orchard.pipeline.phases.get_scheduler")
@patch("orchard.pipeline.phases.ModelTrainer")
@patch("orchard.pipeline.phases.run_final_evaluation")
@patch("orchard.pipeline.phases.get_augmentations_description")
def test_run_training_phase_asserts_paths(
    _m1,
    _m2,
    _m3,
    _m4,
    _m5,
    _m6,
    _m7,
    _m8,
    _m9,
    _m10,
    _m11,
    mock_orchestrator,
):
    """Test run_training_phase raises when paths is None."""
    mock_orchestrator.paths = None

    with pytest.raises(AssertionError):
        run_training_phase(mock_orchestrator)


@pytest.mark.unit
@patch("orchard.pipeline.phases.DatasetRegistryWrapper")
@patch("orchard.pipeline.phases.load_dataset")
@patch("orchard.pipeline.phases.get_dataloaders")
@patch("orchard.pipeline.phases.show_samples_for_dataset")
@patch("orchard.pipeline.phases.get_model")
@patch("orchard.pipeline.phases.get_criterion")
@patch("orchard.pipeline.phases.get_optimizer")
@patch("orchard.pipeline.phases.get_scheduler")
@patch("orchard.pipeline.phases.ModelTrainer")
@patch("orchard.pipeline.phases.run_final_evaluation")
@patch("orchard.pipeline.phases.get_augmentations_description")
def test_run_training_phase_verifies_all_kwargs(
    mock_aug_desc,
    mock_final_eval,
    mock_trainer_cls,
    mock_get_scheduler,
    mock_get_optimizer,
    mock_get_criterion,
    mock_get_model,
    mock_show_samples,
    mock_get_loaders,
    mock_load_dataset,
    mock_registry,
    mock_orchestrator,
):
    """Test run_training_phase passes correct kwargs to every dependency."""
    refs = _setup_training_mocks(
        mock_registry,
        mock_load_dataset,
        mock_get_loaders,
        mock_show_samples,
        mock_get_model,
        mock_get_criterion,
        mock_get_optimizer,
        mock_get_scheduler,
        mock_trainer_cls,
        mock_final_eval,
        mock_aug_desc,
    )
    cfg = mock_orchestrator.cfg
    cfg.training.weighted_loss = False

    result = run_training_phase(mock_orchestrator)

    # DatasetRegistryWrapper
    mock_registry.assert_called_once_with(resolution=cfg.dataset.resolution)
    mock_registry.return_value.get_dataset.assert_called_once_with(cfg.dataset.dataset_name.lower())

    # load_dataset
    mock_load_dataset.assert_called_once_with(refs["ds_meta"])

    # get_dataloaders
    mock_get_loaders.assert_called_once_with(
        mock_load_dataset.return_value,
        cfg.dataset,
        cfg.training,
        cfg.augmentation,
        cfg.num_workers,
    )

    # show_samples_for_dataset
    show_kwargs = mock_show_samples.call_args.kwargs
    assert show_kwargs["loader"] is refs["train_loader"]
    assert show_kwargs["dataset_name"] == cfg.dataset.dataset_name
    assert show_kwargs["run_paths"] is mock_orchestrator.paths
    assert show_kwargs["mean"] == cfg.dataset.mean
    assert show_kwargs["std"] == cfg.dataset.std
    assert show_kwargs["arch_name"] == cfg.architecture.name
    assert show_kwargs["fig_dpi"] == cfg.evaluation.fig_dpi
    assert show_kwargs["num_samples"] == cfg.evaluation.n_samples
    assert show_kwargs["resolution"] == cfg.dataset.resolution

    # get_model
    model_kwargs = mock_get_model.call_args.kwargs
    assert model_kwargs["device"] == mock_orchestrator.get_device()
    assert model_kwargs["dataset_cfg"] is cfg.dataset
    assert model_kwargs["arch_cfg"] is cfg.architecture

    # get_criterion (no class_weights since weighted_loss=False)
    mock_get_criterion.assert_called_once_with(cfg.training, class_weights=None)

    # get_optimizer
    mock_get_optimizer.assert_called_once_with(refs["model"], cfg.training)

    # get_scheduler
    mock_get_scheduler.assert_called_once_with(refs["optimizer"], cfg.training)

    # ModelTrainer
    trainer_kwargs = mock_trainer_cls.call_args.kwargs
    assert trainer_kwargs["model"] is refs["model"]
    assert trainer_kwargs["train_loader"] is refs["train_loader"]
    assert trainer_kwargs["val_loader"] is refs["val_loader"]
    assert trainer_kwargs["optimizer"] is refs["optimizer"]
    assert trainer_kwargs["scheduler"] is refs["scheduler"]
    assert trainer_kwargs["criterion"] is refs["criterion"]
    assert trainer_kwargs["device"] == mock_orchestrator.get_device()
    assert trainer_kwargs["training"] is cfg.training
    assert trainer_kwargs["output_path"] is mock_orchestrator.paths.best_model_path
    assert trainer_kwargs["tracker"] is None

    # run_final_evaluation
    eval_kwargs = mock_final_eval.call_args.kwargs
    assert eval_kwargs["model"] is refs["model"]
    assert eval_kwargs["test_loader"] is refs["test_loader"]
    assert eval_kwargs["train_losses"] == [0.5]
    assert eval_kwargs["val_metrics_history"] == [{"acc": 0.9}]
    assert eval_kwargs["class_names"] == refs["ds_meta"].classes
    assert eval_kwargs["paths"] is mock_orchestrator.paths
    assert eval_kwargs["training"] is cfg.training
    assert eval_kwargs["dataset"] is cfg.dataset
    assert eval_kwargs["augmentation"] is cfg.augmentation
    assert eval_kwargs["evaluation"] is cfg.evaluation
    assert eval_kwargs["arch_name"] == cfg.architecture.name
    assert eval_kwargs["aug_info"] == mock_aug_desc.return_value
    assert eval_kwargs["tracker"] is None

    # get_augmentations_description
    mock_aug_desc.assert_called_once_with(
        cfg.augmentation,
        cfg.dataset.img_size,
        cfg.training.mixup_alpha,
        ds_meta=refs["ds_meta"],
    )

    # Result integrity
    assert result.model is refs["model"]
    assert result.val_metrics == [{"acc": 0.9}]


@pytest.mark.unit
@patch("orchard.pipeline.phases.compute_class_weights")
@patch("orchard.pipeline.phases.DatasetRegistryWrapper")
@patch("orchard.pipeline.phases.load_dataset")
@patch("orchard.pipeline.phases.get_dataloaders")
@patch("orchard.pipeline.phases.show_samples_for_dataset")
@patch("orchard.pipeline.phases.get_model")
@patch("orchard.pipeline.phases.get_criterion")
@patch("orchard.pipeline.phases.get_optimizer")
@patch("orchard.pipeline.phases.get_scheduler")
@patch("orchard.pipeline.phases.ModelTrainer")
@patch("orchard.pipeline.phases.run_final_evaluation")
@patch("orchard.pipeline.phases.get_augmentations_description")
def test_run_training_phase_weighted_loss(
    mock_aug_desc,
    mock_final_eval,
    mock_trainer_cls,
    mock_get_scheduler,
    mock_get_optimizer,
    mock_get_criterion,
    mock_get_model,
    mock_show_samples,
    mock_get_loaders,
    mock_load_dataset,
    mock_registry,
    mock_compute_weights,
    mock_orchestrator,
):
    """Test run_training_phase computes class weights when weighted_loss is True."""
    import numpy as np

    ds_meta = MagicMock(classes=["a", "b"], num_classes=2)
    mock_registry.return_value.get_dataset.return_value = ds_meta

    train_loader = MagicMock()
    train_labels = np.array([0, 1, 0, 1])
    train_loader.dataset.labels.flatten.return_value = train_labels
    mock_get_loaders.return_value = (train_loader, MagicMock(), MagicMock())
    mock_get_model.return_value = MagicMock()

    mock_weights = MagicMock()
    mock_compute_weights.return_value = mock_weights

    mock_trainer = MagicMock()
    mock_trainer.train.return_value = (Path("/mock/best.pth"), [], [])
    mock_trainer_cls.return_value = mock_trainer
    mock_final_eval.return_value = (0.8, 0.85, 0.90)
    mock_aug_desc.return_value = ""

    mock_orchestrator.cfg.training.weighted_loss = True

    run_training_phase(mock_orchestrator)

    # compute_class_weights called with correct args
    mock_compute_weights.assert_called_once_with(
        train_labels, ds_meta.num_classes, mock_orchestrator.get_device()
    )
    # class_weights passed to get_criterion
    mock_get_criterion.assert_called_once_with(
        mock_orchestrator.cfg.training, class_weights=mock_weights
    )


@pytest.mark.unit
@patch("orchard.pipeline.phases.DatasetRegistryWrapper")
@patch("orchard.pipeline.phases.load_dataset")
@patch("orchard.pipeline.phases.get_dataloaders")
@patch("orchard.pipeline.phases.show_samples_for_dataset")
@patch("orchard.pipeline.phases.get_model")
@patch("orchard.pipeline.phases.get_criterion")
@patch("orchard.pipeline.phases.get_optimizer")
@patch("orchard.pipeline.phases.get_scheduler")
@patch("orchard.pipeline.phases.ModelTrainer")
@patch("orchard.pipeline.phases.run_final_evaluation")
@patch("orchard.pipeline.phases.get_augmentations_description")
def test_run_training_phase_passes_tracker(
    mock_aug_desc,
    mock_final_eval,
    mock_trainer_cls,
    mock_get_scheduler,
    mock_get_optimizer,
    mock_get_criterion,
    mock_get_model,
    mock_show_samples,
    mock_get_loaders,
    mock_load_dataset,
    mock_registry,
    mock_orchestrator,
):
    """Test run_training_phase forwards tracker to ModelTrainer and run_final_evaluation."""
    refs = _setup_training_mocks(
        mock_registry,
        mock_load_dataset,
        mock_get_loaders,
        mock_show_samples,
        mock_get_model,
        mock_get_criterion,
        mock_get_optimizer,
        mock_get_scheduler,
        mock_trainer_cls,
        mock_final_eval,
        mock_aug_desc,
    )
    mock_orchestrator.cfg.training.weighted_loss = False

    mock_tracker = MagicMock()
    run_training_phase(mock_orchestrator, tracker=mock_tracker)

    assert mock_trainer_cls.call_args.kwargs["tracker"] is mock_tracker
    assert mock_final_eval.call_args.kwargs["tracker"] is mock_tracker


# EXPORT PHASE: KWARGS AND GUARDS
@pytest.mark.unit
@patch("orchard.pipeline.phases.validate_export")
@patch("orchard.pipeline.phases.get_model")
@patch("orchard.pipeline.phases.export_to_onnx")
def test_run_export_phase_asserts_run_logger(
    _mock_export, _mock_get_model, _mock_validate, mock_orchestrator
):
    """Test run_export_phase raises when run_logger is None."""
    mock_orchestrator.run_logger = None

    with pytest.raises(AssertionError):
        run_export_phase(mock_orchestrator, checkpoint_path=Path("/mock/model.pth"))


@pytest.mark.unit
@patch("orchard.pipeline.phases.validate_export")
@patch("orchard.pipeline.phases.get_model")
@patch("orchard.pipeline.phases.export_to_onnx")
def test_run_export_phase_asserts_paths(
    _mock_export, _mock_get_model, _mock_validate, mock_orchestrator
):
    """Test run_export_phase raises when paths is None."""
    mock_orchestrator.paths = None

    with pytest.raises(AssertionError):
        run_export_phase(mock_orchestrator, checkpoint_path=Path("/mock/model.pth"))


@pytest.mark.unit
@patch("orchard.pipeline.phases.validate_export")
@patch("orchard.pipeline.phases.get_model")
@patch("orchard.pipeline.phases.export_to_onnx")
def test_run_export_phase_get_model_cpu_no_verbose(
    _mock_export, mock_get_model, _mock_validate, mock_orchestrator
):
    """Test run_export_phase loads model on CPU with verbose=False."""
    import torch

    mock_get_model.return_value = MagicMock()

    run_export_phase(mock_orchestrator, checkpoint_path=Path("/mock/model.pth"))

    model_kwargs = mock_get_model.call_args.kwargs
    assert model_kwargs["device"] == torch.device("cpu")
    assert model_kwargs["verbose"] is False
    assert model_kwargs["dataset_cfg"] is mock_orchestrator.cfg.dataset
    assert model_kwargs["arch_cfg"] is mock_orchestrator.cfg.architecture


@pytest.mark.unit
@patch("orchard.pipeline.phases.validate_export")
@patch("orchard.pipeline.phases.get_model")
@patch("orchard.pipeline.phases.export_to_onnx")
def test_run_export_phase_all_export_to_onnx_kwargs(
    mock_export_onnx, mock_get_model, _mock_validate, mock_orchestrator
):
    """Test run_export_phase passes all kwargs to export_to_onnx."""
    mock_model = MagicMock()
    mock_get_model.return_value = mock_model
    checkpoint = Path("/mock/model.pth")

    mock_orchestrator.cfg.export.opset_version = 17
    mock_orchestrator.cfg.export.dynamic_axes = {"input": {0: "batch"}}
    mock_orchestrator.cfg.export.do_constant_folding = True
    mock_orchestrator.cfg.export.validate_export = True

    run_export_phase(mock_orchestrator, checkpoint_path=checkpoint)

    kw = mock_export_onnx.call_args.kwargs
    assert kw["model"] is mock_model
    assert kw["checkpoint_path"] == checkpoint
    assert kw["output_path"] == mock_orchestrator.paths.exports / "model.onnx"
    assert kw["input_shape"] == (3, 28, 28)
    assert kw["opset_version"] == 17
    assert kw["dynamic_axes"] == {"input": {0: "batch"}}
    assert kw["do_constant_folding"] is True
    assert kw["validate"] is True


@pytest.mark.unit
@patch("orchard.pipeline.phases.validate_export")
@patch("orchard.pipeline.phases.get_model")
@patch("orchard.pipeline.phases.export_to_onnx")
def test_run_export_phase_validate_export_kwargs(
    _mock_export, mock_get_model, mock_validate, mock_orchestrator
):
    """Test run_export_phase passes all kwargs to validate_export."""
    mock_model = MagicMock()
    mock_get_model.return_value = mock_model

    mock_orchestrator.cfg.export.validate_export = True
    mock_orchestrator.cfg.export.validation_samples = 5
    mock_orchestrator.cfg.export.max_deviation = 1e-5
    mock_orchestrator.cfg.export.benchmark = False
    mock_orchestrator.cfg.export.quantize = False

    run_export_phase(mock_orchestrator, checkpoint_path=Path("/mock/model.pth"))

    kw = mock_validate.call_args.kwargs
    assert kw["pytorch_model"] is mock_model
    assert kw["onnx_path"] == mock_orchestrator.paths.exports / "model.onnx"
    assert kw["input_shape"] == (3, 28, 28)
    assert kw["num_samples"] == 5
    assert kw["max_deviation"] == pytest.approx(1e-5)


@pytest.mark.unit
@patch("orchard.pipeline.phases.benchmark_onnx_inference")
@patch("orchard.pipeline.phases.validate_export")
@patch("orchard.pipeline.phases.get_model")
@patch("orchard.pipeline.phases.export_to_onnx")
def test_run_export_phase_benchmark_passes_all_kwargs(
    _mock_export, _mock_get_model, _mock_validate, mock_benchmark, mock_orchestrator
):
    """Test run_export_phase passes onnx_path, input_shape to benchmark."""
    mock_orchestrator.cfg.export.benchmark = True
    mock_orchestrator.cfg.export.quantize = False
    mock_orchestrator.cfg.training.seed = 42

    run_export_phase(mock_orchestrator, checkpoint_path=Path("/mock/model.pth"))

    kw = mock_benchmark.call_args.kwargs
    assert kw["onnx_path"] == mock_orchestrator.paths.exports / "model.onnx"
    assert kw["input_shape"] == (3, 28, 28)
    assert kw["seed"] == 42


@pytest.mark.unit
@patch("orchard.pipeline.phases.benchmark_onnx_inference")
@patch("orchard.pipeline.phases.quantize_model")
@patch("orchard.pipeline.phases.validate_export")
@patch("orchard.pipeline.phases.get_model")
@patch("orchard.pipeline.phases.export_to_onnx")
def test_run_export_phase_quantized_benchmark_uses_quantized_path(
    _mock_export, _mock_get_model, _mock_validate, mock_quantize, mock_benchmark, mock_orchestrator
):
    """Test second benchmark call uses quantized_path, not onnx_path."""
    quantized = Path("/mock/test_exports/model_quantized.onnx")
    mock_quantize.return_value = quantized

    mock_orchestrator.cfg.export.quantize = True
    mock_orchestrator.cfg.export.quantization_backend = "fbgemm"
    mock_orchestrator.cfg.export.benchmark = True
    mock_orchestrator.cfg.training.seed = 42

    run_export_phase(mock_orchestrator, checkpoint_path=Path("/mock/model.pth"))

    # Second benchmark call should use quantized path
    second_call = mock_benchmark.call_args_list[1]
    assert second_call.kwargs["onnx_path"] == quantized
    assert second_call.kwargs["input_shape"] == (3, 28, 28)
    assert second_call.kwargs["seed"] == 42


@pytest.mark.unit
@patch("orchard.pipeline.phases.quantize_model")
@patch("orchard.pipeline.phases.validate_export")
@patch("orchard.pipeline.phases.get_model")
@patch("orchard.pipeline.phases.export_to_onnx")
def test_run_export_phase_quantize_passes_onnx_path(
    _mock_export, _mock_get_model, _mock_validate, mock_quantize, mock_orchestrator
):
    """Test run_export_phase passes onnx_path to quantize_model."""
    mock_orchestrator.cfg.export.quantize = True
    mock_orchestrator.cfg.export.quantization_backend = "fbgemm"
    mock_orchestrator.cfg.export.benchmark = False
    mock_quantize.return_value = Path("/mock/quantized.onnx")

    run_export_phase(mock_orchestrator, checkpoint_path=Path("/mock/model.pth"))

    kw = mock_quantize.call_args.kwargs
    assert kw["onnx_path"] == mock_orchestrator.paths.exports / "model.onnx"
    assert kw["backend"] == "fbgemm"


@pytest.mark.unit
@patch("orchard.pipeline.phases.validate_export")
@patch("orchard.pipeline.phases.get_model")
@patch("orchard.pipeline.phases.export_to_onnx")
def test_run_export_phase_skips_validation_when_disabled(
    _mock_export, _mock_get_model, mock_validate, mock_orchestrator
):
    """Test run_export_phase skips validate_export when validate_export is False."""
    mock_orchestrator.cfg.export.validate_export = False
    mock_orchestrator.cfg.export.benchmark = False
    mock_orchestrator.cfg.export.quantize = False

    run_export_phase(mock_orchestrator, checkpoint_path=Path("/mock/model.pth"))

    mock_validate.assert_not_called()


@pytest.mark.unit
@patch("orchard.pipeline.phases.validate_export")
@patch("orchard.pipeline.phases.get_model")
@patch("orchard.pipeline.phases.export_to_onnx")
def test_run_export_phase_cfg_fallback_to_orchestrator(
    _mock_export, _mock_get_model, _mock_validate, mock_orchestrator
):
    """Test run_export_phase uses orchestrator.cfg when cfg=None."""
    run_export_phase(mock_orchestrator, checkpoint_path=Path("/mock/model.pth"))

    # The function should have used orchestrator.cfg (not None)
    _mock_get_model.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
