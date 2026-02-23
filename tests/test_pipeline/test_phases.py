"""
Test Suite for Pipeline Phase Functions.

Tests for run_optimization_phase, run_training_phase, and run_export_phase.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from orchard.pipeline.phases import (
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

    assert len(result) == 7
    best_path, losses, _metrics, _model, f1, acc, auc = result
    assert best_path == Path("/mock/best.pth")
    assert losses == [0.5, 0.4]
    assert f1 == pytest.approx(0.85)
    assert acc == pytest.approx(0.90)
    assert auc == pytest.approx(0.92)


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
def test_run_export_phase_returns_none_when_format_is_none(mock_orchestrator):
    """Test run_export_phase returns None when export_format is 'none'."""
    result = run_export_phase(
        mock_orchestrator,
        checkpoint_path=Path("/mock/model.pth"),
        export_format="none",
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
        export_format="onnx",
        opset_version=17,
    )

    assert result == mock_orchestrator.paths.exports / "model.onnx"
    mock_export_onnx.assert_called_once()
    call_kwargs = mock_export_onnx.call_args.kwargs
    assert call_kwargs["checkpoint_path"] == checkpoint_path
    assert call_kwargs["opset_version"] == 17
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
        export_format="onnx",
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
        export_format="onnx",
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
        export_format="onnx",
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
        export_format="onnx",
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
        export_format="onnx",
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
        export_format="onnx",
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
        export_format="onnx",
    )

    mock_quantize.assert_called_once()
    assert mock_quantize.call_args.kwargs["backend"] == "qnnpack"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
