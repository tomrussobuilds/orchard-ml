"""
Test Suite for Telemetry & Environment Reporting Engine.

Tests LogStyle constants, Reporter logging methods,
and summary logging functions.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import optuna
import pytest
import torch

from orchard.core.logger import (
    LogStyle,
    Reporter,
    log_optimization_header,
    log_optimization_summary,
    log_pipeline_summary,
    log_trial_start,
)


# LOGSTYLE: CONSTANTS
@pytest.mark.unit
def test_logstyle_heavy_separator():
    """Test LogStyle HEAVY separator is 80 characters."""
    assert len(LogStyle.HEAVY) == 80
    assert LogStyle.HEAVY == "━" * 80


@pytest.mark.unit
def test_logstyle_double_separator():
    """Test LogStyle DOUBLE separator is 80 characters."""
    assert len(LogStyle.DOUBLE) == 80
    assert LogStyle.DOUBLE == "═" * 80


@pytest.mark.unit
def test_logstyle_light_separator():
    """Test LogStyle LIGHT separator is 80 characters."""
    assert len(LogStyle.LIGHT) == 80
    assert LogStyle.LIGHT == "─" * 80


@pytest.mark.unit
def test_logstyle_symbols():
    """Test LogStyle symbols are defined correctly."""
    assert LogStyle.ARROW == "»"
    assert LogStyle.BULLET == "•"
    assert LogStyle.WARNING == "⚠"
    assert LogStyle.SUCCESS == "✓"


@pytest.mark.unit
def test_logstyle_indentation():
    """Test LogStyle indentation constants."""
    assert LogStyle.INDENT == "  "
    assert LogStyle.DOUBLE_INDENT == "    "
    assert len(LogStyle.DOUBLE_INDENT) == 2 * len(LogStyle.INDENT)


# REPORTER: INITIALIZATION
@pytest.mark.unit
def test_reporter_init():
    """Test Reporter can be instantiated."""
    reporter = Reporter()
    assert isinstance(reporter, Reporter)


# LOG OPTIMIZATION SUMMARY
@pytest.mark.unit
def test_log_optimization_summary_completed_trials():
    """Test log_optimization_summary with completed trials."""
    mock_logger = MagicMock()
    mock_cfg = MagicMock()
    mock_cfg.dataset.dataset_name = "bloodmnist"
    mock_cfg.optuna.search_space_preset = "default"
    mock_cfg.training.monitor_metric = "auc"
    mock_device = torch.device("cuda")
    mock_paths = MagicMock()
    mock_paths.root = "/mock/outputs"
    mock_study = MagicMock()
    mock_trial = MagicMock()
    mock_trial.state = optuna.trial.TrialState.COMPLETE
    mock_study.trials = [mock_trial, mock_trial, mock_trial]
    mock_study.best_value = 0.9876
    mock_study.best_trial.number = 5

    log_optimization_summary(
        study=mock_study,
        cfg=mock_cfg,
        device=mock_device,
        paths=mock_paths,
        logger_instance=mock_logger,
    )

    assert mock_logger.info.call_count > 8
    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)
    assert "3" in log_output


@pytest.mark.unit
def test_log_optimization_summary_no_completed_trials():
    """Test log_optimization_summary with no completed trials."""
    mock_logger = MagicMock()
    mock_cfg = MagicMock()
    mock_cfg.dataset.dataset_name = "test"
    mock_cfg.optuna.search_space_preset = "default"
    mock_cfg.training.monitor_metric = "auc"
    mock_device = torch.device("cpu")
    mock_paths = MagicMock()
    mock_study = MagicMock()
    mock_trial = MagicMock()
    mock_trial.state = optuna.trial.TrialState.PRUNED
    mock_study.trials = [mock_trial]

    log_optimization_summary(
        study=mock_study,
        cfg=mock_cfg,
        device=mock_device,
        paths=mock_paths,
        logger_instance=mock_logger,
    )
    assert mock_logger.warning.called


@pytest.mark.unit
def test_log_optimization_summary_with_failed_trials():
    """Test log_optimization_summary includes failed trial count."""
    mock_logger = MagicMock()
    mock_cfg = MagicMock()
    mock_cfg.dataset.dataset_name = "test"
    mock_cfg.optuna.search_space_preset = "default"
    mock_cfg.training.monitor_metric = "auc"
    mock_device = torch.device("cpu")
    mock_paths = MagicMock()
    mock_study = MagicMock()
    mock_trial_complete = MagicMock()
    mock_trial_complete.state = optuna.trial.TrialState.COMPLETE
    mock_trial_failed = MagicMock()
    mock_trial_failed.state = optuna.trial.TrialState.FAIL
    mock_study.trials = [mock_trial_complete, mock_trial_failed]
    mock_study.best_value = 0.95
    mock_study.best_trial.number = 0

    log_optimization_summary(
        study=mock_study,
        cfg=mock_cfg,
        device=mock_device,
        paths=mock_paths,
        logger_instance=mock_logger,
    )

    assert mock_logger.info.call_count > 0


# LOG OPTIMIZATION HEADER
@pytest.mark.unit
def test_log_optimization_header_basic():
    """Test log_optimization_header logs search configuration (no duplicate header)."""
    mock_logger = MagicMock()
    mock_cfg = MagicMock()
    mock_cfg.optuna.search_space_preset = "architecture_search"
    mock_cfg.optuna.n_trials = 50
    mock_cfg.optuna.epochs = 10
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.optuna.enable_pruning = True
    mock_cfg.optuna.enable_early_stopping = False
    mock_cfg.optuna.model_pool = None

    log_optimization_header(cfg=mock_cfg, logger_instance=mock_logger)

    # 9 lines: empty, dataset, model_search, search_space, trials, epochs/trial, metric, pruning, empty
    assert mock_logger.info.call_count == 9
    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)
    assert "architecture_search" in log_output
    assert "50" in log_output


@pytest.mark.unit
def test_log_optimization_header_with_early_stopping():
    """Test log_optimization_header includes early stopping info."""
    mock_logger = MagicMock()
    mock_cfg = MagicMock()
    mock_cfg.optuna.search_space_preset = "default"
    mock_cfg.optuna.n_trials = 10
    mock_cfg.optuna.epochs = 5
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.optuna.enable_pruning = False
    mock_cfg.optuna.enable_early_stopping = True
    mock_cfg.optuna.early_stopping_threshold = 0.95
    mock_cfg.optuna.early_stopping_patience = 5
    mock_cfg.optuna.model_pool = None

    log_optimization_header(cfg=mock_cfg, logger_instance=mock_logger)

    # 10 lines: 8 base + early_stop + empty line
    assert mock_logger.info.call_count == 10
    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)
    assert "Early Stop" in log_output or "0.95" in log_output


@pytest.mark.unit
def test_log_optimization_header_logs_only_search_params():
    """Test log_optimization_header logs dataset, model search, and search params."""
    mock_logger = MagicMock()
    mock_cfg = MagicMock()
    mock_cfg.optuna.search_space_preset = "default"
    mock_cfg.optuna.n_trials = 10
    mock_cfg.optuna.epochs = 5
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.optuna.enable_pruning = False
    mock_cfg.optuna.enable_early_stopping = False
    mock_cfg.optuna.model_pool = None

    log_optimization_header(cfg=mock_cfg, logger_instance=mock_logger)

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)
    # Should contain dataset, model search, and search params
    assert "Dataset" in log_output
    assert "Model Search" in log_output
    assert "Search Space" in log_output
    assert "Trials" in log_output


@pytest.mark.unit
def test_log_optimization_header_early_stop_auto_threshold():
    """Test log_optimization_header with auto early stopping threshold."""
    mock_logger = MagicMock()
    mock_cfg = MagicMock()
    mock_cfg.optuna.search_space_preset = "default"
    mock_cfg.optuna.n_trials = 10
    mock_cfg.optuna.epochs = 5
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.optuna.enable_pruning = False
    mock_cfg.optuna.enable_early_stopping = True
    mock_cfg.optuna.early_stopping_threshold = None
    mock_cfg.optuna.early_stopping_patience = 3
    mock_cfg.optuna.model_pool = None

    log_optimization_header(cfg=mock_cfg, logger_instance=mock_logger)

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)
    assert "auto" in log_output


@pytest.mark.unit
def test_log_optimization_header_with_model_pool():
    """Test log_optimization_header logs model pool when active."""
    mock_logger = MagicMock()
    mock_cfg = MagicMock()
    mock_cfg.optuna.search_space_preset = "full"
    mock_cfg.optuna.n_trials = 20
    mock_cfg.optuna.epochs = 15
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.optuna.enable_pruning = True
    mock_cfg.optuna.enable_early_stopping = False
    mock_cfg.optuna.model_pool = ["vit_tiny", "efficientnet_b0"]

    log_optimization_header(cfg=mock_cfg, logger_instance=mock_logger)

    # 10 lines: 9 base + model_pool
    assert mock_logger.info.call_count == 10
    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)
    assert "Model Pool" in log_output
    assert "vit_tiny" in log_output
    assert "efficientnet_b0" in log_output


# LOG TRIAL START
@pytest.mark.unit
def test_log_trial_start_basic():
    """Test log_trial_start logs trial parameters."""
    mock_logger = MagicMock()
    params = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "weight_decay": 0.0001,
    }

    log_trial_start(trial_number=5, params=params, logger_instance=mock_logger)

    assert mock_logger.info.call_count > 3
    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)
    assert "5" in log_output
    assert "0.001" in log_output or "1e-03" in log_output


@pytest.mark.unit
def test_log_trial_start_groups_parameters():
    """Test log_trial_start groups parameters by category."""
    mock_logger = MagicMock()
    params = {
        "learning_rate": 0.001,
        "mixup_alpha": 0.2,
        "rotation_angle": 15,
        "batch_size": 64,
    }

    log_trial_start(trial_number=1, params=params, logger_instance=mock_logger)

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)
    assert "[" in log_output


@pytest.mark.unit
def test_log_trial_start_formats_small_floats():
    """Test log_trial_start formats very small floats in scientific notation."""
    mock_logger = MagicMock()
    params = {
        "learning_rate": 0.0001,
        "weight_decay": 0.5,
    }

    log_trial_start(trial_number=1, params=params, logger_instance=mock_logger)

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)
    assert "e-" in log_output or "1e-04" in log_output


@pytest.mark.unit
def test_log_trial_start_with_string_params():
    """Test log_trial_start handles string parameters."""
    mock_logger = MagicMock()
    params = {
        "model_name": "resnet18",
        "learning_rate": 0.001,
        "weight_variant": "IMAGENET1K_V1",
    }

    log_trial_start(trial_number=3, params=params, logger_instance=mock_logger)

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)
    assert "resnet18" in log_output
    assert "IMAGENET1K_V1" in log_output


@pytest.mark.unit
def test_log_trial_start_with_regular_floats():
    """Test log_trial_start with regular floats (not small)."""
    mock_logger = MagicMock()
    params = {
        "learning_rate": 0.01,
        "mixup_alpha": 0.5,
        "dropout": 0.3,
    }

    log_trial_start(trial_number=1, params=params, logger_instance=mock_logger)

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)
    assert "0.01" in log_output or "0.0100" in log_output
    assert "e-" not in log_output


# REPORTER: LOG INITIAL STATUS
@pytest.mark.unit
def test_reporter_log_initial_status():
    """Test Reporter.log_initial_status logs environment info."""
    reporter = Reporter()
    mock_logger = MagicMock()
    mock_cfg = MagicMock()
    mock_cfg.task_type = "classification"
    mock_cfg.hardware.device = "cuda"
    mock_cfg.training.epochs = 60
    mock_cfg.training.batch_size = 32
    mock_cfg.training.learning_rate = 0.001
    mock_cfg.training.use_tta = True
    mock_cfg.training.use_amp = True
    mock_cfg.training.seed = 42
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.architecture.name = "resnet_18"
    mock_cfg.architecture.pretrained = True
    mock_cfg.architecture.weight_variant = None
    mock_cfg.dataset.metadata = MagicMock()
    mock_cfg.dataset.metadata.display_name = "BloodMNIST"
    mock_cfg.dataset.metadata.num_classes = 8
    mock_cfg.dataset.metadata.in_channels = 3
    mock_cfg.dataset.metadata.resolution_str = "28x28"
    mock_cfg.dataset.metadata.is_anatomical = True
    mock_cfg.dataset.metadata.is_texture_based = False
    mock_cfg.dataset.img_size = 28

    mock_paths = MagicMock()
    mock_paths.root = Path("/mock/run")
    mock_device = torch.device("cuda")

    reporter.log_initial_status(
        logger_instance=mock_logger,
        cfg=mock_cfg,
        paths=mock_paths,
        device=mock_device,
        applied_threads=4,
        num_workers=2,
    )

    assert mock_logger.info.call_count > 20
    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)
    assert "HARDWARE" in log_output or "DATASET" in log_output


@pytest.mark.unit
def test_reporter_log_initial_status_cpu_device():
    """Test Reporter.log_initial_status with CPU device."""
    reporter = Reporter()
    mock_logger = MagicMock()
    mock_cfg = MagicMock()
    mock_cfg.task_type = "classification"
    mock_cfg.hardware.device = "cpu"
    mock_cfg.training.epochs = 10
    mock_cfg.training.batch_size = 16
    mock_cfg.training.learning_rate = 0.01
    mock_cfg.training.use_tta = False
    mock_cfg.training.use_amp = False
    mock_cfg.training.seed = 42
    mock_cfg.hardware.use_deterministic_algorithms = True
    mock_cfg.architecture.name = "model"
    mock_cfg.architecture.pretrained = False
    mock_cfg.architecture.weight_variant = None
    mock_cfg.dataset.metadata = MagicMock()
    mock_cfg.dataset.metadata.display_name = "Test"
    mock_cfg.dataset.metadata.num_classes = 2
    mock_cfg.dataset.metadata.in_channels = 1
    mock_cfg.dataset.metadata.resolution_str = "28x28"
    mock_cfg.dataset.metadata.is_anatomical = False
    mock_cfg.dataset.metadata.is_texture_based = False
    mock_cfg.dataset.img_size = 28

    mock_paths = MagicMock()
    mock_paths.root = Path("/mock")
    mock_device = torch.device("cpu")

    reporter.log_initial_status(
        logger_instance=mock_logger,
        cfg=mock_cfg,
        paths=mock_paths,
        device=mock_device,
        applied_threads=1,
        num_workers=0,
    )

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)
    assert "CPU" in log_output
    assert "GPU Model" not in log_output


@pytest.mark.unit
def test_reporter_log_initial_status_with_weight_variant():
    """Test Reporter logs weight variant when present."""
    reporter = Reporter()
    mock_logger = MagicMock()
    mock_cfg = MagicMock()
    mock_cfg.task_type = "classification"
    mock_cfg.hardware.device = "cuda"
    mock_cfg.training.epochs = 10
    mock_cfg.training.batch_size = 32
    mock_cfg.training.learning_rate = 0.001
    mock_cfg.training.use_tta = False
    mock_cfg.training.use_amp = True
    mock_cfg.training.seed = 42
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.architecture.name = "vit_tiny"
    mock_cfg.architecture.pretrained = True
    mock_cfg.architecture.weight_variant = "IMAGENET1K_V1"
    mock_cfg.dataset.metadata = MagicMock()
    mock_cfg.dataset.metadata.display_name = "Test"
    mock_cfg.dataset.metadata.num_classes = 3
    mock_cfg.dataset.metadata.in_channels = 3
    mock_cfg.dataset.metadata.resolution_str = "224x224"
    mock_cfg.dataset.metadata.is_anatomical = False
    mock_cfg.dataset.metadata.is_texture_based = True
    mock_cfg.dataset.img_size = 224

    mock_paths = MagicMock()
    mock_device = torch.device("cuda")

    reporter.log_initial_status(
        logger_instance=mock_logger,
        cfg=mock_cfg,
        paths=mock_paths,
        device=mock_device,
        applied_threads=4,
        num_workers=2,
    )

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)
    assert "IMAGENET1K_V1" in log_output


@pytest.mark.unit
def test_reporter_log_hardware_section_device_fallback():
    """Test _log_hardware_section logs warning for device fallback."""
    reporter = Reporter()
    mock_logger = MagicMock()
    mock_cfg = MagicMock()
    mock_cfg.hardware.device = "cuda"
    mock_device = torch.device("cpu")

    reporter._log_hardware_section(
        logger_instance=mock_logger,
        cfg=mock_cfg,
        device=mock_device,
        applied_threads=4,
        num_workers=2,
    )

    assert mock_logger.warning.called
    calls = [str(call) for call in mock_logger.warning.call_args_list]
    log_output = " ".join(calls)
    assert "FALLBACK" in log_output or "unavailable" in log_output


@pytest.mark.unit
def test_reporter_log_hardware_section_cuda_device():
    """Test _log_hardware_section logs GPU info when CUDA available."""
    reporter = Reporter()
    mock_logger = MagicMock()
    mock_cfg = MagicMock()
    mock_cfg.hardware.device = "cuda"

    mock_device = torch.device("cuda")

    with patch(
        "orchard.core.logger.env_reporter.get_accelerator_name", return_value="NVIDIA RTX 5070"
    ):
        with patch("orchard.core.logger.env_reporter.get_vram_info", return_value="8 GB"):
            reporter._log_hardware_section(
                logger_instance=mock_logger,
                cfg=mock_cfg,
                device=mock_device,
                applied_threads=8,
                num_workers=4,
            )

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)
    assert "RTX 5070" in log_output or "GPU" in log_output
    assert "8 GB" in log_output or "VRAM" in log_output


# LOG PIPELINE SUMMARY
@pytest.mark.unit
def test_log_pipeline_summary_basic():
    """Test log_pipeline_summary logs basic pipeline completion info."""
    mock_logger = MagicMock()

    log_pipeline_summary(
        test_acc=0.9234,
        macro_f1=0.8765,
        best_model_path="/mock/best_model.pth",
        run_dir="/mock/outputs/run123",
        duration="5m 30s",
        logger_instance=mock_logger,
    )

    assert mock_logger.info.call_count > 0
    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)

    assert "PIPELINE COMPLETE" in log_output
    assert "92.34" in log_output
    assert "0.8765" in log_output
    assert "5m 30s" in log_output


@pytest.mark.unit
def test_log_pipeline_summary_with_onnx_path():
    """Test log_pipeline_summary includes ONNX path when provided."""
    mock_logger = MagicMock()

    log_pipeline_summary(
        test_acc=0.85,
        macro_f1=0.80,
        best_model_path="/mock/best.pth",
        run_dir="/mock/run",
        duration="10m 0s",
        onnx_path="/mock/model.onnx",
        logger_instance=mock_logger,
    )

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)

    assert "ONNX Export" in log_output
    assert "model.onnx" in log_output


@pytest.mark.unit
def test_log_pipeline_summary_without_onnx_path():
    """Test log_pipeline_summary omits ONNX line when not provided."""
    mock_logger = MagicMock()

    log_pipeline_summary(
        test_acc=0.9,
        macro_f1=0.85,
        best_model_path="/mock/best.pth",
        run_dir="/mock/run",
        duration="2m 15s",
        onnx_path=None,
        logger_instance=mock_logger,
    )

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)

    assert "ONNX" not in log_output


@pytest.mark.unit
def test_log_pipeline_summary_uses_module_logger_when_none():
    """Test log_pipeline_summary uses module logger when none provided."""
    with patch("orchard.core.logger.progress.logger") as mock_module_logger:
        log_pipeline_summary(
            test_acc=0.9,
            macro_f1=0.85,
            best_model_path="/mock/best.pth",
            run_dir="/mock/run",
            duration="1m 0s",
            logger_instance=None,
        )

        assert mock_module_logger.info.call_count > 0


@pytest.mark.unit
def test_log_pipeline_summary_with_test_auc():
    """Test log_pipeline_summary includes AUC line when test_auc is provided."""
    mock_logger = MagicMock()

    log_pipeline_summary(
        test_acc=0.92,
        macro_f1=0.88,
        best_model_path="/mock/best.pth",
        run_dir="/mock/run",
        duration="3m 0s",
        test_auc=0.9567,
        logger_instance=mock_logger,
    )

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)

    assert "Test AUC" in log_output
    assert "0.9567" in log_output


# REPORTER: TRACKING SECTION
@pytest.mark.unit
def test_reporter_log_tracking_section_enabled():
    """Test _log_tracking_section logs experiment name when tracking is enabled."""
    reporter = Reporter()
    mock_logger = MagicMock()
    mock_cfg = MagicMock()
    mock_cfg.tracking.enabled = True
    mock_cfg.tracking.experiment_name = "my_experiment"

    reporter._log_tracking_section(mock_logger, mock_cfg)

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)
    assert "TRACKING" in log_output
    assert "Active" in log_output
    assert "my_experiment" in log_output


@pytest.mark.unit
def test_reporter_log_tracking_section_disabled():
    """Test _log_tracking_section logs disabled status."""
    reporter = Reporter()
    mock_logger = MagicMock()
    mock_cfg = MagicMock()
    mock_cfg.tracking.enabled = False

    reporter._log_tracking_section(mock_logger, mock_cfg)

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)
    assert "Disabled" in log_output
    assert "Experiment" not in log_output


@pytest.mark.unit
def test_reporter_log_tracking_section_absent():
    """Test _log_tracking_section does nothing when tracking is not configured."""
    reporter = Reporter()
    mock_logger = MagicMock()
    mock_cfg = MagicMock(spec=[])

    reporter._log_tracking_section(mock_logger, mock_cfg)

    mock_logger.info.assert_not_called()


# REPORTER: OPTIMIZATION SECTION
@pytest.mark.unit
def test_reporter_log_optimization_section():
    """Test _log_optimization_section logs optimization parameters."""
    reporter = Reporter()
    mock_logger = MagicMock()
    mock_cfg = MagicMock()
    mock_cfg.optuna.n_trials = 20
    mock_cfg.optuna.epochs = 15
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.optuna.direction = "maximize"
    mock_cfg.optuna.sampler_type = "tpe"
    mock_cfg.optuna.enable_pruning = True
    mock_cfg.optuna.pruner_type = "median"

    reporter._log_optimization_section(mock_logger, mock_cfg)

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)
    assert "OPTIMIZATION" in log_output
    assert "20" in log_output
    assert "auc" in log_output
    assert "TPE" in log_output
    assert "median" in log_output


@pytest.mark.unit
def test_reporter_log_optimization_section_absent():
    """Test _log_optimization_section does nothing when optuna is not configured."""
    reporter = Reporter()
    mock_logger = MagicMock()
    mock_cfg = MagicMock(spec=[])

    reporter._log_optimization_section(mock_logger, mock_cfg)

    mock_logger.info.assert_not_called()


# REPORTER: EXPORT SECTION
@pytest.mark.unit
def test_reporter_log_export_section_basic():
    """Test _log_export_section logs format, opset, and validate_export."""
    reporter = Reporter()
    mock_logger = MagicMock()
    mock_cfg = MagicMock()
    mock_cfg.export.format = "onnx"
    mock_cfg.export.opset_version = 18
    mock_cfg.export.validate_export = True

    reporter._log_export_section(mock_logger, mock_cfg)

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)
    assert "EXPORT" in log_output
    assert "ONNX" in log_output
    assert "18" in log_output
    assert "True" in log_output


@pytest.mark.unit
@pytest.mark.parametrize("qtype", ["int8", "uint8", "int4", "uint4"])
def test_reporter_log_export_section_quantize_type(qtype):
    """Test _log_export_section shows the actual quantization_type."""
    reporter = Reporter()
    mock_logger = MagicMock()
    mock_cfg = MagicMock()
    mock_cfg.export.format = "onnx"
    mock_cfg.export.opset_version = 18
    mock_cfg.export.validate_export = True
    mock_cfg.export.quantize = True
    mock_cfg.export.quantization_type = qtype
    mock_cfg.export.quantization_backend = "qnnpack"

    reporter._log_export_section(mock_logger, mock_cfg)

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)
    assert qtype.upper() in log_output
    assert "qnnpack" in log_output


@pytest.mark.unit
def test_reporter_log_export_section_absent():
    """Test _log_export_section does nothing when export is not configured."""
    reporter = Reporter()
    mock_logger = MagicMock()
    mock_cfg = MagicMock(spec=[])

    reporter._log_export_section(mock_logger, mock_cfg)

    mock_logger.info.assert_not_called()


# _format_param_value
@pytest.mark.unit
def test_format_param_value_small_float():
    """Small floats (<0.001) use scientific notation."""
    from orchard.core.logger.progress import _format_param_value

    result = _format_param_value(0.0001)
    assert "e-" in result or "E-" in result


@pytest.mark.unit
def test_format_param_value_regular_float():
    """Regular floats (>=0.001) use 4-decimal notation."""
    from orchard.core.logger.progress import _format_param_value

    assert _format_param_value(0.5) == "0.5000"


@pytest.mark.unit
def test_format_param_value_integer():
    """Non-float values pass through str()."""
    from orchard.core.logger.progress import _format_param_value

    assert _format_param_value(42) == "42"


@pytest.mark.unit
def test_format_param_value_zero():
    """Zero is < 0.001, uses scientific notation."""
    from orchard.core.logger.progress import _format_param_value

    assert _format_param_value(0.0) == "0.00e+00"


@pytest.mark.unit
def test_format_param_value_boundary():
    """Exactly 0.001 is NOT < 0.001, uses decimal notation."""
    from orchard.core.logger.progress import _format_param_value

    assert _format_param_value(0.001) == "0.0010"


@pytest.mark.unit
def test_format_param_value_string():
    """String values pass through str()."""
    from orchard.core.logger.progress import _format_param_value

    assert _format_param_value("sgd") == "sgd"


# _count_trial_states
@pytest.mark.unit
def test_count_trial_states_all_complete():
    """All trials completed."""
    from orchard.core.logger.progress import _count_trial_states

    mock_study = MagicMock()
    t1, t2 = MagicMock(), MagicMock()
    t1.state = optuna.trial.TrialState.COMPLETE
    t2.state = optuna.trial.TrialState.COMPLETE
    mock_study.trials = [t1, t2]

    completed, pruned, failed = _count_trial_states(mock_study)
    assert len(completed) == 2
    assert len(pruned) == 0
    assert len(failed) == 0


@pytest.mark.unit
def test_count_trial_states_mixed():
    """Mixed trial states partitioned correctly."""
    from orchard.core.logger.progress import _count_trial_states

    mock_study = MagicMock()
    tc, tp, tf = MagicMock(), MagicMock(), MagicMock()
    tc.state = optuna.trial.TrialState.COMPLETE
    tp.state = optuna.trial.TrialState.PRUNED
    tf.state = optuna.trial.TrialState.FAIL
    mock_study.trials = [tc, tp, tf]

    completed, pruned, failed = _count_trial_states(mock_study)
    assert len(completed) == 1
    assert len(pruned) == 1
    assert len(failed) == 1


@pytest.mark.unit
def test_count_trial_states_empty():
    """Empty study returns empty lists."""
    from orchard.core.logger.progress import _count_trial_states

    mock_study = MagicMock()
    mock_study.trials = []

    completed, pruned, failed = _count_trial_states(mock_study)
    assert len(completed) == 0
    assert len(pruned) == 0
    assert len(failed) == 0


# ---------------------------------------------------------------------------
# Mutation-killing: exact category names + keys in log_trial_start,
# log.warning calls in summary
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_log_trial_start_exact_category_names():
    """Verify all 6 exact category names are passed as exact args (not substrings)."""
    mock_logger = MagicMock()
    # Provide at least one param per category
    params = {
        "learning_rate": 0.01,  # Optimization
        "criterion_type": "cross_entropy",  # Loss
        "mixup_alpha": 0.2,  # Regularization
        "scheduler_type": "cosine",  # Scheduling
        "rotation_angle": 15,  # Augmentation
        "model_name": "resnet18",  # Architecture
    }

    log_trial_start(trial_number=0, params=params, logger_instance=mock_logger)

    # Extract exact category name args from calls matching "%s[%s]" format
    category_args = []
    for call in mock_logger.info.call_args_list:
        args = call[0]
        if len(args) >= 3 and args[0] == "%s[%s]":
            category_args.append(args[2])

    expected = {
        "Optimization",
        "Loss",
        "Regularization",
        "Scheduling",
        "Augmentation",
        "Architecture",
    }
    assert set(category_args) == expected


@pytest.mark.unit
def test_log_trial_start_optimization_keys():
    """Verify Optimization category includes all 4 expected keys."""
    mock_logger = MagicMock()
    params = {
        "learning_rate": 0.01,
        "weight_decay": 0.001,
        "momentum": 0.9,
        "min_lr": 0.0001,
    }

    log_trial_start(trial_number=0, params=params, logger_instance=mock_logger)

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)

    assert "Optimization" in log_output
    for key in ["learning_rate", "weight_decay", "momentum", "min_lr"]:
        assert key in log_output, f"Key '{key}' missing from Optimization category"


@pytest.mark.unit
def test_log_trial_start_loss_keys():
    """Verify Loss category includes all 3 expected keys."""
    mock_logger = MagicMock()
    params = {
        "criterion_type": "focal",
        "focal_gamma": 2.0,
        "label_smoothing": 0.1,
    }

    log_trial_start(trial_number=0, params=params, logger_instance=mock_logger)

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)

    assert "Loss" in log_output
    for key in ["criterion_type", "focal_gamma", "label_smoothing"]:
        assert key in log_output, f"Key '{key}' missing from Loss category"


@pytest.mark.unit
def test_log_trial_start_regularization_keys():
    """Verify Regularization category includes mixup_alpha and dropout."""
    mock_logger = MagicMock()
    params = {"mixup_alpha": 0.2, "dropout": 0.3}

    log_trial_start(trial_number=0, params=params, logger_instance=mock_logger)

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)

    assert "Regularization" in log_output
    for key in ["mixup_alpha", "dropout"]:
        assert key in log_output


@pytest.mark.unit
def test_log_trial_start_scheduling_keys():
    """Verify Scheduling category includes all 3 expected keys."""
    mock_logger = MagicMock()
    params = {"scheduler_type": "cosine", "scheduler_patience": 5, "batch_size": 32}

    log_trial_start(trial_number=0, params=params, logger_instance=mock_logger)

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)

    assert "Scheduling" in log_output
    for key in ["scheduler_type", "scheduler_patience", "batch_size"]:
        assert key in log_output


@pytest.mark.unit
def test_log_trial_start_augmentation_keys():
    """Verify Augmentation category includes all 3 expected keys."""
    mock_logger = MagicMock()
    params = {"rotation_angle": 15, "jitter_val": 0.1, "min_scale": 0.8}

    log_trial_start(trial_number=0, params=params, logger_instance=mock_logger)

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)

    assert "Augmentation" in log_output
    for key in ["rotation_angle", "jitter_val", "min_scale"]:
        assert key in log_output


@pytest.mark.unit
def test_log_trial_start_architecture_keys():
    """Verify Architecture category includes all 3 expected keys."""
    mock_logger = MagicMock()
    params = {"model_name": "vit_tiny", "pretrained": True, "weight_variant": "DEFAULT"}

    log_trial_start(trial_number=0, params=params, logger_instance=mock_logger)

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)

    assert "Architecture" in log_output
    for key in ["model_name", "pretrained", "weight_variant"]:
        assert key in log_output


@pytest.mark.unit
def test_log_trial_start_empty_category_not_shown():
    """Category with no matching params should not appear."""
    mock_logger = MagicMock()
    params = {"learning_rate": 0.01}  # Only Optimization

    log_trial_start(trial_number=0, params=params, logger_instance=mock_logger)

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)

    assert "Optimization" in log_output
    assert "Loss" not in log_output
    assert "Regularization" not in log_output
    assert "Augmentation" not in log_output
    assert "Architecture" not in log_output


@pytest.mark.unit
def test_log_trial_start_format_string_uses_key_and_value():
    """Verify each param is logged with key and formatted value."""
    mock_logger = MagicMock()
    params = {"learning_rate": 0.5}

    log_trial_start(trial_number=0, params=params, logger_instance=mock_logger)

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)

    assert "learning_rate" in log_output
    assert "0.5000" in log_output  # _format_param_value(0.5) == "0.5000"


@pytest.mark.unit
def test_log_optimization_summary_warning_no_completed():
    """log.warning called with exact (fmt, I, W) args when no completed trials."""
    mock_logger = MagicMock()
    mock_study = MagicMock()
    mock_study.trials = [MagicMock(state=optuna.trial.TrialState.PRUNED)]

    mock_cfg = MagicMock()
    mock_cfg.dataset.dataset_name = "test"
    mock_cfg.optuna.search_space_preset = "default"
    mock_cfg.training.monitor_metric = "auc"

    log_optimization_summary(
        study=mock_study,
        cfg=mock_cfg,
        device=MagicMock(__str__=lambda _: "cpu"),
        paths=MagicMock(root="/tmp/test"),
        logger_instance=mock_logger,
    )

    mock_logger.warning.assert_called()
    # Verify exact positional args: (fmt, INDENT, WARNING_SYMBOL)
    found = False
    for call in mock_logger.warning.call_args_list:
        args = call[0]
        if len(args) >= 3 and "No trials completed" in args[0]:
            assert args[1] == LogStyle.INDENT, f"Expected INDENT, got {args[1]!r}"
            assert args[2] == LogStyle.WARNING, f"Expected WARNING symbol, got {args[2]!r}"
            found = True
            break
    assert found, "No 'No trials completed' warning call found"


# ---------------------------------------------------------------------------
# Mutation-killing: Reporter exact argument assertions
# ---------------------------------------------------------------------------


def _build_cpu_cfg() -> MagicMock:
    """Build a minimal mock cfg for CPU-only Reporter tests."""
    cfg = MagicMock()
    cfg.hardware.device = "cpu"
    cfg.training.epochs = 10
    cfg.training.batch_size = 16
    cfg.training.learning_rate = 0.01
    cfg.training.use_tta = False
    cfg.training.use_amp = False
    cfg.training.seed = 42
    cfg.training.monitor_metric = "auc"
    cfg.hardware.use_deterministic_algorithms = False
    cfg.architecture.name = "mini_cnn"
    cfg.architecture.pretrained = False
    cfg.architecture.weight_variant = None
    cfg.augmentation.tta_mode = "none"
    cfg.dataset.metadata.display_name = "Test"
    cfg.dataset.metadata.num_classes = 3
    cfg.dataset.metadata.in_channels = 1
    cfg.dataset.metadata.resolution_str = "28x28"
    cfg.dataset.metadata.is_anatomical = False
    cfg.dataset.metadata.is_texture_based = True
    cfg.dataset.img_size = 28
    cfg.task_type = "classification"
    cfg.run_slug = "test-run"
    return cfg


def _find_call_with_label(calls, label: str):
    """Find the call tuple that contains a specific label like 'Weights'.

    Logger calls use: logger.info(fmt, I, A, label, value, ...) where
    args[0]=fmt, args[1]=I, args[2]=A, args[3]=label, args[4]=value.
    """
    for c in calls:
        args = c[0]
        if len(args) >= 4 and args[3] is not None and label in str(args[3]):
            return args
    return None


# --- log_initial_status: delegation and argument passing ---
@pytest.mark.unit
def test_reporter_initial_status_passes_indent_and_arrow():
    """Kill mutmut I=None and A=None in log_initial_status."""
    reporter = Reporter()
    log = MagicMock()
    cfg = _build_cpu_cfg()
    paths = MagicMock()
    paths.root = Path("/run")
    device = torch.device("cpu")

    reporter.log_initial_status(log, cfg, paths, device, 4, 2)

    # Experiment line uses I, A — verify they are not None
    exp_call = _find_call_with_label(log.info.call_args_list, "Experiment")
    assert exp_call is not None
    assert exp_call[1] == LogStyle.INDENT
    assert exp_call[2] == LogStyle.ARROW


@pytest.mark.unit
def test_reporter_initial_status_passes_applied_threads_and_workers():
    """Kill mutmut applied_threads=None and num_workers=None in log_initial_status."""
    reporter = Reporter()
    log = MagicMock()
    cfg = _build_cpu_cfg()
    paths = MagicMock()
    paths.root = Path("/run")
    device = torch.device("cpu")

    reporter.log_initial_status(log, cfg, paths, device, 7, 3)

    # Check hardware section got the right values
    dl_call = _find_call_with_label(log.info.call_args_list, "DataLoader")
    assert dl_call is not None
    assert dl_call[4] == 3  # num_workers

    ct_call = _find_call_with_label(log.info.call_args_list, "Compute Threads")
    assert ct_call is not None
    assert ct_call[4] == 7  # applied_threads


@pytest.mark.unit
def test_reporter_initial_status_lr_formatting():
    """Kill mutmut lr=None, lr_str=None, and str(None) mutations."""
    reporter = Reporter()
    log = MagicMock()
    cfg = _build_cpu_cfg()
    cfg.training.learning_rate = 0.001
    paths = MagicMock()
    paths.root = Path("/run")
    device = torch.device("cpu")

    reporter.log_initial_status(log, cfg, paths, device, 4, 2)

    lr_call = _find_call_with_label(log.info.call_args_list, "Initial LR")
    assert lr_call is not None
    assert lr_call[4] == "1.00e-03"  # exact scientific notation


@pytest.mark.unit
def test_reporter_initial_status_lr_string_passthrough():
    """Kill str(None) mutation — when LR is string, pass through as-is."""
    reporter = Reporter()
    log = MagicMock()
    cfg = _build_cpu_cfg()
    cfg.training.learning_rate = "auto"
    paths = MagicMock()
    paths.root = Path("/run")
    device = torch.device("cpu")

    reporter.log_initial_status(log, cfg, paths, device, 4, 2)

    lr_call = _find_call_with_label(log.info.call_args_list, "Initial LR")
    assert lr_call is not None
    assert lr_call[4] == "auto"


@pytest.mark.unit
def test_reporter_initial_status_env_init_header():
    """Kill mutmut XX/lowercase mutations on ENVIRONMENT INITIALIZATION header."""
    reporter = Reporter()
    log = MagicMock()
    cfg = _build_cpu_cfg()
    paths = MagicMock()
    paths.root = Path("/run")
    device = torch.device("cpu")

    reporter.log_initial_status(log, cfg, paths, device, 4, 2)

    # log_phase_header centers the title — verify exact centered string
    expected_centered = "ENVIRONMENT INITIALIZATION".center(LogStyle.HEADER_WIDTH)
    found = False
    for c in log.info.call_args_list:
        args = c[0]
        if len(args) == 1 and args[0] == expected_centered:
            found = True
            break
    assert found, "ENVIRONMENT INITIALIZATION header not found with exact content"


@pytest.mark.unit
def test_reporter_initial_status_logs_task_section():
    """log_initial_status includes [TASK] section with capitalized task_type."""
    reporter = Reporter()
    log = MagicMock()
    cfg = _build_cpu_cfg()
    paths = MagicMock()
    paths.root = Path("/run")
    device = torch.device("cpu")

    reporter.log_initial_status(log, cfg, paths, device, 4, 2)

    task_call = _find_call_with_label(log.info.call_args_list, "Type")
    assert task_call is not None
    assert task_call[4] == "Classification"

    calls_str = " ".join(str(c) for c in log.info.call_args_list)
    assert "[TASK]" in calls_str


@pytest.mark.unit
def test_reporter_initial_status_delegates_tracking_cfg():
    """Kill mutmut cfg→None mutation for _log_tracking_section."""
    reporter = Reporter()
    log = MagicMock()
    cfg = _build_cpu_cfg()
    cfg.tracking.enabled = True
    cfg.tracking.experiment_name = "exp1"
    paths = MagicMock()
    paths.root = Path("/run")
    device = torch.device("cpu")

    reporter.log_initial_status(log, cfg, paths, device, 4, 2)

    # If cfg was replaced by None, getattr(None, 'tracking', None) returns None
    # and [TRACKING] section would not appear
    calls_str = " ".join(str(c) for c in log.info.call_args_list)
    assert "TRACKING" in calls_str
    assert "exp1" in calls_str


@pytest.mark.unit
def test_reporter_initial_status_delegates_optimization_cfg():
    """Kill mutmut cfg→None mutation for _log_optimization_section."""
    reporter = Reporter()
    log = MagicMock()
    cfg = _build_cpu_cfg()
    cfg.optuna.n_trials = 10
    cfg.optuna.epochs = 5
    cfg.optuna.direction = "maximize"
    cfg.optuna.sampler_type = "tpe"
    cfg.optuna.enable_pruning = False
    paths = MagicMock()
    paths.root = Path("/run")
    device = torch.device("cpu")

    reporter.log_initial_status(log, cfg, paths, device, 4, 2)

    calls_str = " ".join(str(c) for c in log.info.call_args_list)
    assert "OPTIMIZATION" in calls_str


@pytest.mark.unit
def test_reporter_initial_status_delegates_export_cfg():
    """Kill mutmut cfg→None mutation for _log_export_section."""
    reporter = Reporter()
    log = MagicMock()
    cfg = _build_cpu_cfg()
    cfg.export.format = "onnx"
    cfg.export.opset_version = 18
    cfg.export.validate_export = True
    cfg.export.quantize = False
    paths = MagicMock()
    paths.root = Path("/run")
    device = torch.device("cpu")

    reporter.log_initial_status(log, cfg, paths, device, 4, 2)

    calls_str = " ".join(str(c) for c in log.info.call_args_list)
    assert "EXPORT" in calls_str


# --- _log_hardware_section: exact argument assertions ---
@pytest.mark.unit
def test_hardware_section_exact_args_cpu():
    """Kill I=None, A=None, device.lower() mutations in hardware section."""
    reporter = Reporter()
    log = MagicMock()
    cfg = MagicMock()
    cfg.hardware.device = "cpu"
    device = torch.device("cpu")

    reporter._log_hardware_section(log, cfg, device, 4, 2)

    I = LogStyle.INDENT  # noqa: E741
    A = LogStyle.ARROW

    # Active Device uses str(device).upper()
    ad_call = _find_call_with_label(log.info.call_args_list, "Active Device")
    assert ad_call is not None
    assert ad_call[1] == I
    assert ad_call[2] == A
    assert ad_call[4] == "CPU"

    # DataLoader
    dl_call = _find_call_with_label(log.info.call_args_list, "DataLoader")
    assert dl_call is not None
    assert dl_call[4] == 2  # num_workers

    # Compute Threads
    ct_call = _find_call_with_label(log.info.call_args_list, "Compute Threads")
    assert ct_call is not None
    assert ct_call[4] == 4  # applied_threads


@pytest.mark.unit
def test_hardware_section_fallback_warning_exact_args():
    """Kill I, LogStyle.WARNING, requested_device arg swap/None mutations."""
    reporter = Reporter()
    log = MagicMock()
    cfg = MagicMock()
    cfg.hardware.device = "cuda"
    device = torch.device("cpu")

    reporter._log_hardware_section(log, cfg, device, 4, 2)

    assert log.warning.called
    warn_args = log.warning.call_args[0]
    # args: (fmt, I, WARNING, requested_device)
    assert warn_args[1] == LogStyle.INDENT
    assert warn_args[2] == LogStyle.WARNING
    assert warn_args[3] == "cuda"  # requested_device after .lower()


@pytest.mark.unit
def test_hardware_section_no_fallback_when_cpu_requested():
    """Kill 'or' mutation: if requested=='cpu' and active=='cpu', no warning."""
    reporter = Reporter()
    log = MagicMock()
    cfg = MagicMock()
    cfg.hardware.device = "cpu"
    device = torch.device("cpu")

    reporter._log_hardware_section(log, cfg, device, 4, 2)

    log.warning.assert_not_called()


@pytest.mark.unit
def test_hardware_section_lower_not_upper():
    """Kill .upper() mutation on requested_device = cfg.hardware.device.lower()."""
    reporter = Reporter()
    log = MagicMock()
    cfg = MagicMock()
    cfg.hardware.device = "CUDA"
    device = torch.device("cpu")

    reporter._log_hardware_section(log, cfg, device, 4, 2)

    # requested_device should be "cuda" (lower), not "CUDA" (upper)
    assert log.warning.called
    warn_args = log.warning.call_args[0]
    assert warn_args[3] == "cuda"


@pytest.mark.unit
def test_hardware_section_mps_no_vram():
    """Kill 'MPS'→'XXmpsXX' mutation in active_type in ('cuda', 'mps')."""
    reporter = Reporter()
    log = MagicMock()
    cfg = MagicMock()
    cfg.hardware.device = "mps"
    device = MagicMock()
    device.type = "mps"
    device.__str__ = lambda _: "mps"

    with patch("orchard.core.logger.env_reporter.get_accelerator_name", return_value="Apple M2"):
        reporter._log_hardware_section(log, cfg, device, 4, 2)

    calls_str = " ".join(str(c) for c in log.info.call_args_list)
    assert "Apple M2" in calls_str
    # MPS should NOT log VRAM
    assert "VRAM" not in calls_str


@pytest.mark.unit
def test_hardware_section_fallback_cpu_string_exact():
    """Kill 'cpu'→'XXcpuXX' and 'cpu'→'CPU' mutations in condition."""
    reporter = Reporter()
    log = MagicMock()
    cfg = MagicMock()

    # requested='cpu' (exact), active='cpu' => no warning
    cfg.hardware.device = "cpu"
    reporter._log_hardware_section(log, cfg, torch.device("cpu"), 4, 2)
    log.warning.assert_not_called()

    # requested='cuda', active='cpu' => warning
    log.reset_mock()
    cfg.hardware.device = "cuda"
    reporter._log_hardware_section(log, cfg, torch.device("cpu"), 4, 2)
    assert log.warning.called


# --- _log_dataset_section: I and A not None ---
@pytest.mark.unit
def test_dataset_section_exact_indent_arrow():
    """Kill I=None, A=None mutations in _log_dataset_section."""
    reporter = Reporter()
    log = MagicMock()
    cfg = MagicMock()
    cfg.dataset.metadata.display_name = "Blood"
    cfg.dataset.metadata.num_classes = 8
    cfg.dataset.metadata.in_channels = 3
    cfg.dataset.metadata.resolution_str = "28x28"
    cfg.dataset.metadata.is_anatomical = True
    cfg.dataset.metadata.is_texture_based = False
    cfg.dataset.img_size = 28

    reporter._log_dataset_section(log, cfg)

    name_call = _find_call_with_label(log.info.call_args_list, "Name")
    assert name_call is not None
    assert name_call[1] == LogStyle.INDENT
    assert name_call[2] == LogStyle.ARROW
    assert name_call[4] == "Blood"


# --- _log_strategy_section: exact values for weights, precision, repro, tta ---
@pytest.mark.unit
def test_strategy_section_pretrained_weights():
    """Kill 'Pretrained'→'XXPretrainedXX'/'pretrained'/'PRETRAINED'/None."""
    reporter = Reporter()
    log = MagicMock()
    cfg = MagicMock()
    cfg.architecture.pretrained = True
    cfg.architecture.name = "resnet_18"
    cfg.architecture.weight_variant = None
    cfg.training.use_tta = False
    cfg.training.use_amp = False
    cfg.training.seed = 42
    cfg.hardware.use_deterministic_algorithms = False
    cfg.augmentation.tta_mode = "none"

    with patch("orchard.core.logger.env_reporter.determine_tta_mode", return_value="disabled"):
        reporter._log_strategy_section(log, cfg, torch.device("cpu"))

    w_call = _find_call_with_label(log.info.call_args_list, "Weights")
    assert w_call is not None
    assert w_call[4] == "Pretrained"


@pytest.mark.unit
def test_strategy_section_random_weights():
    """Kill 'Random'→'XXRandomXX'/'random'/'RANDOM' mutations."""
    reporter = Reporter()
    log = MagicMock()
    cfg = MagicMock()
    cfg.architecture.pretrained = False
    cfg.architecture.name = "mini_cnn"
    cfg.architecture.weight_variant = None
    cfg.training.use_tta = False
    cfg.training.use_amp = False
    cfg.training.seed = 42
    cfg.hardware.use_deterministic_algorithms = False
    cfg.augmentation.tta_mode = "none"

    with patch("orchard.core.logger.env_reporter.determine_tta_mode", return_value="disabled"):
        reporter._log_strategy_section(log, cfg, torch.device("cpu"))

    w_call = _find_call_with_label(log.info.call_args_list, "Weights")
    assert w_call is not None
    assert w_call[4] == "Random"


@pytest.mark.unit
def test_strategy_section_amp_precision():
    """Kill 'AMP (Mixed)'→mutations and None."""
    reporter = Reporter()
    log = MagicMock()
    cfg = MagicMock()
    cfg.architecture.pretrained = False
    cfg.architecture.name = "x"
    cfg.architecture.weight_variant = None
    cfg.training.use_tta = False
    cfg.training.use_amp = True
    cfg.training.seed = 1
    cfg.hardware.use_deterministic_algorithms = False
    cfg.augmentation.tta_mode = "none"

    with patch("orchard.core.logger.env_reporter.determine_tta_mode", return_value="disabled"):
        reporter._log_strategy_section(log, cfg, torch.device("cpu"))

    p_call = _find_call_with_label(log.info.call_args_list, "Precision")
    assert p_call is not None
    assert p_call[4] == "AMP (Mixed)"


@pytest.mark.unit
def test_strategy_section_fp32_precision():
    """Kill 'FP32'→'XXFP32XX'/'fp32' mutations."""
    reporter = Reporter()
    log = MagicMock()
    cfg = MagicMock()
    cfg.architecture.pretrained = False
    cfg.architecture.name = "x"
    cfg.architecture.weight_variant = None
    cfg.training.use_tta = False
    cfg.training.use_amp = False
    cfg.training.seed = 1
    cfg.hardware.use_deterministic_algorithms = False
    cfg.augmentation.tta_mode = "none"

    with patch("orchard.core.logger.env_reporter.determine_tta_mode", return_value="disabled"):
        reporter._log_strategy_section(log, cfg, torch.device("cpu"))

    p_call = _find_call_with_label(log.info.call_args_list, "Precision")
    assert p_call is not None
    assert p_call[4] == "FP32"


@pytest.mark.unit
def test_strategy_section_strict_repro_mode():
    """Kill 'Strict'→mutations and None."""
    reporter = Reporter()
    log = MagicMock()
    cfg = MagicMock()
    cfg.architecture.pretrained = False
    cfg.architecture.name = "x"
    cfg.architecture.weight_variant = None
    cfg.training.use_tta = False
    cfg.training.use_amp = False
    cfg.training.seed = 1
    cfg.hardware.use_deterministic_algorithms = True
    cfg.augmentation.tta_mode = "none"

    with patch("orchard.core.logger.env_reporter.determine_tta_mode", return_value="disabled"):
        reporter._log_strategy_section(log, cfg, torch.device("cpu"))

    r_call = _find_call_with_label(log.info.call_args_list, "Repro. Mode")
    assert r_call is not None
    assert r_call[4] == "Strict"


@pytest.mark.unit
def test_strategy_section_standard_repro_mode():
    """Kill 'Standard'→mutations."""
    reporter = Reporter()
    log = MagicMock()
    cfg = MagicMock()
    cfg.architecture.pretrained = False
    cfg.architecture.name = "x"
    cfg.architecture.weight_variant = None
    cfg.training.use_tta = False
    cfg.training.use_amp = False
    cfg.training.seed = 1
    cfg.hardware.use_deterministic_algorithms = False
    cfg.augmentation.tta_mode = "none"

    with patch("orchard.core.logger.env_reporter.determine_tta_mode", return_value="disabled"):
        reporter._log_strategy_section(log, cfg, torch.device("cpu"))

    r_call = _find_call_with_label(log.info.call_args_list, "Repro. Mode")
    assert r_call is not None
    assert r_call[4] == "Standard"


@pytest.mark.unit
def test_strategy_section_tta_mode_exact():
    """Kill tta_status=None and determine_tta_mode arg mutations."""
    reporter = Reporter()
    log = MagicMock()
    cfg = MagicMock()
    cfg.architecture.pretrained = False
    cfg.architecture.name = "x"
    cfg.architecture.weight_variant = None
    cfg.training.use_tta = True
    cfg.training.use_amp = False
    cfg.training.seed = 1
    cfg.hardware.use_deterministic_algorithms = False
    cfg.augmentation.tta_mode = "flip"

    with patch(
        "orchard.core.logger.env_reporter.determine_tta_mode", return_value="flip"
    ) as mock_tta:
        reporter._log_strategy_section(log, cfg, torch.device("cpu"))
        mock_tta.assert_called_once_with(True, "cpu", "flip")

    tta_call = _find_call_with_label(log.info.call_args_list, "TTA Mode")
    assert tta_call is not None
    assert tta_call[4] == "flip"


@pytest.mark.unit
def test_strategy_section_indent_arrow_not_none():
    """Kill I=None, A=None in _log_strategy_section."""
    reporter = Reporter()
    log = MagicMock()
    cfg = MagicMock()
    cfg.architecture.pretrained = False
    cfg.architecture.name = "x"
    cfg.architecture.weight_variant = None
    cfg.training.use_tta = False
    cfg.training.use_amp = False
    cfg.training.seed = 1
    cfg.hardware.use_deterministic_algorithms = False
    cfg.augmentation.tta_mode = "none"

    with patch("orchard.core.logger.env_reporter.determine_tta_mode", return_value="none"):
        reporter._log_strategy_section(log, cfg, torch.device("cpu"))

    arch_call = _find_call_with_label(log.info.call_args_list, "Architecture")
    assert arch_call is not None
    assert arch_call[1] == LogStyle.INDENT
    assert arch_call[2] == LogStyle.ARROW


# --- _log_tracking_section: exact status values ---
@pytest.mark.unit
def test_tracking_section_active_exact():
    """Kill 'Active'→'XXActiveXX' mutation."""
    reporter = Reporter()
    log = MagicMock()
    cfg = MagicMock()
    cfg.tracking.enabled = True
    cfg.tracking.experiment_name = "exp"

    reporter._log_tracking_section(log, cfg)

    status_call = _find_call_with_label(log.info.call_args_list, "Status")
    assert status_call is not None
    assert status_call[4] == "Active"
    assert status_call[1] == LogStyle.INDENT
    assert status_call[2] == LogStyle.ARROW


@pytest.mark.unit
def test_tracking_section_disabled_exact():
    """Kill 'Disabled'→'XXDisabledXX' mutation."""
    reporter = Reporter()
    log = MagicMock()
    cfg = MagicMock()
    cfg.tracking.enabled = False

    reporter._log_tracking_section(log, cfg)

    status_call = _find_call_with_label(log.info.call_args_list, "Status")
    assert status_call is not None
    assert status_call[4] == "Disabled"


# --- _log_optimization_section: I/A not None ---
@pytest.mark.unit
def test_optimization_section_indent_arrow_not_none():
    """Kill I=None, A=None in _log_optimization_section."""
    reporter = Reporter()
    log = MagicMock()
    cfg = MagicMock()
    cfg.optuna.n_trials = 20
    cfg.optuna.epochs = 10
    cfg.training.monitor_metric = "auc"
    cfg.optuna.direction = "maximize"
    cfg.optuna.sampler_type = "tpe"
    cfg.optuna.enable_pruning = False

    reporter._log_optimization_section(log, cfg)

    trials_call = _find_call_with_label(log.info.call_args_list, "Trials")
    assert trials_call is not None
    assert trials_call[1] == LogStyle.INDENT
    assert trials_call[2] == LogStyle.ARROW


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
