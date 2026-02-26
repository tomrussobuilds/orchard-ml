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

    with patch("orchard.core.logger.reporter.get_cuda_name", return_value="NVIDIA RTX 5070"):
        with patch("orchard.core.logger.reporter.get_vram_info", return_value="8 GB"):
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
    assert "92.34%" in log_output
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
def test_reporter_log_export_section_absent():
    """Test _log_export_section does nothing when export is not configured."""
    reporter = Reporter()
    mock_logger = MagicMock()
    mock_cfg = MagicMock(spec=[])

    reporter._log_export_section(mock_logger, mock_cfg)

    mock_logger.info.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
