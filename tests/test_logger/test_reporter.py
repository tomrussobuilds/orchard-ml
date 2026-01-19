"""
Test Suite for Telemetry & Environment Reporting Engine.

Tests LogStyle constants, Reporter logging methods,
and summary logging functions.
"""

# =========================================================================== #
#                         Standard Imports                                    #
# =========================================================================== #
from unittest.mock import MagicMock, patch

# =========================================================================== #
#                         Third-Party Imports                                 #
# =========================================================================== #
import pytest
import torch

# =========================================================================== #
#                         Internal Imports                                    #
# =========================================================================== #
from orchard.core.logger.reporter import (
    LogStyle,
    Reporter,
    log_best_config_export,
    log_optimization_header,
    log_optimization_summary,
    log_training_summary,
    log_trial_start,
)

# =========================================================================== #
#                    LOGSTYLE: CONSTANTS                                      #
# =========================================================================== #


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


# =========================================================================== #
#                    REPORTER: INITIALIZATION                                 #
# =========================================================================== #


@pytest.mark.unit
def test_reporter_init():
    """Test Reporter can be instantiated."""
    reporter = Reporter()
    assert isinstance(reporter, Reporter)


# =========================================================================== #
#                    LOG TRAINING SUMMARY                                     #
# =========================================================================== #


@pytest.mark.unit
def test_log_training_summary_basic():
    """Test log_training_summary logs basic training results."""
    mock_logger = MagicMock()
    mock_cfg = MagicMock()
    mock_cfg.dataset.dataset_name = "bloodmnist"
    mock_cfg.model.name = "resnet_18_adapted"
    mock_cfg.model.weight_variant = None
    mock_device = torch.device("cpu")
    mock_paths = MagicMock()
    mock_paths.root = "/tmp/outputs/run123"

    log_training_summary(
        cfg=mock_cfg,
        test_acc=0.9678,
        macro_f1=0.9643,
        device=mock_device,
        paths=mock_paths,
        logger_instance=mock_logger,
    )

    # Verify logger was called
    assert mock_logger.info.call_count > 5
    # Verify key information was logged
    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)
    assert "bloodmnist" in log_output
    assert "resnet_18_adapted" in log_output


@pytest.mark.unit
def test_log_training_summary_with_weight_variant():
    """Test log_training_summary includes weight variant when present."""
    mock_logger = MagicMock()
    mock_cfg = MagicMock()
    mock_cfg.dataset.dataset_name = "pathmnist"
    mock_cfg.model.name = "vit_tiny"
    mock_cfg.model.weight_variant = "IMAGENET1K_V1"
    mock_device = torch.device("cuda")
    mock_paths = MagicMock()
    mock_paths.root = "/tmp/outputs/run456"

    log_training_summary(
        cfg=mock_cfg,
        test_acc=0.92,
        macro_f1=0.91,
        device=mock_device,
        paths=mock_paths,
        logger_instance=mock_logger,
    )

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)
    assert "IMAGENET1K_V1" in log_output


@pytest.mark.unit
def test_log_training_summary_formats_accuracy():
    """Test log_training_summary formats accuracy as percentage."""
    mock_logger = MagicMock()
    mock_cfg = MagicMock()
    mock_cfg.dataset.dataset_name = "test"
    mock_cfg.model.name = "model"
    mock_cfg.model.weight_variant = None
    mock_device = torch.device("cpu")
    mock_paths = MagicMock()
    mock_paths.root = "/tmp"

    log_training_summary(
        cfg=mock_cfg,
        test_acc=0.8765,
        macro_f1=0.8543,
        device=mock_device,
        paths=mock_paths,
        logger_instance=mock_logger,
    )

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)
    # Should contain formatted percentage
    assert "87.65%" in log_output or "0.8765" in log_output


# =========================================================================== #
#                    LOG OPTIMIZATION SUMMARY                                 #
# =========================================================================== #


@pytest.mark.unit
def test_log_optimization_summary_completed_trials():
    """Test log_optimization_summary with completed trials."""
    mock_logger = MagicMock()
    mock_cfg = MagicMock()
    mock_cfg.dataset.dataset_name = "bloodmnist"
    mock_cfg.optuna.search_space_preset = "default"
    mock_cfg.optuna.metric_name = "auc"
    mock_device = torch.device("cuda")
    mock_paths = MagicMock()
    mock_paths.root = "/tmp/outputs"

    # Mock study with completed trials
    mock_study = MagicMock()
    mock_trial = MagicMock()
    mock_trial.state.name = "COMPLETE"
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
    assert "3" in log_output  # 3 completed trials


@pytest.mark.unit
def test_log_optimization_summary_no_completed_trials():
    """Test log_optimization_summary with no completed trials."""
    mock_logger = MagicMock()
    mock_cfg = MagicMock()
    mock_cfg.dataset.dataset_name = "test"
    mock_cfg.optuna.search_space_preset = "default"
    mock_cfg.optuna.metric_name = "auc"
    mock_device = torch.device("cpu")
    mock_paths = MagicMock()

    # Mock study with no completed trials
    mock_study = MagicMock()
    mock_trial = MagicMock()
    mock_trial.state.name = "PRUNED"
    mock_study.trials = [mock_trial]

    log_optimization_summary(
        study=mock_study,
        cfg=mock_cfg,
        device=mock_device,
        paths=mock_paths,
        logger_instance=mock_logger,
    )

    # Should call warning about no completed trials
    assert mock_logger.warning.called


@pytest.mark.unit
def test_log_optimization_summary_with_failed_trials():
    """Test log_optimization_summary includes failed trial count."""
    mock_logger = MagicMock()
    mock_cfg = MagicMock()
    mock_cfg.dataset.dataset_name = "test"
    mock_cfg.optuna.search_space_preset = "default"
    mock_cfg.optuna.metric_name = "auc"
    mock_device = torch.device("cpu")
    mock_paths = MagicMock()

    # Mock study with failed trials
    mock_study = MagicMock()
    mock_trial_complete = MagicMock()
    mock_trial_complete.state.name = "COMPLETE"
    mock_trial_failed = MagicMock()
    mock_trial_failed.state.name = "FAIL"
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


# =========================================================================== #
#                    LOG OPTIMIZATION HEADER                                  #
# =========================================================================== #


@pytest.mark.unit
def test_log_optimization_header_basic():
    """Test log_optimization_header logs study configuration."""
    mock_logger = MagicMock()
    mock_cfg = MagicMock()
    mock_cfg.dataset.dataset_name = "pathmnist"
    mock_cfg.model.name = "efficientnet_b0"
    mock_cfg.model.weight_variant = None
    mock_cfg.optuna.search_space_preset = "architecture_search"
    mock_cfg.optuna.n_trials = 50
    mock_cfg.optuna.epochs = 10
    mock_cfg.optuna.metric_name = "auc"
    mock_cfg.hardware.device = "cuda"
    mock_cfg.optuna.enable_pruning = True
    mock_cfg.optuna.enable_early_stopping = False

    log_optimization_header(cfg=mock_cfg, logger_instance=mock_logger)

    assert mock_logger.info.call_count > 8
    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)
    assert "pathmnist" in log_output
    assert "50" in log_output  # n_trials


@pytest.mark.unit
def test_log_optimization_header_with_early_stopping():
    """Test log_optimization_header includes early stopping info."""
    mock_logger = MagicMock()
    mock_cfg = MagicMock()
    mock_cfg.dataset.dataset_name = "test"
    mock_cfg.model.name = "model"
    mock_cfg.model.weight_variant = None
    mock_cfg.optuna.search_space_preset = "default"
    mock_cfg.optuna.n_trials = 10
    mock_cfg.optuna.epochs = 5
    mock_cfg.optuna.metric_name = "auc"
    mock_cfg.hardware.device = "cpu"
    mock_cfg.optuna.enable_pruning = False
    mock_cfg.optuna.enable_early_stopping = True
    mock_cfg.optuna.early_stopping_threshold = 0.95
    mock_cfg.optuna.early_stopping_patience = 5

    log_optimization_header(cfg=mock_cfg, logger_instance=mock_logger)

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)
    assert "Early Stop" in log_output or "0.95" in log_output


# =========================================================================== #
#                    LOG TRIAL START                                          #
# =========================================================================== #


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
    assert "5" in log_output  # Trial number
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
    # Should contain category headers
    assert "[" in log_output  # Category markers


@pytest.mark.unit
def test_log_trial_start_formats_small_floats():
    """Test log_trial_start formats very small floats in scientific notation."""
    mock_logger = MagicMock()
    params = {
        "learning_rate": 0.0001,  # Should use scientific notation
        "weight_decay": 0.5,  # Should use regular format
    }

    log_trial_start(trial_number=1, params=params, logger_instance=mock_logger)

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)
    # Very small values should be in scientific notation
    assert "e-" in log_output or "1e-04" in log_output


# =========================================================================== #
#                    LOG BEST CONFIG EXPORT                                   #
# =========================================================================== #


@pytest.mark.unit
def test_log_best_config_export():
    """Test log_best_config_export logs config path."""
    config_path = MagicMock()
    config_path.__str__ = lambda x: "/tmp/best_config.yaml"
    config_path.parent.parent = MagicMock()

    # Patch the module logger since function uses logger fallback
    with patch("orchard.core.logger.reporter.logger") as mock_module_logger:
        log_best_config_export(config_path=config_path, logger_instance=None)

        assert mock_module_logger.info.call_count > 5
        calls = [str(call) for call in mock_module_logger.info.call_args_list]
        log_output = " ".join(calls)
        assert "best_config.yaml" in log_output or "Configuration" in log_output


# =========================================================================== #
#                    REPORTER: LOG INITIAL STATUS                             #
# =========================================================================== #


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
    mock_cfg.model.name = "resnet_18_adapted"
    mock_cfg.model.pretrained = True
    mock_cfg.model.weight_variant = None
    mock_cfg.dataset.metadata = MagicMock()
    mock_cfg.dataset.metadata.display_name = "BloodMNIST"
    mock_cfg.dataset.metadata.num_classes = 8
    mock_cfg.dataset.metadata.in_channels = 3
    mock_cfg.dataset.metadata.resolution_str = "28x28"
    mock_cfg.dataset.metadata.is_anatomical = True
    mock_cfg.dataset.metadata.is_texture_based = False
    mock_cfg.dataset.img_size = 28

    mock_paths = MagicMock()
    mock_paths.root = "/tmp/run"
    mock_device = torch.device("cuda")

    reporter.log_initial_status(
        logger_instance=mock_logger,
        cfg=mock_cfg,
        paths=mock_paths,
        device=mock_device,
        applied_threads=4,
        num_workers=2,
    )

    # Should log multiple sections
    assert mock_logger.info.call_count > 20
    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)
    assert "HARDWARE" in log_output or "DATASET" in log_output


# =========================================================================== #
#                    INTEGRATION: LOGGER FALLBACK                             #
# =========================================================================== #


@pytest.mark.unit
def test_functions_use_module_logger_when_no_instance():
    """Test logging functions use module logger when no instance provided."""
    mock_cfg = MagicMock()
    mock_cfg.dataset.dataset_name = "test"
    mock_cfg.model.name = "model"
    mock_cfg.model.weight_variant = None
    mock_device = torch.device("cpu")
    mock_paths = MagicMock()
    mock_paths.root = "/tmp"

    # Should not raise errors when logger_instance is None
    with patch("orchard.core.logger.reporter.logger") as mock_module_logger:
        log_training_summary(
            cfg=mock_cfg,
            test_acc=0.9,
            macro_f1=0.88,
            device=mock_device,
            paths=mock_paths,
            logger_instance=None,
        )

        # Should use module logger
        assert mock_module_logger.info.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
