"""
Test Suite for Progress & Optimization Logging Functions.

Tests log_optimization_summary, log_optimization_header,
log_trial_start, log_pipeline_summary, and their helpers.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import optuna
import pytest
import torch

from orchard.core.logger import (
    LogStyle,
    log_optimization_header,
    log_optimization_summary,
    log_pipeline_summary,
    log_trial_start,
)


# LOG OPTIMIZATION SUMMARY
@pytest.mark.unit
def test_log_optimization_summary_completed_trials() -> None:
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
def test_log_optimization_summary_no_completed_trials() -> None:
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
def test_log_optimization_summary_with_failed_trials() -> None:
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
def test_log_optimization_header_basic() -> None:
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
def test_log_optimization_header_with_early_stopping() -> None:
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
def test_log_optimization_header_logs_only_search_params() -> None:
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
def test_log_optimization_header_early_stop_auto_threshold() -> None:
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
def test_log_optimization_header_with_model_pool() -> None:
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
def test_log_trial_start_basic() -> None:
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
def test_log_trial_start_groups_parameters() -> None:
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
def test_log_trial_start_formats_small_floats() -> None:
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
def test_log_trial_start_with_string_params() -> None:
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
def test_log_trial_start_with_regular_floats() -> None:
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


# LOG PIPELINE SUMMARY
@pytest.mark.unit
def test_log_pipeline_summary_basic() -> None:
    """Test log_pipeline_summary logs basic pipeline completion info."""
    mock_logger = MagicMock()

    log_pipeline_summary(
        test_metrics={"accuracy": 0.9234, "macro_f1": 0.8765},
        best_model_path=Path("/mock/best_model.pth"),
        run_dir=Path("/mock/outputs/run123"),
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
    assert "Macro F1" in log_output


@pytest.mark.unit
def test_log_pipeline_summary_with_onnx_path() -> None:
    """Test log_pipeline_summary includes ONNX path when provided."""
    mock_logger = MagicMock()

    log_pipeline_summary(
        test_metrics={"accuracy": 0.85, "f1": 0.80},
        best_model_path=Path("/mock/best.pth"),
        run_dir=Path("/mock/run"),
        duration="10m 0s",
        onnx_path=Path("/mock/model.onnx"),
        logger_instance=mock_logger,
    )

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)

    assert "ONNX Export" in log_output
    assert "model.onnx" in log_output


@pytest.mark.unit
def test_log_pipeline_summary_without_onnx_path() -> None:
    """Test log_pipeline_summary omits ONNX line when not provided."""
    mock_logger = MagicMock()

    log_pipeline_summary(
        test_metrics={"accuracy": 0.9, "f1": 0.85},
        best_model_path=Path("/mock/best.pth"),
        run_dir=Path("/mock/run"),
        duration="2m 15s",
        onnx_path=None,
        logger_instance=mock_logger,
    )

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)

    assert "ONNX" not in log_output


@pytest.mark.unit
def test_log_pipeline_summary_uses_module_logger_when_none() -> None:
    """Test log_pipeline_summary uses module logger when none provided."""
    with patch("orchard.core.logger.progress.logger") as mock_module_logger:
        log_pipeline_summary(
            test_metrics={"accuracy": 0.9, "f1": 0.85},
            best_model_path=Path("/mock/best.pth"),
            run_dir=Path("/mock/run"),
            duration="1m 0s",
            logger_instance=None,
        )

        assert mock_module_logger.info.call_count > 0


@pytest.mark.unit
def test_log_pipeline_summary_with_auc() -> None:
    """Test log_pipeline_summary includes AUC line when auc metric is present."""
    mock_logger = MagicMock()

    log_pipeline_summary(
        test_metrics={"accuracy": 0.92, "f1": 0.88, "auc": 0.9567},
        best_model_path=Path("/mock/best.pth"),
        run_dir=Path("/mock/run"),
        duration="3m 0s",
        logger_instance=mock_logger,
    )

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)

    assert "Auc" in log_output
    assert "0.9567" in log_output


# _format_param_value
@pytest.mark.unit
def test_format_param_value_small_float() -> None:
    """Small floats (<0.001) use scientific notation."""
    from orchard.core.logger.progress import _format_param_value

    result = _format_param_value(0.0001)
    assert "e-" in result or "E-" in result


@pytest.mark.unit
def test_format_param_value_regular_float() -> None:
    """Regular floats (>=0.001) use 4-decimal notation."""
    from orchard.core.logger.progress import _format_param_value

    assert _format_param_value(0.5) == "0.5000"


@pytest.mark.unit
def test_format_param_value_integer() -> None:
    """Non-float values pass through str()."""
    from orchard.core.logger.progress import _format_param_value

    assert _format_param_value(42) == "42"


@pytest.mark.unit
def test_format_param_value_zero() -> None:
    """Zero is < 0.001, uses scientific notation."""
    from orchard.core.logger.progress import _format_param_value

    assert _format_param_value(0.0) == "0.00e+00"


@pytest.mark.unit
def test_format_param_value_boundary() -> None:
    """Exactly 0.001 is NOT < 0.001, uses decimal notation."""
    from orchard.core.logger.progress import _format_param_value

    assert _format_param_value(0.001) == "0.0010"


@pytest.mark.unit
def test_format_param_value_string() -> None:
    """String values pass through str()."""
    from orchard.core.logger.progress import _format_param_value

    assert _format_param_value("sgd") == "sgd"


# _count_trial_states
@pytest.mark.unit
def test_count_trial_states_all_complete() -> None:
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
def test_count_trial_states_mixed() -> None:
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
def test_count_trial_states_empty() -> None:
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
def test_log_trial_start_exact_category_names() -> None:
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
def test_log_trial_start_optimization_keys() -> None:
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
def test_log_trial_start_loss_keys() -> None:
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
def test_log_trial_start_regularization_keys() -> None:
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
def test_log_trial_start_scheduling_keys() -> None:
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
def test_log_trial_start_augmentation_keys() -> None:
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
def test_log_trial_start_architecture_keys() -> None:
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
def test_log_trial_start_empty_category_not_shown() -> None:
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
def test_log_trial_start_format_string_uses_key_and_value() -> None:
    """Verify each param is logged with key and formatted value."""
    mock_logger = MagicMock()
    params = {"learning_rate": 0.5}

    log_trial_start(trial_number=0, params=params, logger_instance=mock_logger)

    calls = [str(call) for call in mock_logger.info.call_args_list]
    log_output = " ".join(calls)

    assert "learning_rate" in log_output
    assert "0.5000" in log_output  # _format_param_value(0.5) == "0.5000"


@pytest.mark.unit
def test_log_optimization_summary_warning_no_completed() -> None:
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


@pytest.mark.unit
def test_log_optimization_summary_best_trial_value_error() -> None:
    """log.error called when study.best_value raises ValueError."""
    mock_logger = MagicMock()
    mock_study = MagicMock()
    mock_trial = MagicMock(state=optuna.trial.TrialState.COMPLETE)
    mock_study.trials = [mock_trial]
    mock_study.best_value = property(lambda self: (_ for _ in ()).throw(ValueError))
    type(mock_study).best_value = property(lambda self: (_ for _ in ()).throw(ValueError))

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

    mock_logger.error.assert_called_once()
    args = mock_logger.error.call_args[0]
    assert "Best trial lookup failed" in args[0]
    assert args[1] == LogStyle.INDENT
    assert args[2] == LogStyle.WARNING


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
