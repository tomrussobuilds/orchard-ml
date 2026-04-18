"""
Test Suite for Telemetry & Environment Reporting Engine.

Tests LogStyle constants and Reporter logging methods.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import torch

from orchard.core.logger import (
    LogStyle,
    Reporter,
)


# LOGSTYLE: CONSTANTS
@pytest.mark.unit
def test_logstyle_heavy_separator() -> None:
    """Test LogStyle HEAVY separator is 80 characters."""
    assert len(LogStyle.HEAVY) == 80
    assert LogStyle.HEAVY == "━" * 80


@pytest.mark.unit
def test_logstyle_double_separator() -> None:
    """Test LogStyle DOUBLE separator is 80 characters."""
    assert len(LogStyle.DOUBLE) == 80
    assert LogStyle.DOUBLE == "═" * 80


@pytest.mark.unit
def test_logstyle_light_separator() -> None:
    """Test LogStyle LIGHT separator is 80 characters."""
    assert len(LogStyle.LIGHT) == 80
    assert LogStyle.LIGHT == "─" * 80


@pytest.mark.unit
def test_logstyle_symbols() -> None:
    """Test LogStyle symbols are defined correctly."""
    assert LogStyle.ARROW == "»"
    assert LogStyle.BULLET == "•"
    assert LogStyle.WARNING == "⚠"
    assert LogStyle.SUCCESS == "✓"


@pytest.mark.unit
def test_logstyle_indentation() -> None:
    """Test LogStyle indentation constants."""
    assert LogStyle.INDENT == "  "
    assert LogStyle.DOUBLE_INDENT == "    "
    assert len(LogStyle.DOUBLE_INDENT) == 2 * len(LogStyle.INDENT)


# REPORTER: INITIALIZATION
@pytest.mark.unit
def test_reporter_init() -> None:
    """Test Reporter can be instantiated."""
    reporter = Reporter()
    assert isinstance(reporter, Reporter)


# REPORTER: LOG INITIAL STATUS
@pytest.mark.unit
def test_reporter_log_initial_status() -> None:
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
def test_reporter_log_initial_status_cpu_device() -> None:
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
def test_reporter_log_initial_status_with_weight_variant() -> None:
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
def test_reporter_log_hardware_section_device_fallback() -> None:
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
def test_reporter_log_hardware_section_cuda_device() -> None:
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


# REPORTER: TRACKING SECTION
@pytest.mark.unit
def test_reporter_log_tracking_section_enabled() -> None:
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
def test_reporter_log_tracking_section_disabled() -> None:
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
def test_reporter_log_tracking_section_absent() -> None:
    """Test _log_tracking_section does nothing when tracking is not configured."""
    reporter = Reporter()
    mock_logger = MagicMock()
    mock_cfg = MagicMock(spec=[])

    reporter._log_tracking_section(mock_logger, mock_cfg)

    mock_logger.info.assert_not_called()


# REPORTER: OPTIMIZATION SECTION
@pytest.mark.unit
def test_reporter_log_optimization_section() -> None:
    """Test _log_optimization_section logs optimization parameters."""
    reporter = Reporter()
    mock_logger = MagicMock()
    mock_cfg = MagicMock()
    mock_cfg.optuna.n_trials = 20
    mock_cfg.optuna.epochs = 15
    mock_cfg.training.monitor_metric = "auc"
    mock_cfg.training.monitor_direction = "maximize"
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
def test_reporter_log_optimization_section_absent() -> None:
    """Test _log_optimization_section does nothing when optuna is not configured."""
    reporter = Reporter()
    mock_logger = MagicMock()
    mock_cfg = MagicMock(spec=[])

    reporter._log_optimization_section(mock_logger, mock_cfg)

    mock_logger.info.assert_not_called()


# REPORTER: EXPORT SECTION
@pytest.mark.unit
def test_reporter_log_export_section_basic() -> None:
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
def test_reporter_log_export_section_quantize_type(qtype: str) -> None:
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
def test_reporter_log_export_section_absent() -> None:
    """Test _log_export_section does nothing when export is not configured."""
    reporter = Reporter()
    mock_logger = MagicMock()
    mock_cfg = MagicMock(spec=[])

    reporter._log_export_section(mock_logger, mock_cfg)

    mock_logger.info.assert_not_called()


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


def _find_call_with_label(calls: list[Any], label: str) -> tuple[Any, ...] | None:
    """Find the call tuple that contains a specific label like 'Weights'.

    Logger calls use: logger.info(fmt, I, A, label, value, ...) where
    args[0]=fmt, args[1]=I, args[2]=A, args[3]=label, args[4]=value.
    """
    for c in calls:
        args = c[0]
        if len(args) >= 4 and args[3] is not None and label in str(args[3]):
            return cast(tuple[Any, ...], args)
    return None


# --- log_initial_status: delegation and argument passing ---
@pytest.mark.unit
def test_reporter_initial_status_passes_indent_and_arrow() -> None:
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
def test_reporter_initial_status_passes_applied_threads_and_workers() -> None:
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
def test_reporter_initial_status_lr_formatting() -> None:
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
def test_reporter_initial_status_lr_string_passthrough() -> None:
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
def test_reporter_initial_status_env_init_header() -> None:
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
def test_reporter_initial_status_logs_task_section() -> None:
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
def test_reporter_initial_status_delegates_tracking_cfg() -> None:
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
def test_reporter_initial_status_delegates_optimization_cfg() -> None:
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
def test_reporter_initial_status_delegates_export_cfg() -> None:
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
def test_hardware_section_exact_args_cpu() -> None:
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
def test_hardware_section_fallback_warning_exact_args() -> None:
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
def test_hardware_section_no_fallback_when_cpu_requested() -> None:
    """Kill 'or' mutation: if requested=='cpu' and active=='cpu', no warning."""
    reporter = Reporter()
    log = MagicMock()
    cfg = MagicMock()
    cfg.hardware.device = "cpu"
    device = torch.device("cpu")

    reporter._log_hardware_section(log, cfg, device, 4, 2)

    log.warning.assert_not_called()


@pytest.mark.unit
def test_hardware_section_lower_not_upper() -> None:
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
def test_hardware_section_mps_no_vram() -> None:
    """Kill 'MPS'→'XXmpsXX' mutation in active_type in ('cuda', 'mps')."""
    reporter = Reporter()
    log = MagicMock()
    cfg = MagicMock()
    cfg.hardware.device = "mps"
    device = MagicMock()
    device.type = "mps"
    cast(Any, device).__str__ = lambda _: "mps"

    with patch("orchard.core.logger.env_reporter.get_accelerator_name", return_value="Apple M2"):
        reporter._log_hardware_section(log, cfg, device, 4, 2)

    calls_str = " ".join(str(c) for c in log.info.call_args_list)
    assert "Apple M2" in calls_str
    # MPS should NOT log VRAM
    assert "VRAM" not in calls_str


@pytest.mark.unit
def test_hardware_section_fallback_cpu_string_exact() -> None:
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
def test_dataset_section_exact_indent_arrow() -> None:
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
def test_strategy_section_pretrained_weights() -> None:
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
def test_strategy_section_random_weights() -> None:
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
def test_strategy_section_amp_precision() -> None:
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
def test_strategy_section_fp32_precision() -> None:
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
def test_strategy_section_strict_repro_mode() -> None:
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
def test_strategy_section_standard_repro_mode() -> None:
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
def test_strategy_section_tta_mode_exact() -> None:
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
def test_strategy_section_indent_arrow_not_none() -> None:
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
def test_tracking_section_active_exact() -> None:
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
def test_tracking_section_disabled_exact() -> None:
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
def test_optimization_section_indent_arrow_not_none() -> None:
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
