"""
Tests suite for RootOrchestrator and TimeTracker.
Tests all phases, __enter__, __exit__, and edge cases.
Achieves high coverage through dependency injection and mocking.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from orchard.core import LOGGER_NAME, Config, RootOrchestrator, TimeTracker


# TIMETRACKER TESTS
@pytest.mark.unit
def test_timetracker_init_state():
    """Test __init__ sets _start_time and _end_time to None (not empty string)."""
    tracker = TimeTracker()
    assert tracker._start_time is None
    assert tracker._end_time is None


@pytest.mark.unit
def test_timetracker_elapsed_seconds_before_start():
    """Test elapsed_seconds returns 0 before start() is called."""
    tracker = TimeTracker()
    assert tracker.elapsed_seconds == pytest.approx(0.0)


@pytest.mark.unit
def test_timetracker_elapsed_formatted_seconds():
    """Test elapsed_formatted returns seconds format for < 1 minute."""
    tracker = TimeTracker()
    tracker._start_time = time.time() - 45.5  # 45.5 seconds ago
    tracker._end_time = time.time()

    result = tracker.elapsed_formatted
    assert "s" in result
    assert "m" not in result
    assert "h" not in result


@pytest.mark.unit
def test_timetracker_elapsed_formatted_minutes():
    """Test elapsed_formatted returns minutes format for >= 1 minute."""
    tracker = TimeTracker()
    tracker._start_time = time.time() - 125  # 2 minutes 5 seconds ago
    tracker._end_time = time.time()

    result = tracker.elapsed_formatted
    assert "m" in result
    assert "s" in result
    assert "h" not in result
    assert result == "2m 5s"


@pytest.mark.unit
def test_timetracker_elapsed_formatted_hours():
    """Test elapsed_formatted returns hours format for >= 1 hour."""
    tracker = TimeTracker()
    tracker._start_time = time.time() - 3725  # 1 hour 2 minutes 5 seconds
    tracker._end_time = time.time()

    result = tracker.elapsed_formatted
    assert "h" in result
    assert "m" in result
    assert "s" in result
    assert result == "1h 2m 5s"


@pytest.mark.unit
def test_timetracker_start_resets_end_time():
    """Test start() resets _end_time to None."""
    tracker = TimeTracker()
    tracker._end_time = time.time()

    tracker.start()

    assert tracker._end_time is None
    assert tracker._start_time is not None


@pytest.mark.unit
def test_timetracker_start_sets_float_start_time():
    """Test start() sets _start_time to a float (not None)."""
    tracker = TimeTracker()
    tracker.start()
    assert isinstance(tracker._start_time, float)


@pytest.mark.unit
def test_timetracker_stop_returns_elapsed():
    """Test stop() returns elapsed seconds."""
    tracker = TimeTracker()
    tracker.start()
    time.sleep(0.01)  # Small delay

    elapsed = tracker.stop()

    assert elapsed > 0
    assert isinstance(tracker._end_time, float)


# ORCHESTRATOR: INITIALIZATION
@pytest.mark.unit
def test_orchestrator_init_with_defaults():
    """Test RootOrchestrator initializes with default dependencies."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 4

    orch = RootOrchestrator(cfg=mock_cfg)

    assert orch.cfg == mock_cfg
    assert orch.repro_mode is False
    assert orch.warn_only_mode is False
    assert orch.num_workers == 4
    assert orch.paths is None
    assert orch.run_logger is None


@pytest.mark.unit
def test_orchestrator_init_extracts_policies():
    """Test RootOrchestrator extracts policies from config."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = True
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 8

    orch = RootOrchestrator(cfg=mock_cfg)

    assert orch.repro_mode is True
    assert orch.warn_only_mode is False
    assert orch.num_workers == 8


@pytest.mark.unit
def test_init_lazy_attributes_and_policy_extraction():
    """Test __init__ sets lazy attributes and extracts policy flags."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = True
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 5

    orch = RootOrchestrator(cfg=mock_cfg)

    assert orch.paths is None
    assert orch.run_logger is None
    assert orch._device_cache is None
    assert orch.repro_mode is True
    assert orch.warn_only_mode is False
    assert orch.num_workers == 5


# CONTEXT MANAGER: __ENTER__
@pytest.mark.unit
def test_context_manager_enter():
    """Test __enter__ calls initialize_core_services."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 4

    orch = RootOrchestrator(cfg=mock_cfg)
    orch.initialize_core_services = MagicMock(return_value=MagicMock())

    result = orch.__enter__()

    orch.initialize_core_services.assert_called_once()
    assert result == orch


@pytest.mark.unit
def test_context_manager_enter_exception_cleanup():
    """Test __enter__ calls cleanup on exception."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 4

    orch = RootOrchestrator(cfg=mock_cfg)
    orch.initialize_core_services = MagicMock(side_effect=RuntimeError("Init failed"))
    orch.cleanup = MagicMock()

    with pytest.raises(RuntimeError, match="Init failed"):
        orch.__enter__()

    orch.cleanup.assert_called_once()


# CONTEXT MANAGER: __EXIT__
@pytest.mark.unit
def test_context_manager_exit_calls_cleanup():
    """Test __exit__ calls cleanup."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 4

    orch = RootOrchestrator(cfg=mock_cfg)
    orch.cleanup = MagicMock()

    result = orch.__exit__(None, None, None)

    orch.cleanup.assert_called_once()
    assert result is False


@pytest.mark.unit
def test_context_manager_exit_propagates_exception():
    """Test __exit__ returns False to propagate exceptions."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 4

    orch = RootOrchestrator(cfg=mock_cfg)
    orch.cleanup = MagicMock()

    result = orch.__exit__(ValueError, ValueError("test"), None)

    assert result is False


@pytest.mark.unit
def test_context_manager_exit_stops_timer():
    """Test __exit__ stops the time tracker and runs cleanup."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 4

    mock_time_tracker = MagicMock()

    orch = RootOrchestrator(cfg=mock_cfg, time_tracker=mock_time_tracker)
    orch.cleanup = MagicMock()

    orch.__exit__(None, None, None)

    mock_time_tracker.stop.assert_called_once()
    orch.cleanup.assert_called_once()


# GET DEVICE
@pytest.mark.unit
def test_get_device_returns_cpu():
    """Test get_device returns CPU device."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.device = "cpu"
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 4

    orch = RootOrchestrator(cfg=mock_cfg)

    device = orch.get_device()

    assert device.type == "cpu"


@pytest.mark.unit
def test_get_device_caches_result():
    """Test get_device caches device object."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.device = "cpu"
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 4

    orch = RootOrchestrator(cfg=mock_cfg)

    device1 = orch.get_device()
    device2 = orch.get_device()

    assert device1 is device2


@pytest.mark.unit
def test_get_device_calls_resolver_when_cache_none():
    """Test get_device calls device resolver when _device_cache is None."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.device = "cpu"
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 1
    mock_resolver = MagicMock(return_value=torch.device("cpu"))

    orch = RootOrchestrator(cfg=mock_cfg, device_resolver=mock_resolver)
    orch._device_cache = None

    device = orch.get_device()

    mock_resolver.assert_called_once_with(device_str="cpu", local_rank=0)
    assert device.type == "cpu"


# CLEANUP
@pytest.mark.unit
def test_cleanup_handles_infra_release_exception(monkeypatch):
    """Test cleanup handles exceptions raised by infra.release_resources."""
    mock_cfg = MagicMock()
    mock_infra = MagicMock()
    mock_infra.release_resources.side_effect = RuntimeError("release failed")
    mock_logger = MagicMock()
    mock_handler = MagicMock()
    mock_logger.handlers = [mock_handler]

    orch = RootOrchestrator(cfg=mock_cfg, infra_manager=mock_infra)
    orch.run_logger = mock_logger

    orch.cleanup()

    mock_handler.close.assert_called_once()
    mock_logger.removeHandler.assert_called_once_with(mock_handler)


@pytest.mark.unit
def test_cleanup_release_resources_fails_no_logger(caplog):
    """Test cleanup logs error if release_resources fails and no logger."""
    import logging

    mock_cfg = MagicMock()
    mock_infra = MagicMock()
    mock_infra.release_resources.side_effect = RuntimeError("fail")

    orch = RootOrchestrator(cfg=mock_cfg)
    orch.infra = mock_infra
    orch._infra_lock_acquired = True
    orch.run_logger = None

    # Enable propagate for caplog to capture fallback logger
    fallback_logger = logging.getLogger("OrchardML")
    fallback_logger.propagate = True

    with caplog.at_level(logging.ERROR, logger="OrchardML"):
        orch.cleanup()

    assert any("fail" in rec.message for rec in caplog.records)


# ORCHESTRATOR: PHASES 1-7
@pytest.mark.unit
def test_phase_1_determinism_always_calls_seed_setter():
    """Test _phase_1_determinism always calls seed setter."""
    mock_cfg = MagicMock()
    mock_cfg.training.seed = 123
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_seed_setter = MagicMock()
    orch = RootOrchestrator(cfg=mock_cfg, seed_setter=mock_seed_setter)

    orch._phase_1_determinism()
    mock_seed_setter.assert_called_once_with(123, strict=False, warn_only=False)


@pytest.mark.unit
def test_phase_2_runtime_configuration_applies_threads_and_system():
    mock_cfg = MagicMock()
    mock_cfg.hardware.effective_num_workers = 4
    mock_thread_applier = MagicMock(return_value=7)
    mock_system_configurator = MagicMock()

    orch = RootOrchestrator(
        cfg=mock_cfg,
        thread_applier=mock_thread_applier,
        system_configurator=mock_system_configurator,
    )
    threads = orch._phase_2_runtime_configuration()
    assert threads == 7
    mock_thread_applier.assert_called_once_with(4)
    mock_system_configurator.assert_called_once()


@pytest.mark.unit
def test_phase_3_filesystem_provisioning_calls_static_setup_and_runpaths():
    import orchard.core.orchestrator as orch_module

    orig_create = orch_module.RunPaths.create
    orch_module.RunPaths.create = MagicMock(return_value="runpaths_obj")

    mock_cfg = MagicMock()
    mock_cfg.dataset.dataset_name = "ds"
    mock_cfg.architecture.name = "architecture"
    mock_cfg.telemetry.output_dir = "/mock/out"
    mock_cfg.dump_serialized = MagicMock(return_value={"some": "data"})
    mock_static_setup = MagicMock()

    orch = RootOrchestrator(cfg=mock_cfg, static_dir_setup=mock_static_setup)
    orch._phase_3_filesystem_provisioning()

    mock_static_setup.assert_called_once()
    orch_module.RunPaths.create.assert_called_once_with(
        dataset_slug="ds",
        architecture_name="architecture",
        training_cfg={"some": "data"},
        base_dir="/mock/out",
    )
    assert orch.paths == "runpaths_obj"
    orch_module.RunPaths.create = orig_create


@pytest.mark.unit
def test_phase_4_logging_initialization_sets_logger():
    mock_cfg = MagicMock()
    mock_cfg.telemetry.log_level = "INFO"
    orch = RootOrchestrator(cfg=mock_cfg)
    orch.paths = MagicMock()
    orch.paths.logs = "/mock/logs"
    mock_log_initializer = MagicMock(return_value="logger_obj")
    orch._log_initializer = mock_log_initializer

    orch._phase_4_logging_initialization()

    mock_log_initializer.assert_called_once_with(
        name=LOGGER_NAME,
        log_dir="/mock/logs",
        level="INFO",
    )
    assert orch.run_logger == "logger_obj"


@pytest.mark.unit
def test_phase_5_run_manifest_saves_config(tmp_path):
    mock_cfg = MagicMock()
    mock_audit = MagicMock()
    orch = RootOrchestrator(cfg=mock_cfg, audit_saver=mock_audit)
    orch.paths = MagicMock()
    orch.paths.get_config_path = MagicMock(return_value="/mock/config.yaml")
    orch.paths.reports = tmp_path

    orch._phase_5_run_manifest()
    mock_audit.save_config.assert_called_once_with(data=mock_cfg, yaml_path="/mock/config.yaml")
    mock_audit.dump_requirements.assert_called_once_with(tmp_path / "requirements.txt")


@pytest.mark.unit
def test_phase_6_infra_prepare_raises_on_failure_no_logger():
    """Test _phase_6_infrastructure_guarding raises OrchardInfrastructureError on failure."""
    from orchard.exceptions import OrchardInfrastructureError

    mock_cfg = MagicMock()
    mock_infra = MagicMock()
    mock_infra.prepare_environment.side_effect = RuntimeError("fail")

    orch = RootOrchestrator(cfg=mock_cfg)
    orch.infra = mock_infra
    orch.run_logger = None

    with pytest.raises(OrchardInfrastructureError, match="fail"):
        orch._phase_6_infrastructure_guarding()

    assert orch._infra_lock_acquired is False


@pytest.mark.unit
def test_device_resolver_fails_raises_device_error_during_init():
    """Test initialize_core_services raises OrchardDeviceError if device resolver fails."""
    from orchard.exceptions import OrchardDeviceError

    mock_cfg = MagicMock()
    mock_cfg.hardware.effective_num_workers = 2
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.device = "cuda"
    mock_paths = MagicMock()

    def failing_resolver(**kwargs):
        raise RuntimeError("device fail")

    with pytest.MonkeyPatch.context() as m:
        m.setattr("orchard.core.orchestrator.RunPaths.create", lambda **kw: mock_paths)

        orch = RootOrchestrator(
            cfg=mock_cfg,
            device_resolver=failing_resolver,
            seed_setter=MagicMock(),
            thread_applier=MagicMock(return_value=4),
            system_configurator=MagicMock(),
            static_dir_setup=MagicMock(),
            log_initializer=MagicMock(return_value=MagicMock()),
            audit_saver=MagicMock(),
            infra_manager=MagicMock(),
        )

        with pytest.raises(OrchardDeviceError, match="Device resolution failed at runtime"):
            orch.initialize_core_services()


# CLEANUP: EDGE CASES
@pytest.mark.unit
def test_cleanup_with_no_infra_manager():
    """Test cleanup when infra manager is None."""
    mock_cfg = MagicMock()
    mock_logger = MagicMock()

    orch = RootOrchestrator(cfg=mock_cfg, infra_manager=None)
    orch.run_logger = mock_logger

    try:
        orch.cleanup()
    except Exception as e:
        pytest.fail(f"cleanup() raised an exception: {e}")


@pytest.mark.unit
def test_cleanup_closes_logger_handlers_on_infra_failure():
    """Test cleanup closes logger handlers when infra.release_resources fails."""
    mock_cfg = MagicMock()
    mock_infra = MagicMock()
    mock_infra.release_resources.side_effect = RuntimeError("release failed")

    mock_handler1 = MagicMock()
    mock_handler2 = MagicMock()
    mock_logger = MagicMock()
    mock_logger.handlers = [mock_handler1, mock_handler2]

    orch = RootOrchestrator(cfg=mock_cfg, infra_manager=mock_infra)
    orch._infra_lock_acquired = True
    orch.run_logger = mock_logger

    orch.cleanup()

    mock_handler1.close.assert_called_once()
    mock_handler2.close.assert_called_once()

    assert mock_logger.removeHandler.call_count == 2
    mock_logger.removeHandler.assert_any_call(mock_handler1)
    mock_logger.removeHandler.assert_any_call(mock_handler2)


@pytest.mark.unit
def test_cleanup_with_empty_handlers_list():
    """Test cleanup when logger has no handlers."""
    mock_cfg = MagicMock()
    mock_infra = MagicMock()
    mock_infra.release_resources.side_effect = RuntimeError("fail")

    mock_logger = MagicMock()
    mock_logger.handlers = []

    orch = RootOrchestrator(cfg=mock_cfg, infra_manager=mock_infra)
    orch._infra_lock_acquired = True
    orch.run_logger = mock_logger

    orch.cleanup()


# GET DEVICE: ADDITIONAL EDGE CASES
@pytest.mark.unit
def test_get_device_with_cuda_string():
    """Test get_device with 'cuda' device string."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.device = "cuda"
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 4

    mock_resolver = MagicMock(return_value=torch.device("cuda"))

    orch = RootOrchestrator(cfg=mock_cfg, device_resolver=mock_resolver)

    device = orch.get_device()

    mock_resolver.assert_called_once_with(device_str="cuda", local_rank=0)
    assert device.type == "cuda"


@pytest.mark.unit
def test_get_device_with_mps_string():
    """Test get_device with 'mps' device string."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.device = "mps"
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 4

    mock_resolver = MagicMock(return_value=torch.device("mps"))

    orch = RootOrchestrator(cfg=mock_cfg, device_resolver=mock_resolver)

    device = orch.get_device()

    mock_resolver.assert_called_once_with(device_str="mps", local_rank=0)
    assert device.type == "mps"


# PHASE 7: DEVICE RESOLUTION EDGE CASES
@pytest.mark.unit
def test_phase_7_device_already_cached_with_logger(caplog):
    """Test _phase_7 when device is already cached."""
    import logging

    mock_cfg = MagicMock()
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 2
    mock_reporter = MagicMock()
    mock_logger = MagicMock()

    orch = RootOrchestrator(cfg=mock_cfg, reporter=mock_reporter)
    orch._device_cache = torch.device("cpu")
    orch.run_logger = mock_logger
    orch.paths = MagicMock()

    with caplog.at_level(logging.WARNING):
        orch._phase_7_environment_report(applied_threads=1)

    assert orch._device_cache.type == "cpu"
    mock_reporter.log_initial_status.assert_called_once()


@pytest.mark.unit
def test_phase_7_uses_cached_device():
    """Test _phase_7 uses pre-resolved device cache for reporting."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 2
    mock_reporter = MagicMock()
    mock_logger = MagicMock()

    orch = RootOrchestrator(cfg=mock_cfg, reporter=mock_reporter)
    orch._device_cache = torch.device("cuda")
    orch.run_logger = mock_logger
    orch.paths = MagicMock()

    orch._phase_7_environment_report(applied_threads=4)

    mock_reporter.log_initial_status.assert_called_once()
    call_kwargs = mock_reporter.log_initial_status.call_args[1]
    assert call_kwargs["device"] == torch.device("cuda")
    assert call_kwargs["applied_threads"] == 4


# PHASE 6: INFRASTRUCTURE GUARDING EDGE CASES
@pytest.mark.unit
def test_phase_6_prepare_fails_with_logger():
    """Test _phase_6 raises OrchardInfrastructureError when prepare_environment fails."""
    from orchard.exceptions import OrchardInfrastructureError

    mock_cfg = MagicMock()
    mock_infra = MagicMock()
    mock_infra.prepare_environment.side_effect = RuntimeError("lock failed")
    mock_logger = MagicMock()

    orch = RootOrchestrator(cfg=mock_cfg, infra_manager=mock_infra)
    orch.run_logger = mock_logger

    with pytest.raises(OrchardInfrastructureError, match="lock failed"):
        orch._phase_6_infrastructure_guarding()

    assert orch._infra_lock_acquired is False


# INTEGRATION: FULL LIFECYCLE
@pytest.mark.integration
def test_full_lifecycle_with_all_phases(tmp_path):
    """Test complete initialization through all phases."""
    mock_cfg = MagicMock()
    mock_cfg.training.seed = 42
    mock_cfg.hardware.use_deterministic_algorithms = True
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 2
    mock_cfg.hardware.device = "cpu"
    mock_cfg.dataset.dataset_name = "test_dataset"
    mock_cfg.architecture.name = "test_architecture"
    mock_cfg.telemetry.output_dir = str(tmp_path)
    mock_cfg.telemetry.log_level = "INFO"
    mock_cfg.dump_serialized = MagicMock(return_value={"test": "data"})

    # Mock all dependencies
    mock_infra = MagicMock()
    mock_reporter = MagicMock()
    mock_seed_setter = MagicMock()
    mock_thread_applier = MagicMock(return_value=2)
    mock_system_configurator = MagicMock()
    mock_static_setup = MagicMock()
    mock_device_resolver = MagicMock(return_value=torch.device("cpu"))

    # Create mock RunPaths
    mock_paths = MagicMock()
    mock_paths.logs = str(tmp_path / "logs")
    mock_paths.get_config_path = MagicMock(return_value=str(tmp_path / "config.yaml"))

    # Create mock logger
    mock_logger = MagicMock()

    with pytest.MonkeyPatch.context() as m:
        # Mock RunPaths.create
        m.setattr("orchard.core.orchestrator.RunPaths.create", lambda **kwargs: mock_paths)

        # Mock Logger.setup
        mock_log_initializer = MagicMock(return_value=mock_logger)

        orch = RootOrchestrator(
            cfg=mock_cfg,
            infra_manager=mock_infra,
            reporter=mock_reporter,
            log_initializer=mock_log_initializer,
            seed_setter=mock_seed_setter,
            thread_applier=mock_thread_applier,
            system_configurator=mock_system_configurator,
            static_dir_setup=mock_static_setup,
            audit_saver=MagicMock(),
            device_resolver=mock_device_resolver,
        )

        # Initialize all services
        paths = orch.initialize_core_services()

        # Phase 7 is deferred — call it explicitly (like CLI does after tracker.start_run)
        orch.log_environment_report()

        # Verify all phases were executed
        mock_seed_setter.assert_called_once_with(42, strict=True, warn_only=False)
        mock_thread_applier.assert_called_once_with(2)
        mock_system_configurator.assert_called_once()
        mock_static_setup.assert_called_once()
        mock_log_initializer.assert_called_once()
        mock_infra.prepare_environment.assert_called_once()
        mock_reporter.log_initial_status.assert_called_once()

        assert paths == mock_paths
        assert orch.run_logger == mock_logger


@pytest.mark.integration
def test_context_manager_full_lifecycle():
    """Test full lifecycle using context manager."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 1

    mock_infra = MagicMock()

    orch = RootOrchestrator(cfg=mock_cfg, infra_manager=mock_infra)
    orch.initialize_core_services = MagicMock(return_value=MagicMock())
    orch._infra_lock_acquired = True

    with orch as orchestrator:
        assert orchestrator == orch
        orch.initialize_core_services.assert_called_once()

    mock_infra.release_resources.assert_called_once()


# INTEGRATION: REAL DEPENDENCIES
def _make_integration_cfg(tmp_path):
    """Build a minimal real Config rooted at tmp_path."""
    return Config(
        dataset={"name": "bloodmnist", "resolution": 28},
        architecture={"name": "mini_cnn", "pretrained": False},
        training={"epochs": 1, "mixup_epochs": 0, "use_amp": False},
        hardware={"device": "cpu", "project_name": "test-integration"},
        telemetry={"output_dir": str(tmp_path)},
    )


@pytest.mark.integration
def test_integration_phases_1_through_4_real_filesystem(tmp_path):
    """Phases 1-4 with real set_seed, RunPaths.create, Logger.setup."""
    cfg = _make_integration_cfg(tmp_path)
    mock_infra = MagicMock()

    orch = RootOrchestrator(
        cfg=cfg,
        infra_manager=mock_infra,
        audit_saver=MagicMock(),
        reporter=MagicMock(),
        static_dir_setup=lambda: None,
    )
    paths = orch.initialize_core_services()

    assert paths is not None
    assert paths.root.exists()
    assert paths.logs.exists()
    assert paths.checkpoints.exists()
    assert paths.reports.exists()
    assert list(paths.logs.glob("*.log"))
    assert orch.run_logger is not None
    assert orch._initialized is True
    mock_infra.prepare_environment.assert_called_once()

    orch.cleanup()


@pytest.mark.integration
def test_integration_context_manager_lifecycle(tmp_path):
    """Context manager creates real dirs and cleans up logging on exit."""
    cfg = _make_integration_cfg(tmp_path)
    mock_infra = MagicMock()

    with RootOrchestrator(
        cfg=cfg,
        infra_manager=mock_infra,
        audit_saver=MagicMock(),
        reporter=MagicMock(),
        static_dir_setup=lambda: None,
    ) as orch:
        assert orch.paths is not None
        assert orch.paths.root.exists()
        assert orch.run_logger is not None

    assert orch._cleaned_up is True
    assert orch.run_logger is None
    mock_infra.release_resources.assert_called_once()


@pytest.mark.integration
def test_integration_phase_5_writes_real_config_yaml(tmp_path):
    """Real AuditSaver.save_config writes a parseable config.yaml."""
    from orchard.core.io import load_config_from_yaml
    from orchard.core.io.serialization import AuditSaver

    cfg = _make_integration_cfg(tmp_path)
    audit = AuditSaver()

    with patch.object(audit, "dump_requirements"):
        orch = RootOrchestrator(
            cfg=cfg,
            infra_manager=MagicMock(),
            audit_saver=audit,
            reporter=MagicMock(),
            static_dir_setup=lambda: None,
        )
        orch.initialize_core_services()

    config_yaml = orch.paths.get_config_path()
    assert config_yaml.exists()
    loaded = load_config_from_yaml(config_yaml)
    assert loaded["dataset"]["name"] == "bloodmnist"

    orch.cleanup()


@pytest.mark.integration
def test_integration_phase_7_deferred_report(tmp_path):
    """Reporter called only after explicit log_environment_report()."""
    cfg = _make_integration_cfg(tmp_path)
    mock_reporter = MagicMock()

    orch = RootOrchestrator(
        cfg=cfg,
        infra_manager=MagicMock(),
        audit_saver=MagicMock(),
        reporter=mock_reporter,
        static_dir_setup=lambda: None,
    )
    orch.initialize_core_services()

    mock_reporter.log_initial_status.assert_not_called()
    orch.log_environment_report()
    mock_reporter.log_initial_status.assert_called_once()

    call_kwargs = mock_reporter.log_initial_status.call_args
    assert call_kwargs.kwargs["device"] == torch.device("cpu")

    orch.cleanup()


@pytest.mark.integration
def test_integration_idempotent_single_run_dir(tmp_path):
    """Two initialize_core_services() calls create only one run directory."""
    cfg = _make_integration_cfg(tmp_path)

    orch = RootOrchestrator(
        cfg=cfg,
        infra_manager=MagicMock(),
        audit_saver=MagicMock(),
        reporter=MagicMock(),
        static_dir_setup=lambda: None,
    )
    paths1 = orch.initialize_core_services()
    paths2 = orch.initialize_core_services()

    assert paths1 is paths2
    run_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
    assert len(run_dirs) == 1

    orch.cleanup()


# IDEMPOTENCY
@pytest.mark.unit
def test_initialize_core_services_idempotent_returns_cached_paths(tmp_path):
    """Test that calling initialize_core_services twice returns same paths without re-executing phases."""
    mock_cfg = MagicMock()
    mock_cfg.training.seed = 42
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 2
    mock_cfg.hardware.device = "cpu"
    mock_cfg.dataset.dataset_name = "ds"
    mock_cfg.architecture.name = "arch"
    mock_cfg.telemetry.output_dir = str(tmp_path)
    mock_cfg.telemetry.log_level = "INFO"
    mock_cfg.dump_serialized = MagicMock(return_value={"k": "v"})

    mock_paths = MagicMock()
    mock_paths.logs = str(tmp_path / "logs")
    mock_paths.get_config_path = MagicMock(return_value=str(tmp_path / "config.yaml"))

    mock_seed_setter = MagicMock()
    mock_thread_applier = MagicMock(return_value=2)
    mock_infra = MagicMock()
    mock_reporter = MagicMock()
    mock_log_initializer = MagicMock(return_value=MagicMock())
    mock_audit_saver = MagicMock()

    with pytest.MonkeyPatch.context() as m:
        m.setattr("orchard.core.orchestrator.RunPaths.create", lambda **kwargs: mock_paths)

        orch = RootOrchestrator(
            cfg=mock_cfg,
            infra_manager=mock_infra,
            reporter=mock_reporter,
            log_initializer=mock_log_initializer,
            seed_setter=mock_seed_setter,
            thread_applier=mock_thread_applier,
            system_configurator=MagicMock(),
            static_dir_setup=MagicMock(),
            audit_saver=mock_audit_saver,
            device_resolver=MagicMock(return_value=torch.device("cpu")),
        )

        # First call: full initialization
        paths1 = orch.initialize_core_services()

        # Reset mocks to verify second call does NOT re-invoke phases
        mock_seed_setter.reset_mock()
        mock_thread_applier.reset_mock()
        mock_infra.reset_mock()
        mock_reporter.reset_mock()
        mock_log_initializer.reset_mock()
        mock_audit_saver.reset_mock()

        # Second call: should return cached paths
        paths2 = orch.initialize_core_services()

        assert paths1 is paths2
        assert paths2 is mock_paths

        # No phases re-executed
        mock_seed_setter.assert_not_called()
        mock_thread_applier.assert_not_called()
        mock_infra.prepare_environment.assert_not_called()
        mock_reporter.log_initial_status.assert_not_called()
        mock_log_initializer.assert_not_called()
        mock_audit_saver.save_config.assert_not_called()


@pytest.mark.unit
def test_initialize_core_services_skips_when_already_initialized():
    """Test that initialize_core_services returns immediately when _initialized is True."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 2

    existing_paths = MagicMock()
    mock_seed_setter = MagicMock()

    orch = RootOrchestrator(cfg=mock_cfg, seed_setter=mock_seed_setter)
    orch.paths = existing_paths
    orch._initialized = True

    result = orch.initialize_core_services()

    assert result is existing_paths
    mock_seed_setter.assert_not_called()


# RANK-AWARENESS: INITIALIZATION
@pytest.mark.unit
def test_orchestrator_rank_defaults_to_zero(monkeypatch):
    """Test rank defaults to 0 when RANK env var is not set."""
    monkeypatch.delenv("RANK", raising=False)
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 4

    orch = RootOrchestrator(cfg=mock_cfg)

    assert orch.rank == 0
    assert orch.is_main_process is True


@pytest.mark.unit
def test_orchestrator_rank_from_env(monkeypatch):
    """Test rank is read from RANK env var."""
    monkeypatch.setenv("RANK", "3")
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 4

    orch = RootOrchestrator(cfg=mock_cfg)

    assert orch.rank == 3
    assert orch.is_main_process is False


@pytest.mark.unit
def test_orchestrator_rank_injectable():
    """Test rank can be injected directly, overriding env var."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 4

    orch = RootOrchestrator(cfg=mock_cfg, rank=2)

    assert orch.rank == 2
    assert orch.is_main_process is False


# RANK-AWARENESS: PHASE GATING
@pytest.mark.unit
def test_rank_zero_executes_all_phases(tmp_path):
    """Test rank 0 executes all phases."""
    mock_cfg = MagicMock()
    mock_cfg.training.seed = 42
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 2
    mock_cfg.hardware.device = "cpu"
    mock_cfg.dataset.dataset_name = "ds"
    mock_cfg.architecture.name = "arch"
    mock_cfg.telemetry.output_dir = str(tmp_path)
    mock_cfg.telemetry.log_level = "INFO"
    mock_cfg.dump_serialized = MagicMock(return_value={"k": "v"})

    mock_paths = MagicMock()
    mock_paths.logs = str(tmp_path / "logs")
    mock_paths.get_config_path = MagicMock(return_value=str(tmp_path / "config.yaml"))

    mock_seed_setter = MagicMock()
    mock_thread_applier = MagicMock(return_value=2)
    mock_infra = MagicMock()
    mock_reporter = MagicMock()
    mock_log_initializer = MagicMock(return_value=MagicMock())
    mock_audit_saver = MagicMock()

    with pytest.MonkeyPatch.context() as m:
        m.setattr("orchard.core.orchestrator.RunPaths.create", lambda **kwargs: mock_paths)

        orch = RootOrchestrator(
            cfg=mock_cfg,
            infra_manager=mock_infra,
            reporter=mock_reporter,
            log_initializer=mock_log_initializer,
            seed_setter=mock_seed_setter,
            thread_applier=mock_thread_applier,
            system_configurator=MagicMock(),
            static_dir_setup=MagicMock(),
            audit_saver=mock_audit_saver,
            device_resolver=MagicMock(return_value=torch.device("cpu")),
            rank=0,
        )

        paths = orch.initialize_core_services()
        orch.log_environment_report()

    # All phases executed
    mock_seed_setter.assert_called_once()
    mock_thread_applier.assert_called_once()
    mock_log_initializer.assert_called_once()
    mock_audit_saver.save_config.assert_called_once()
    mock_infra.prepare_environment.assert_called_once()
    mock_reporter.log_initial_status.assert_called_once()
    assert paths is mock_paths


@pytest.mark.unit
def test_non_main_rank_skips_phases_3_through_7():
    """Test non-main rank only executes phases 1-2, skipping 3-7."""
    mock_cfg = MagicMock()
    mock_cfg.training.seed = 42
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 2

    mock_seed_setter = MagicMock()
    mock_thread_applier = MagicMock(return_value=2)
    mock_infra = MagicMock()
    mock_reporter = MagicMock()
    mock_log_initializer = MagicMock()
    mock_audit_saver = MagicMock()
    mock_static_dir_setup = MagicMock()

    orch = RootOrchestrator(
        cfg=mock_cfg,
        infra_manager=mock_infra,
        reporter=mock_reporter,
        audit_saver=mock_audit_saver,
        log_initializer=mock_log_initializer,
        seed_setter=mock_seed_setter,
        thread_applier=mock_thread_applier,
        system_configurator=MagicMock(),
        static_dir_setup=mock_static_dir_setup,
        device_resolver=MagicMock(return_value=torch.device("cpu")),
        rank=1,
    )

    result = orch.initialize_core_services()

    # Phases 1-2 executed
    mock_seed_setter.assert_called_once_with(42, strict=False, warn_only=False)
    mock_thread_applier.assert_called_once_with(2)

    # Phases 3-6 skipped
    mock_static_dir_setup.assert_not_called()
    mock_log_initializer.assert_not_called()
    mock_audit_saver.save_config.assert_not_called()
    mock_infra.prepare_environment.assert_not_called()
    mock_reporter.log_initial_status.assert_not_called()

    # Device still resolved for DDP readiness
    assert orch._device_cache is not None

    # paths and run_logger remain None
    assert result is None
    assert orch.paths is None
    assert orch.run_logger is None


@pytest.mark.unit
def test_non_main_rank_cleanup_is_noop():
    """Test cleanup is a no-op for non-main ranks."""
    mock_cfg = MagicMock()
    mock_infra = MagicMock()
    mock_logger = MagicMock()
    mock_handler = MagicMock()
    mock_logger.handlers = [mock_handler]

    orch = RootOrchestrator(cfg=mock_cfg, infra_manager=mock_infra, rank=1)
    orch.run_logger = mock_logger

    orch.cleanup()

    mock_infra.release_resources.assert_not_called()
    mock_handler.close.assert_not_called()


@pytest.mark.unit
def test_rank_zero_cleanup_releases_resources():
    """Test cleanup releases resources for rank 0."""
    mock_cfg = MagicMock()
    mock_infra = MagicMock()
    mock_handler = MagicMock()
    mock_logger = MagicMock()
    mock_logger.handlers = [mock_handler]

    orch = RootOrchestrator(cfg=mock_cfg, infra_manager=mock_infra, rank=0)
    orch._infra_lock_acquired = True
    orch.run_logger = mock_logger

    orch.cleanup()

    mock_infra.release_resources.assert_called_once()
    mock_handler.close.assert_called_once()


@pytest.mark.unit
def test_non_main_rank_context_manager_lifecycle():
    """Test full context manager lifecycle for non-main rank."""
    mock_cfg = MagicMock()
    mock_cfg.training.seed = 42
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 2

    mock_infra = MagicMock()
    mock_seed_setter = MagicMock()
    mock_thread_applier = MagicMock(return_value=2)

    orch = RootOrchestrator(
        cfg=mock_cfg,
        infra_manager=mock_infra,
        seed_setter=mock_seed_setter,
        thread_applier=mock_thread_applier,
        system_configurator=MagicMock(),
        device_resolver=MagicMock(return_value=torch.device("cpu")),
        rank=1,
    )

    with orch as orchestrator:
        assert orchestrator.rank == 1
        assert orchestrator.paths is None
        assert orchestrator.run_logger is None

    # No resource release on non-main rank
    mock_infra.release_resources.assert_not_called()


# ORCHESTRATOR: INITIAL STATE AND KWARGS VERIFICATION
@pytest.mark.unit
def test_init_initialized_starts_false():
    """Test _initialized flag starts as False."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 4

    orch = RootOrchestrator(cfg=mock_cfg)

    assert orch._initialized is False


@pytest.mark.unit
def test_init_applied_threads_starts_zero():
    """Test _applied_threads starts as 0."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 4

    orch = RootOrchestrator(cfg=mock_cfg)

    assert orch._applied_threads == 0


@pytest.mark.unit
def test_phase_7_reporter_receives_all_kwargs():
    """Test _phase_7_environment_report passes all kwargs to reporter.log_initial_status."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 3
    mock_reporter = MagicMock()
    mock_logger = MagicMock()
    mock_paths = MagicMock()

    orch = RootOrchestrator(cfg=mock_cfg, reporter=mock_reporter)
    orch._device_cache = torch.device("cpu")
    orch.run_logger = mock_logger
    orch.paths = mock_paths

    orch._phase_7_environment_report(applied_threads=8)

    kw = mock_reporter.log_initial_status.call_args.kwargs
    assert kw["logger_instance"] is mock_logger
    assert kw["cfg"] is mock_cfg
    assert kw["paths"] is mock_paths
    assert kw["device"] == torch.device("cpu")
    assert kw["applied_threads"] == 8
    assert kw["num_workers"] == 3


@pytest.mark.unit
def test_cleanup_error_message_content():
    """Test cleanup logs specific error message when release fails."""
    mock_cfg = MagicMock()
    mock_infra = MagicMock()
    mock_infra.release_resources.side_effect = OSError("disk full")
    mock_logger = MagicMock()
    mock_logger.handlers = []

    orch = RootOrchestrator(cfg=mock_cfg, infra_manager=mock_infra)
    orch._infra_lock_acquired = True
    orch.run_logger = mock_logger

    orch.cleanup()

    mock_logger.error.assert_called_once()
    assert "Failed to release system lock" in mock_logger.error.call_args[0][0]
    assert "disk full" in str(mock_logger.error.call_args[0][1])


@pytest.mark.unit
def test_log_environment_report_noop_on_non_main_rank():
    """Test log_environment_report is a no-op for non-main ranks."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 2
    mock_reporter = MagicMock()

    orch = RootOrchestrator(cfg=mock_cfg, reporter=mock_reporter, rank=1)
    orch.paths = MagicMock()
    orch.run_logger = MagicMock()

    orch.log_environment_report()

    mock_reporter.log_initial_status.assert_not_called()


@pytest.mark.unit
def test_applied_threads_stored_from_phase_2(tmp_path):
    """Test _applied_threads is set from phase 2 return value."""
    mock_cfg = MagicMock()
    mock_cfg.training.seed = 42
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 2
    mock_cfg.dataset.dataset_name = "ds"
    mock_cfg.architecture.name = "arch"
    mock_cfg.telemetry.output_dir = str(tmp_path)
    mock_cfg.telemetry.log_level = "INFO"
    mock_cfg.dump_serialized = MagicMock(return_value={})

    mock_paths = MagicMock()
    mock_paths.logs = str(tmp_path / "logs")
    mock_paths.get_config_path = MagicMock(return_value=str(tmp_path / "cfg.yaml"))

    mock_thread_applier = MagicMock(return_value=12)

    with pytest.MonkeyPatch.context() as m:
        m.setattr("orchard.core.orchestrator.RunPaths.create", lambda **kw: mock_paths)

        orch = RootOrchestrator(
            cfg=mock_cfg,
            seed_setter=MagicMock(),
            thread_applier=mock_thread_applier,
            system_configurator=MagicMock(),
            static_dir_setup=MagicMock(),
            audit_saver=MagicMock(),
            log_initializer=MagicMock(return_value=MagicMock()),
            infra_manager=MagicMock(),
            device_resolver=MagicMock(return_value=torch.device("cpu")),
            rank=0,
        )

        orch.initialize_core_services()

    assert orch._applied_threads == 12


@pytest.mark.unit
def test_applied_threads_stored_non_main_rank():
    """Test _applied_threads is set even for non-main ranks."""
    mock_cfg = MagicMock()
    mock_cfg.training.seed = 42
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 2

    mock_thread_applier = MagicMock(return_value=6)

    orch = RootOrchestrator(
        cfg=mock_cfg,
        seed_setter=MagicMock(),
        thread_applier=mock_thread_applier,
        system_configurator=MagicMock(),
        device_resolver=MagicMock(return_value=torch.device("cpu")),
        rank=1,
    )

    orch.initialize_core_services()

    assert orch._applied_threads == 6


@pytest.mark.unit
def test_phase_6_prepare_environment_receives_cfg_and_logger():
    """Test _phase_6 passes cfg and logger to infra.prepare_environment."""
    mock_cfg = MagicMock()
    mock_infra = MagicMock()
    mock_logger = MagicMock()

    orch = RootOrchestrator(cfg=mock_cfg, infra_manager=mock_infra)
    orch.run_logger = mock_logger

    orch._phase_6_infrastructure_guarding()

    mock_infra.prepare_environment.assert_called_once_with(mock_cfg, logger=mock_logger)


@pytest.mark.unit
def test_cleanup_passes_cfg_and_logger_to_release():
    """Test cleanup passes cfg and logger to infra.release_resources."""
    mock_cfg = MagicMock()
    mock_infra = MagicMock()
    mock_logger = MagicMock()
    mock_logger.handlers = []

    orch = RootOrchestrator(cfg=mock_cfg, infra_manager=mock_infra, rank=0)
    orch._infra_lock_acquired = True
    orch.run_logger = mock_logger

    orch.cleanup()

    mock_infra.release_resources.assert_called_once_with(mock_cfg, logger=mock_logger)


@pytest.mark.unit
def test_initialized_flag_set_after_init():
    """Test _initialized is True after initialize_core_services completes."""
    mock_cfg = MagicMock()
    mock_cfg.training.seed = 42
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 2

    orch = RootOrchestrator(
        cfg=mock_cfg,
        seed_setter=MagicMock(),
        thread_applier=MagicMock(return_value=2),
        system_configurator=MagicMock(),
        device_resolver=MagicMock(return_value=torch.device("cpu")),
        rank=1,
    )

    assert orch._initialized is False
    orch.initialize_core_services()
    assert orch._initialized is True


@pytest.mark.unit
def test_phase_5_audit_saver_receives_data_and_path():
    """Test _phase_5 passes data= and yaml_path= to audit_saver."""
    mock_cfg = MagicMock()
    mock_audit = MagicMock()
    mock_paths = MagicMock()
    mock_paths.get_config_path.return_value = "/mock/config.yaml"
    mock_paths.reports = MagicMock()

    orch = RootOrchestrator(cfg=mock_cfg, audit_saver=mock_audit)
    orch.paths = mock_paths

    orch._phase_5_run_manifest()

    mock_audit.save_config.assert_called_once_with(data=mock_cfg, yaml_path="/mock/config.yaml")
    mock_audit.dump_requirements.assert_called_once_with(mock_paths.reports / "requirements.txt")


# GUARD: log_environment_report no-op when not initialized
@pytest.mark.unit
def test_log_environment_report_noop_when_not_initialized():
    """Test log_environment_report is a no-op when _initialized is False."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 2
    mock_reporter = MagicMock()

    orch = RootOrchestrator(cfg=mock_cfg, reporter=mock_reporter, rank=0)
    orch.paths = MagicMock()
    orch.run_logger = MagicMock()
    # _initialized is False by default

    orch.log_environment_report()

    mock_reporter.log_initial_status.assert_not_called()


# GUARD: _cleaned_up blocks re-initialization
@pytest.mark.unit
def test_cleaned_up_blocks_reinitialize():
    """Test initialize_core_services raises RuntimeError after cleanup."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 2

    orch = RootOrchestrator(cfg=mock_cfg, rank=0)
    orch._cleaned_up = True

    with pytest.raises(RuntimeError, match="Cannot re-initialize after cleanup"):
        orch.initialize_core_services()


@pytest.mark.unit
def test_cleanup_sets_cleaned_up_flag():
    """Test cleanup sets _cleaned_up to True."""
    mock_cfg = MagicMock()
    mock_infra = MagicMock()
    mock_logger = MagicMock()
    mock_logger.handlers = []

    orch = RootOrchestrator(cfg=mock_cfg, infra_manager=mock_infra, rank=0)
    orch.run_logger = mock_logger

    assert orch._cleaned_up is False
    orch.cleanup()
    assert orch._cleaned_up is True


# GUARD: run_logger nulled after cleanup
@pytest.mark.unit
def test_cleanup_nulls_run_logger():
    """Test cleanup sets run_logger to None after closing handlers."""
    mock_cfg = MagicMock()
    mock_infra = MagicMock()
    mock_handler = MagicMock()
    mock_logger = MagicMock()
    mock_logger.handlers = [mock_handler]

    orch = RootOrchestrator(cfg=mock_cfg, infra_manager=mock_infra, rank=0)
    orch.run_logger = mock_logger

    orch.cleanup()

    assert orch.run_logger is None
    mock_handler.close.assert_called_once()


# GUARD: _infra_lock_acquired flag
@pytest.mark.unit
def test_infra_lock_acquired_true_on_success():
    """Test _infra_lock_acquired is True when prepare_environment succeeds."""
    mock_cfg = MagicMock()
    mock_infra = MagicMock()
    mock_logger = MagicMock()

    orch = RootOrchestrator(cfg=mock_cfg, infra_manager=mock_infra)
    orch.run_logger = mock_logger

    orch._phase_6_infrastructure_guarding()

    assert orch._infra_lock_acquired is True


@pytest.mark.unit
def test_infra_lock_acquired_false_on_failure():
    """Test _infra_lock_acquired stays False when prepare_environment fails."""
    from orchard.exceptions import OrchardInfrastructureError

    mock_cfg = MagicMock()
    mock_infra = MagicMock()
    mock_infra.prepare_environment.side_effect = OSError("permission denied")
    mock_logger = MagicMock()

    orch = RootOrchestrator(cfg=mock_cfg, infra_manager=mock_infra)
    orch.run_logger = mock_logger

    with pytest.raises(OrchardInfrastructureError, match="permission denied"):
        orch._phase_6_infrastructure_guarding()

    assert orch._infra_lock_acquired is False


@pytest.mark.unit
def test_infra_lock_acquired_false_by_default():
    """Test _infra_lock_acquired starts as False."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 4

    orch = RootOrchestrator(cfg=mock_cfg)

    assert orch._infra_lock_acquired is False


# LOCAL_RANK AND WARN_ONLY SUPPORT
@pytest.mark.unit
def test_orchestrator_local_rank_defaults_to_zero():
    """Test local_rank defaults to 0 when LOCAL_RANK env var is not set."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 4

    orch = RootOrchestrator(cfg=mock_cfg)

    assert orch.local_rank == 0


@pytest.mark.unit
def test_orchestrator_local_rank_from_env(monkeypatch):
    """Test local_rank is read from LOCAL_RANK env var."""
    monkeypatch.setenv("LOCAL_RANK", "2")
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 4

    orch = RootOrchestrator(cfg=mock_cfg)

    assert orch.local_rank == 2


@pytest.mark.unit
def test_orchestrator_local_rank_injectable():
    """Test local_rank can be injected directly."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 4

    orch = RootOrchestrator(cfg=mock_cfg, local_rank=3)

    assert orch.local_rank == 3


@pytest.mark.unit
def test_orchestrator_stores_warn_only_mode():
    """Test warn_only_mode is extracted from config."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = True
    mock_cfg.hardware.deterministic_warn_only = True
    mock_cfg.hardware.effective_num_workers = 0

    orch = RootOrchestrator(cfg=mock_cfg)

    assert orch.warn_only_mode is True


@pytest.mark.unit
def test_phase_1_passes_warn_only_true():
    """Test _phase_1_determinism passes warn_only=True to seed setter."""
    mock_cfg = MagicMock()
    mock_cfg.training.seed = 42
    mock_cfg.hardware.use_deterministic_algorithms = True
    mock_cfg.hardware.deterministic_warn_only = True
    mock_cfg.hardware.effective_num_workers = 0
    mock_seed_setter = MagicMock()

    orch = RootOrchestrator(cfg=mock_cfg, seed_setter=mock_seed_setter)
    orch._phase_1_determinism()

    mock_seed_setter.assert_called_once_with(42, strict=True, warn_only=True)


@pytest.mark.unit
def test_get_device_passes_local_rank():
    """Test get_device passes local_rank to device resolver."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.device = "cuda"
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 4
    mock_resolver = MagicMock(return_value=torch.device("cuda:1"))

    orch = RootOrchestrator(cfg=mock_cfg, device_resolver=mock_resolver, local_rank=1)
    device = orch.get_device()

    mock_resolver.assert_called_once_with(device_str="cuda", local_rank=1)
    assert device == torch.device("cuda:1")


@pytest.mark.unit
def test_non_main_rank_resolves_device():
    """Test non-main ranks resolve their device during initialization."""
    mock_cfg = MagicMock()
    mock_cfg.training.seed = 42
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 2
    mock_cfg.hardware.device = "cpu"
    mock_resolver = MagicMock(return_value=torch.device("cpu"))

    orch = RootOrchestrator(
        cfg=mock_cfg,
        seed_setter=MagicMock(),
        thread_applier=MagicMock(return_value=2),
        system_configurator=MagicMock(),
        device_resolver=mock_resolver,
        rank=1,
        local_rank=1,
    )

    orch.initialize_core_services()

    mock_resolver.assert_called_once_with(device_str="cpu", local_rank=1)
    assert orch._device_cache == torch.device("cpu")


@pytest.mark.unit
def test_non_main_rank_device_failure_raises():
    """Test non-main rank device resolution failure raises OrchardDeviceError."""
    from orchard.exceptions import OrchardDeviceError

    mock_cfg = MagicMock()
    mock_cfg.training.seed = 42
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.deterministic_warn_only = False
    mock_cfg.hardware.effective_num_workers = 2
    mock_cfg.hardware.device = "cuda"

    def failing_resolver(**kwargs):
        raise RuntimeError("GPU gone")

    orch = RootOrchestrator(
        cfg=mock_cfg,
        seed_setter=MagicMock(),
        thread_applier=MagicMock(return_value=2),
        system_configurator=MagicMock(),
        device_resolver=failing_resolver,
        rank=1,
        local_rank=1,
    )

    with pytest.raises(OrchardDeviceError, match="Device resolution failed"):
        orch.initialize_core_services()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
