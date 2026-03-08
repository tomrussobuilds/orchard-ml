"""
Test Suite for InfrastructureManager.

Tests infrastructure resource management, lock file handling,
and compute cache flushing.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from pydantic import ValidationError

from orchard.core.config import HardwareConfig, InfrastructureManager
from orchard.core.paths.constants import LOGGER_NAME, LogStyle


# INFRASTRUCTURE MANAGER: CREATION
@pytest.mark.unit
def test_infrastructure_manager_creation():
    """Test InfrastructureManager can be instantiated."""
    manager = InfrastructureManager()

    assert manager is not None


@pytest.mark.unit
def test_infrastructure_manager_is_singleton_like():
    """Test multiple InfrastructureManager instances are independent."""
    manager1 = InfrastructureManager()
    manager2 = InfrastructureManager()

    assert manager1 is not manager2


# INFRASTRUCTURE MANAGER: LOCK FILE MANAGEMENT
@pytest.mark.integration
def test_prepare_environment_creates_lock(tmp_path):
    """Test prepare_environment() creates lock file."""
    manager = InfrastructureManager()

    class MockHardware:
        allow_process_kill = False
        lock_file_path = tmp_path / "test.lock"

    config = SimpleNamespace(hardware=MockHardware())

    manager.prepare_environment(config)
    assert config.hardware.lock_file_path.exists()
    manager.release_resources(config)


@pytest.mark.integration
def test_release_resources_removes_lock(tmp_path):
    """Test release_resources() removes lock file."""
    manager = InfrastructureManager()

    class MockHardware:
        allow_process_kill = False
        lock_file_path = tmp_path / "test.lock"

    config = SimpleNamespace(hardware=MockHardware())

    manager.prepare_environment(config)
    assert config.hardware.lock_file_path.exists()

    manager.release_resources(config)
    assert not config.hardware.lock_file_path.exists()


@pytest.mark.integration
def test_prepare_environment_with_existing_lock(tmp_path):
    """Test prepare_environment() behavior with existing lock."""
    manager = InfrastructureManager()

    lock_path = tmp_path / "existing.lock"
    lock_path.touch()

    class MockHardware:
        allow_process_kill = False
        lock_file_path = lock_path

    config = SimpleNamespace(hardware=MockHardware())
    manager.prepare_environment(config)

    assert lock_path.exists()

    manager.release_resources(config)


@pytest.mark.integration
def test_release_resources_idempotent(tmp_path):
    """Test release_resources() can be called multiple times safely."""
    manager = InfrastructureManager()

    class MockHardware:
        allow_process_kill = False
        lock_file_path = tmp_path / "test.lock"

    config = SimpleNamespace(hardware=MockHardware())

    manager.prepare_environment(config)

    manager.release_resources(config)
    manager.release_resources(config)

    assert not config.hardware.lock_file_path.exists()


# INFRASTRUCTURE MANAGER: COMPUTE CACHE
@pytest.mark.unit
def test_flush_compute_cache_no_error():
    """Test _flush_compute_cache() runs without error."""
    manager = InfrastructureManager()

    manager._flush_compute_cache()


@pytest.mark.unit
def test_flush_compute_cache_callable():
    """Test _flush_compute_cache() is callable."""
    manager = InfrastructureManager()

    assert callable(manager._flush_compute_cache)


# INFRASTRUCTURE MANAGER: INTEGRATION WITH CONFIG
@pytest.mark.integration
def test_integration_with_hardware_config(tmp_path):
    """Test InfrastructureManager works with real HardwareConfig."""
    manager = InfrastructureManager()

    hw_config = HardwareConfig(project_name="test-integration")
    _ = SimpleNamespace(hardware=hw_config)

    manager._flush_compute_cache()
    assert manager is not None


# INFRASTRUCTURE MANAGER: ERROR HANDLING
@pytest.mark.integration
def test_prepare_environment_with_logger(tmp_path):
    """Test prepare_environment() accepts optional logger."""
    manager = InfrastructureManager()

    class MockHardware:
        allow_process_kill = False
        lock_file_path = tmp_path / "test.lock"

    class MockConfig:
        hardware = MockHardware()

    mock_logger = SimpleNamespace(
        info=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
        debug=lambda *a, **kw: None,
    )

    manager.prepare_environment(MockConfig(), logger=mock_logger)
    manager.release_resources(MockConfig())


@pytest.mark.integration
def test_release_resources_with_logger(tmp_path):
    """Test release_resources() accepts optional logger."""
    manager = InfrastructureManager()

    class MockHardware:
        allow_process_kill = False
        lock_file_path = tmp_path / "test.lock"

    mock_logger = SimpleNamespace(info=lambda *a, **kw: None, debug=lambda *a, **kw: None)

    config = SimpleNamespace(hardware=MockHardware())
    manager.prepare_environment(config)
    manager.release_resources(config, logger=mock_logger)


# INFRASTRUCTURE MANAGER: IMMUTABILITY
@pytest.mark.unit
def test_infrastructure_manager_frozen():
    """Test InfrastructureManager is frozen."""
    manager = InfrastructureManager()

    with pytest.raises(ValidationError):
        manager.new_field = "should_fail"


@pytest.mark.unit
def test_infrastructure_manager_rejects_extra_fields():
    """Test InfrastructureManager forbids extra fields."""
    with pytest.raises(ValidationError):
        InfrastructureManager(foo="bar")


# INTEGRATION: WITH OPTUNA CONFIG
@pytest.mark.integration
def test_optuna_hardware_integration():
    """Test InfrastructureManager works with Optuna-optimized HardwareConfig."""
    from orchard.core.config import OptunaConfig

    hw_config = HardwareConfig(device="cpu", reproducible=True)
    optuna_config = OptunaConfig(n_trials=10)

    assert hw_config.reproducible is True
    assert hw_config.effective_num_workers == 0
    assert optuna_config.n_trials == 10

    manager = InfrastructureManager()
    assert manager is not None


# INFRASTRUCTURE MANAGER: PROCESS CLEANUP
@pytest.mark.integration
def test_prepare_environment_with_process_kill_enabled(tmp_path, monkeypatch):
    """Test prepare_environment() with process kill enabled on non-shared environment."""
    manager = InfrastructureManager()

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.delenv("LSB_JOBID", raising=False)

    class MockHardware:
        allow_process_kill = True
        lock_file_path = tmp_path / "test.lock"

    class MockLogger:
        def __init__(self):
            self.messages = []

        def info(self, msg, *args):
            self.messages.append(("info", msg % args if args else msg))

        def warning(self, msg, *args):
            self.messages.append(("warning", msg % args if args else msg))

        def debug(self, msg, *args):
            self.messages.append(("debug", msg % args if args else msg))

    logger = MockLogger()
    config = SimpleNamespace(hardware=MockHardware())

    manager.prepare_environment(config, logger=logger)

    debug_messages = [msg for level, msg in logger.messages if level == "debug"]
    assert any("Duplicate processes terminated" in msg for msg in debug_messages)

    manager.release_resources(config)


@pytest.mark.integration
def test_prepare_environment_skips_process_kill_on_shared_env(tmp_path, monkeypatch):
    """Test prepare_environment() skips process kill on shared compute."""
    manager = InfrastructureManager()

    monkeypatch.setenv("SLURM_JOB_ID", "12345")

    class MockHardware:
        allow_process_kill = True
        lock_file_path = tmp_path / "test.lock"

    class MockLogger:
        def __init__(self):
            self.messages = []

        def info(self, msg, *args):
            self.messages.append(("info", msg % args if args else msg))

        def debug(self, msg, *args):
            self.messages.append(("debug", msg % args if args else msg))

        def warning(self, msg, *args):
            self.messages.append(("warning", msg % args if args else msg))

    logger = MockLogger()
    config = SimpleNamespace(hardware=MockHardware())

    manager.prepare_environment(config, logger=logger)

    debug_messages = [msg for level, msg in logger.messages if level == "debug"]
    assert any("Shared environment detected" in msg for msg in debug_messages)

    manager.release_resources(config)


@pytest.mark.integration
def test_prepare_environment_with_pbs_environment(tmp_path, monkeypatch):
    """Test prepare_environment() detects PBS environment."""
    manager = InfrastructureManager()

    monkeypatch.setenv("PBS_JOBID", "67890")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    class MockHardware:
        allow_process_kill = True
        lock_file_path = tmp_path / "test.lock"

    debug_calls = []

    mock_logger = SimpleNamespace(
        info=lambda msg, *a: None,
        debug=lambda msg, *a: debug_calls.append(msg % a if a else msg),
        warning=lambda msg, *a: None,
    )

    config = SimpleNamespace(hardware=MockHardware())
    manager.prepare_environment(config, logger=mock_logger)

    assert any("Shared environment detected" in msg for msg in debug_calls)

    manager.release_resources(config, logger=mock_logger)


# INFRASTRUCTURE MANAGER: CACHE FLUSHING
@pytest.mark.unit
def test_flush_compute_cache_with_cuda(monkeypatch):
    """Test _flush_compute_cache() with CUDA available."""
    manager = InfrastructureManager()

    cuda_cache_cleared = False

    def mock_empty_cache():
        nonlocal cuda_cache_cleared
        cuda_cache_cleared = True

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", mock_empty_cache)

    class MockLogger:
        def __init__(self):
            self.debug_messages = []

        def debug(self, msg, *args):
            self.debug_messages.append(msg % args if args else msg)

    logger = MockLogger()
    manager._flush_compute_cache(log=logger)

    assert cuda_cache_cleared
    assert any("CUDA cache cleared" in msg for msg in logger.debug_messages)


@pytest.mark.unit
def test_flush_compute_cache_with_mps(monkeypatch):
    """Test _flush_compute_cache() with MPS available."""
    manager = InfrastructureManager()

    mps_cache_cleared = False

    def mock_mps_empty_cache():
        nonlocal mps_cache_cleared
        mps_cache_cleared = True

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class MockMPSBackend:
        @staticmethod
        def is_available():
            return True

    class MockMPS:
        @staticmethod
        def empty_cache():
            mock_mps_empty_cache()

    if not hasattr(torch, "backends"):
        monkeypatch.setattr(torch, "backends", type("obj", (), {}))

    monkeypatch.setattr(torch.backends, "mps", MockMPSBackend())
    monkeypatch.setattr(torch, "mps", MockMPS())

    class MockLogger:
        def __init__(self):
            self.debug_messages = []

        def debug(self, msg, *args):
            self.debug_messages.append(msg % args if args else msg)

    logger = MockLogger()
    manager._flush_compute_cache(log=logger)

    assert mps_cache_cleared
    assert any("MPS cache cleared" in msg for msg in logger.debug_messages)


@pytest.mark.unit
def test_flush_compute_cache_mps_failure(monkeypatch):
    """Test _flush_compute_cache() handles MPS failures gracefully."""
    manager = InfrastructureManager()

    def mock_mps_empty_cache_fail():
        raise RuntimeError("MPS error")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class MockMPSBackend:
        @staticmethod
        def is_available():
            return True

    class MockMPS:
        @staticmethod
        def empty_cache():
            mock_mps_empty_cache_fail()

    if not hasattr(torch, "backends"):
        monkeypatch.setattr(torch, "backends", type("obj", (), {}))

    monkeypatch.setattr(torch.backends, "mps", MockMPSBackend())
    monkeypatch.setattr(torch, "mps", MockMPS())

    class MockLogger:
        def __init__(self):
            self.warning_messages: list[str] = []
            self.debug_messages: list[str] = []

        def debug(self, msg, *args):
            self.debug_messages.append(msg % args if args else msg)

        def warning(self, msg, *args):
            self.warning_messages.append(msg % args if args else msg)

    logger = MockLogger()
    manager._flush_compute_cache(log=logger)

    assert any("MPS cache cleanup failed" in msg for msg in logger.warning_messages)
    assert any(str(LogStyle.ARROW) in msg for msg in logger.warning_messages)


@pytest.mark.unit
def test_release_resources_lock_failure(tmp_path):
    """Test release_resources() logs and re-raises lock release failures.

    Uses mock to simulate release_single_instance raising an exception.
    The error is logged at ERROR level and then re-raised.
    """
    manager = InfrastructureManager()

    lock_path = tmp_path / "test.lock"

    class MockHardware:
        allow_process_kill = False
        lock_file_path = lock_path

    errors: list[str] = []

    mock_logger = SimpleNamespace(
        info=lambda msg, *a: None,
        debug=lambda msg, *a: None,
        warning=lambda msg, *a: None,
        error=lambda msg, *a: errors.append(msg % a if a else msg),
    )

    config = SimpleNamespace(hardware=MockHardware())

    with patch(
        "orchard.core.config.infrastructure_config.release_single_instance",
        side_effect=PermissionError("Cannot release lock: permission denied"),
    ):
        with pytest.raises(PermissionError, match="Cannot release lock"):
            manager.release_resources(config, logger=mock_logger)

    assert any("Failed to release lock" in msg for msg in errors)


# INFRASTRUCTURE MANAGER: SHARED ENV VAR DETECTION (per-variable)
@pytest.mark.integration
def test_prepare_environment_detects_lsb_jobid(tmp_path, monkeypatch):
    """Test prepare_environment() detects LSF (LSB_JOBID) environment."""
    manager = InfrastructureManager()

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv("LSB_JOBID", "99999")

    class MockHardware:
        allow_process_kill = True
        lock_file_path = tmp_path / "test.lock"

    debug_calls: list[str] = []
    mock_logger = SimpleNamespace(
        info=lambda msg, *a: None,
        debug=lambda msg, *a: debug_calls.append(msg % a if a else msg),
        warning=lambda msg, *a: None,
    )
    config = SimpleNamespace(hardware=MockHardware())
    manager.prepare_environment(config, logger=mock_logger)

    assert any("Shared environment detected" in msg for msg in debug_calls)
    manager.release_resources(config, logger=mock_logger)


@pytest.mark.integration
def test_prepare_environment_detects_rank_env(tmp_path, monkeypatch):
    """Test prepare_environment() detects distributed RANK environment."""
    manager = InfrastructureManager()

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.delenv("LSB_JOBID", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv("RANK", "0")

    class MockHardware:
        allow_process_kill = True
        lock_file_path = tmp_path / "test.lock"

    debug_calls: list[str] = []
    mock_logger = SimpleNamespace(
        info=lambda msg, *a: None,
        debug=lambda msg, *a: debug_calls.append(msg % a if a else msg),
        warning=lambda msg, *a: None,
    )
    config = SimpleNamespace(hardware=MockHardware())
    manager.prepare_environment(config, logger=mock_logger)

    assert any("Shared environment detected" in msg for msg in debug_calls)
    manager.release_resources(config, logger=mock_logger)


@pytest.mark.integration
def test_prepare_environment_detects_local_rank_env(tmp_path, monkeypatch):
    """Test prepare_environment() detects distributed LOCAL_RANK environment."""
    manager = InfrastructureManager()

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.delenv("LSB_JOBID", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setenv("LOCAL_RANK", "0")

    class MockHardware:
        allow_process_kill = True
        lock_file_path = tmp_path / "test.lock"

    debug_calls: list[str] = []
    mock_logger = SimpleNamespace(
        info=lambda msg, *a: None,
        debug=lambda msg, *a: debug_calls.append(msg % a if a else msg),
        warning=lambda msg, *a: None,
    )
    config = SimpleNamespace(hardware=MockHardware())
    manager.prepare_environment(config, logger=mock_logger)

    assert any("Shared environment detected" in msg for msg in debug_calls)
    manager.release_resources(config, logger=mock_logger)


# INFRASTRUCTURE MANAGER: VERIFY FUNCTION CALL ARGUMENTS
@pytest.mark.integration
def test_prepare_environment_calls_ensure_single_instance_with_correct_kwargs(tmp_path):
    """Test that ensure_single_instance receives lock_file and logger kwargs."""
    manager = InfrastructureManager()

    class MockHardware:
        allow_process_kill = False
        lock_file_path = tmp_path / "test.lock"

    config = SimpleNamespace(hardware=MockHardware())

    with patch("orchard.core.config.infrastructure_config.ensure_single_instance") as mock_ensure:
        manager.prepare_environment(config)

        mock_ensure.assert_called_once()
        call_kwargs = mock_ensure.call_args
        assert call_kwargs.kwargs["lock_file"] == tmp_path / "test.lock"
        assert call_kwargs.kwargs["logger"] is not None


@pytest.mark.integration
def test_release_resources_calls_release_single_instance_with_lock_path(tmp_path):
    """Test that release_single_instance receives the correct lock path."""
    manager = InfrastructureManager()

    class MockHardware:
        allow_process_kill = False
        lock_file_path = tmp_path / "test.lock"

    config = SimpleNamespace(hardware=MockHardware())

    with patch("orchard.core.config.infrastructure_config.release_single_instance") as mock_release:
        manager.release_resources(config)

        mock_release.assert_called_once_with(tmp_path / "test.lock")


@pytest.mark.unit
def test_release_resources_calls_flush_compute_cache(tmp_path, monkeypatch):
    """Test that release_resources delegates to _flush_compute_cache with log."""
    manager = InfrastructureManager()

    class MockHardware:
        allow_process_kill = False
        lock_file_path = tmp_path / "test.lock"

    config = SimpleNamespace(hardware=MockHardware())

    flush_calls: list[dict] = []

    def tracking_flush(self, **kwargs):
        flush_calls.append(kwargs)

    with patch("orchard.core.config.infrastructure_config.release_single_instance"):
        monkeypatch.setattr(InfrastructureManager, "_flush_compute_cache", tracking_flush)
        manager.release_resources(config)

    assert len(flush_calls) == 1
    assert flush_calls[0]["log"] is not None


@pytest.mark.unit
def test_prepare_environment_default_logger_fallback(tmp_path):
    """Test prepare_environment uses default logger with LOGGER_NAME when none is passed."""
    manager = InfrastructureManager()

    class MockHardware:
        allow_process_kill = False
        lock_file_path = tmp_path / "test.lock"

    config = SimpleNamespace(hardware=MockHardware())

    with patch("orchard.core.config.infrastructure_config.ensure_single_instance") as mock_ensure:
        manager.prepare_environment(config)
        _, kwargs = mock_ensure.call_args
        assert kwargs["logger"].name == LOGGER_NAME


@pytest.mark.unit
def test_release_resources_default_logger_fallback(tmp_path, monkeypatch):
    """Test release_resources uses default logger with LOGGER_NAME when none is passed."""
    manager = InfrastructureManager()

    class MockHardware:
        allow_process_kill = False
        lock_file_path = tmp_path / "test.lock"

    config = SimpleNamespace(hardware=MockHardware())

    flush_calls: list[dict] = []

    def tracking_flush(self, **kwargs):
        flush_calls.append(kwargs)

    with patch("orchard.core.config.infrastructure_config.release_single_instance"):
        monkeypatch.setattr(InfrastructureManager, "_flush_compute_cache", tracking_flush)
        manager.release_resources(config)

    assert len(flush_calls) == 1
    assert flush_calls[0]["log"].name == LOGGER_NAME


@pytest.mark.unit
def test_flush_compute_cache_default_logger_fallback(monkeypatch):
    """Test _flush_compute_cache uses default logger with LOGGER_NAME when None is passed."""
    manager = InfrastructureManager()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("orchard.core.config.infrastructure_config.has_mps_backend", lambda: False)

    with patch("logging.getLogger") as mock_get_logger:
        mock_get_logger.return_value = SimpleNamespace(
            debug=lambda *a: None, warning=lambda *a: None
        )
        manager._flush_compute_cache(log=None)
        mock_get_logger.assert_called_once_with(LOGGER_NAME)


@pytest.mark.integration
def test_prepare_environment_process_kill_calls_terminate_duplicates(tmp_path, monkeypatch):
    """Test that terminate_duplicates is actually called on non-shared env."""
    manager = InfrastructureManager()

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.delenv("LSB_JOBID", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    class MockHardware:
        allow_process_kill = True
        lock_file_path = tmp_path / "test.lock"

    config = SimpleNamespace(hardware=MockHardware())

    with (
        patch("orchard.core.config.infrastructure_config.DuplicateProcessCleaner") as mock_cls,
        patch("orchard.core.config.infrastructure_config.ensure_single_instance"),
    ):
        mock_instance = mock_cls.return_value
        mock_instance.terminate_duplicates.return_value = 0

        manager.prepare_environment(config)

        mock_instance.terminate_duplicates.assert_called_once()


@pytest.mark.integration
def test_prepare_environment_any_shared_env_skips_kill(tmp_path, monkeypatch):
    """Test that ANY single shared env var is enough to skip process kill (any vs all)."""
    manager = InfrastructureManager()

    # Only set ONE env var — if mutmut changes any() to all(), this would fail
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.delenv("LSB_JOBID", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv("RANK", "0")

    class MockHardware:
        allow_process_kill = True
        lock_file_path = tmp_path / "test.lock"

    config = SimpleNamespace(hardware=MockHardware())

    with (
        patch("orchard.core.config.infrastructure_config.DuplicateProcessCleaner") as mock_cls,
        patch("orchard.core.config.infrastructure_config.ensure_single_instance"),
    ):
        mock_instance = mock_cls.return_value

        manager.prepare_environment(config)

        # terminate_duplicates should NOT have been called (shared env detected)
        mock_instance.terminate_duplicates.assert_not_called()


# MUTATION KILLERS: verify exact log levels and message content
@pytest.mark.unit
def test_prepare_environment_lock_acquired_message(tmp_path, monkeypatch):
    """Test that 'Lock acquired at <path>' is logged at DEBUG level."""
    manager = InfrastructureManager()

    class MockHardware:
        allow_process_kill = False
        lock_file_path = tmp_path / "test.lock"

    config = SimpleNamespace(hardware=MockHardware())

    debug_calls: list[str] = []
    mock_logger = SimpleNamespace(
        info=lambda msg, *a: None,
        debug=lambda msg, *a: debug_calls.append(msg % a if a else msg),
        warning=lambda msg, *a: None,
    )

    with patch("orchard.core.config.infrastructure_config.ensure_single_instance"):
        manager.prepare_environment(config, logger=mock_logger)

    assert any("Lock acquired" in msg for msg in debug_calls)
    assert any(str(tmp_path / "test.lock") in msg for msg in debug_calls)


@pytest.mark.unit
def test_release_resources_info_level_for_lock_released(tmp_path):
    """Test that 'System lock released' is logged at INFO level (not debug)."""
    manager = InfrastructureManager()

    class MockHardware:
        allow_process_kill = False
        lock_file_path = tmp_path / "test.lock"

    config = SimpleNamespace(hardware=MockHardware())

    info_calls: list[str] = []
    debug_calls: list[str] = []

    mock_logger = SimpleNamespace(
        info=lambda msg, *a: info_calls.append(msg % a if a else msg),
        debug=lambda msg, *a: debug_calls.append(msg % a if a else msg),
        warning=lambda msg, *a: None,
    )

    with patch("orchard.core.config.infrastructure_config.release_single_instance"):
        manager.release_resources(config, logger=mock_logger)

    assert any("System lock released" in msg for msg in info_calls)
    # Must be at INFO, not DEBUG
    assert not any("System lock released" in msg for msg in debug_calls)


@pytest.mark.unit
def test_release_resources_error_level_for_lock_failure(tmp_path):
    """Test that lock release failure is logged at ERROR level and re-raised."""
    manager = InfrastructureManager()

    class MockHardware:
        allow_process_kill = False
        lock_file_path = tmp_path / "test.lock"

    config = SimpleNamespace(hardware=MockHardware())

    error_calls: list[str] = []
    info_calls: list[str] = []

    mock_logger = SimpleNamespace(
        info=lambda msg, *a: info_calls.append(msg % a if a else msg),
        debug=lambda msg, *a: None,
        warning=lambda msg, *a: None,
        error=lambda msg, *a: error_calls.append(msg % a if a else msg),
    )

    with patch(
        "orchard.core.config.infrastructure_config.release_single_instance",
        side_effect=OSError("lock error"),
    ):
        with pytest.raises(OSError, match="lock error"):
            manager.release_resources(config, logger=mock_logger)

    assert any("Failed to release lock" in msg for msg in error_calls)
    assert not any("Failed to release lock" in msg for msg in info_calls)
    # Verify LogStyle.ARROW and exception text appear in the message
    assert any(str(LogStyle.ARROW) in msg for msg in error_calls)
    assert any("lock error" in msg for msg in error_calls)


@pytest.mark.integration
def test_prepare_environment_terminate_duplicates_receives_logger(tmp_path, monkeypatch):
    """Test that terminate_duplicates is called with logger=log kwarg."""
    manager = InfrastructureManager()

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.delenv("LSB_JOBID", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    class MockHardware:
        allow_process_kill = True
        lock_file_path = tmp_path / "test.lock"

    config = SimpleNamespace(hardware=MockHardware())

    mock_logger = SimpleNamespace(
        info=lambda msg, *a: None,
        debug=lambda msg, *a: None,
        warning=lambda msg, *a: None,
    )

    with (
        patch("orchard.core.config.infrastructure_config.DuplicateProcessCleaner") as mock_cls,
        patch("orchard.core.config.infrastructure_config.ensure_single_instance"),
    ):
        mock_instance = mock_cls.return_value
        mock_instance.terminate_duplicates.return_value = 0

        manager.prepare_environment(config, logger=mock_logger)

        # Verify logger kwarg was passed
        call_kwargs = mock_instance.terminate_duplicates.call_args
        assert call_kwargs.kwargs["logger"] is mock_logger


@pytest.mark.integration
def test_prepare_environment_num_zombies_in_message(tmp_path, monkeypatch):
    """Test that the actual number of terminated zombies appears in log message."""
    manager = InfrastructureManager()

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.delenv("LSB_JOBID", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    class MockHardware:
        allow_process_kill = True
        lock_file_path = tmp_path / "test.lock"

    config = SimpleNamespace(hardware=MockHardware())

    debug_calls: list[str] = []
    mock_logger = SimpleNamespace(
        info=lambda msg, *a: None,
        debug=lambda msg, *a: debug_calls.append(msg % a if a else msg),
        warning=lambda msg, *a: None,
    )

    with (
        patch("orchard.core.config.infrastructure_config.DuplicateProcessCleaner") as mock_cls,
        patch("orchard.core.config.infrastructure_config.ensure_single_instance"),
    ):
        mock_instance = mock_cls.return_value
        mock_instance.terminate_duplicates.return_value = 3

        manager.prepare_environment(config, logger=mock_logger)

    # The message must include the actual count returned by terminate_duplicates
    assert any("3" in msg and "Duplicate processes terminated" in msg for msg in debug_calls)


@pytest.mark.unit
def test_flush_compute_cache_no_cuda_no_mps_no_log(monkeypatch):
    """Test _flush_compute_cache does not log when neither CUDA nor MPS available."""
    manager = InfrastructureManager()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("orchard.core.config.infrastructure_config.has_mps_backend", lambda: False)

    debug_calls: list[str] = []

    class MockLogger:
        def debug(self, msg, *args):
            debug_calls.append(msg % args if args else msg)

    manager._flush_compute_cache(log=MockLogger())

    # No cache-related messages when neither backend is available
    assert not any("cache" in msg.lower() for msg in debug_calls)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
