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

    mock_logger = SimpleNamespace(info=lambda _: None, warning=lambda _: None, debug=lambda _: None)

    manager.prepare_environment(MockConfig(), logger=mock_logger)
    manager.release_resources(MockConfig())


@pytest.mark.integration
def test_release_resources_with_logger(tmp_path):
    """Test release_resources() accepts optional logger."""
    manager = InfrastructureManager()

    class MockHardware:
        allow_process_kill = False
        lock_file_path = tmp_path / "test.lock"

    mock_logger = SimpleNamespace(info=lambda _: None, debug=lambda _: None)

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

        def info(self, msg):
            self.messages.append(("info", msg))

        def warning(self, msg):
            self.messages.append(("warning", msg))

        def debug(self, msg):
            self.messages.append(("debug", msg))

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

        def info(self, msg):
            self.messages.append(("info", msg))

        def debug(self, msg):
            self.messages.append(("debug", msg))

        def warning(self, msg):
            self.messages.append(("warning", msg))

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
        info=lambda msg: None, debug=lambda msg: debug_calls.append(msg), warning=lambda msg: None
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

        def debug(self, msg):
            self.debug_messages.append(msg)

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

        def debug(self, msg):
            self.debug_messages.append(msg)

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
            self.debug_messages = []

        def debug(self, msg):
            self.debug_messages.append(msg)

    logger = MockLogger()
    manager._flush_compute_cache(log=logger)

    assert any("MPS cache cleanup failed" in msg for msg in logger.debug_messages)


@pytest.mark.unit
def test_release_resources_lock_failure(tmp_path):
    """Test release_resources() handles lock release failures.

    Uses mock to simulate release_single_instance raising an exception,
    since the real function catches exceptions internally (correct cleanup behavior).
    This approach works across all Python versions without platform-specific issues.
    """
    manager = InfrastructureManager()

    lock_path = tmp_path / "test.lock"

    class MockHardware:
        allow_process_kill = False
        lock_file_path = lock_path

    # Capture warnings in a list
    warnings = []

    mock_logger = SimpleNamespace(
        info=lambda msg: None,
        debug=lambda msg: None,
        warning=lambda msg: warnings.append(msg),
    )

    config = SimpleNamespace(hardware=MockHardware())

    # Mock release_single_instance to raise an exception
    with patch(
        "orchard.core.config.infrastructure_config.release_single_instance",
        side_effect=PermissionError("Cannot release lock: permission denied"),
    ):
        manager.release_resources(config, logger=mock_logger)

    assert any("Failed to release lock" in msg for msg in warnings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
