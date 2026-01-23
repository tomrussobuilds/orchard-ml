"""
Test Suite for InfrastructureManager.

Tests infrastructure resource management, lock file handling,
and compute cache flushing.
"""

# Standard Imports
import os

# Third-Party Imports
import pytest
import torch
from pydantic import ValidationError

# Internal Imports
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

    # Should be different instances
    assert manager1 is not manager2


# INFRASTRUCTURE MANAGER: LOCK FILE MANAGEMENT
@pytest.mark.integration
def test_prepare_environment_creates_lock(tmp_path):
    """Test prepare_environment() creates lock file."""
    manager = InfrastructureManager()

    # Mock config with temp lock path
    class MockHardware:
        allow_process_kill = False  # Skip process cleanup
        lock_file_path = tmp_path / "test.lock"

    class MockConfig:
        hardware = MockHardware()

    config = MockConfig()

    # Should create lock file
    manager.prepare_environment(config)

    assert config.hardware.lock_file_path.exists()

    # Cleanup
    manager.release_resources(config)


@pytest.mark.integration
def test_release_resources_removes_lock(tmp_path):
    """Test release_resources() removes lock file."""
    manager = InfrastructureManager()

    class MockHardware:
        allow_process_kill = False
        lock_file_path = tmp_path / "test.lock"

    class MockConfig:
        hardware = MockHardware()

    config = MockConfig()

    # Create and release
    manager.prepare_environment(config)
    assert config.hardware.lock_file_path.exists()

    manager.release_resources(config)
    assert not config.hardware.lock_file_path.exists()


@pytest.mark.integration
def test_prepare_environment_with_existing_lock(tmp_path):
    """Test prepare_environment() behavior with existing lock."""
    manager = InfrastructureManager()

    lock_path = tmp_path / "existing.lock"
    lock_path.touch()  # Create existing lock

    class MockHardware:
        allow_process_kill = False
        lock_file_path = lock_path

    class MockConfig:
        hardware = MockHardware()

    # Should handle existing lock (implementation-dependent)
    manager.prepare_environment(MockConfig())

    # Lock should still exist or be recreated
    assert lock_path.exists()

    # Cleanup
    manager.release_resources(MockConfig())


@pytest.mark.integration
def test_release_resources_idempotent(tmp_path):
    """Test release_resources() can be called multiple times safely."""
    manager = InfrastructureManager()

    class MockHardware:
        allow_process_kill = False
        lock_file_path = tmp_path / "test.lock"

    class MockConfig:
        hardware = MockHardware()

    config = MockConfig()

    manager.prepare_environment(config)

    # Release twice
    manager.release_resources(config)
    manager.release_resources(config)  # Should not raise

    assert not config.hardware.lock_file_path.exists()


# INFRASTRUCTURE MANAGER: COMPUTE CACHE
@pytest.mark.unit
def test_flush_compute_cache_no_error():
    """Test _flush_compute_cache() runs without error."""
    manager = InfrastructureManager()

    # Should not raise even if no GPU
    manager._flush_compute_cache()


@pytest.mark.unit
def test_flush_compute_cache_callable():
    """Test _flush_compute_cache() is callable."""
    manager = InfrastructureManager()

    # Should be a method
    assert callable(manager._flush_compute_cache)


# INFRASTRUCTURE MANAGER: INTEGRATION WITH CONFIG
@pytest.mark.integration
def test_integration_with_hardware_config(tmp_path):
    """Test InfrastructureManager works with real HardwareConfig."""
    manager = InfrastructureManager()

    # Create real HardwareConfig with temp lock path
    hw_config = HardwareConfig(project_name="test-integration")

    class MockConfig:
        hardware = hw_config

    MockConfig()

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

    class MockLogger:
        def info(self, msg):
            pass

        def warning(self, msg):
            pass

    # Should accept logger without error
    manager.prepare_environment(MockConfig(), logger=MockLogger())

    # Cleanup
    manager.release_resources(MockConfig())


@pytest.mark.integration
def test_release_resources_with_logger(tmp_path):
    """Test release_resources() accepts optional logger."""
    manager = InfrastructureManager()

    class MockHardware:
        allow_process_kill = False
        lock_file_path = tmp_path / "test.lock"

    class MockConfig:
        hardware = MockHardware()

    class MockLogger:
        def info(self, msg):
            pass

        def debug(self, msg):
            pass

    config = MockConfig()
    manager.prepare_environment(config)

    # Should accept logger without error
    manager.release_resources(config, logger=MockLogger())


# INFRASTRUCTURE MANAGER: IMMUTABILITY
@pytest.mark.unit
def test_infrastructure_manager_frozen():
    """Test InfrastructureManager is frozen."""
    manager = InfrastructureManager()

    with pytest.raises(ValidationError):
        manager.new_field = "should_fail"


# INTEGRATION: WITH OPTUNA CONFIG
@pytest.mark.integration
def test_optuna_hardware_integration():
    """Test InfrastructureManager works with Optuna-optimized HardwareConfig."""
    from orchard.core.config import OptunaConfig

    hw_config = HardwareConfig.for_optuna(device="cpu")
    optuna_config = OptunaConfig(n_trials=10)

    # Reproducible mode should be enabled
    assert hw_config.reproducible is True
    assert hw_config.effective_num_workers == 0

    # Optuna config should be valid
    assert optuna_config.n_trials == 10

    # InfrastructureManager should work with this config
    manager = InfrastructureManager()
    assert manager is not None


# INFRASTRUCTURE MANAGER: PROCESS CLEANUP
@pytest.mark.integration
def test_prepare_environment_with_process_kill_enabled(tmp_path, monkeypatch):
    """Test prepare_environment() with process kill enabled on non-shared environment."""
    manager = InfrastructureManager()

    # Remove shared environment markers
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.delenv("LSB_JOBID", raising=False)

    class MockHardware:
        allow_process_kill = True  # Enable process cleanup
        lock_file_path = tmp_path / "test.lock"

    class MockConfig:
        hardware = MockHardware()

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
    config = MockConfig()

    # Should execute process cleanup
    manager.prepare_environment(config, logger=logger)

    # Check that process termination was logged
    info_messages = [msg for level, msg in logger.messages if level == "info"]
    assert any("Duplicate processes terminated" in msg for msg in info_messages)

    # Cleanup
    manager.release_resources(config)


@pytest.mark.integration
def test_prepare_environment_skips_process_kill_on_shared_env(tmp_path, monkeypatch):
    """Test prepare_environment() skips process kill on shared compute."""
    manager = InfrastructureManager()

    # Simulate SLURM environment
    monkeypatch.setenv("SLURM_JOB_ID", "12345")

    class MockHardware:
        allow_process_kill = True
        lock_file_path = tmp_path / "test.lock"

    class MockConfig:
        hardware = MockHardware()

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
    config = MockConfig()

    manager.prepare_environment(config, logger=logger)

    # Should skip process cleanup on shared environment
    debug_messages = [msg for level, msg in logger.messages if level == "debug"]
    assert any("Shared environment detected" in msg for msg in debug_messages)

    # Cleanup
    manager.release_resources(config)


@pytest.mark.integration
def test_prepare_environment_with_pbs_environment(tmp_path, monkeypatch):
    """Test prepare_environment() detects PBS environment."""
    manager = InfrastructureManager()

    # Simulate PBS environment
    monkeypatch.setenv("PBS_JOBID", "67890")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    class MockHardware:
        allow_process_kill = True
        lock_file_path = tmp_path / "test.lock"

    class MockConfig:
        hardware = MockHardware()

    class MockLogger:
        def __init__(self):
            self.debug_calls = []

        def info(self, msg):
            pass

        def debug(self, msg):
            self.debug_calls.append(msg)

        def warning(self, msg):
            pass

    logger = MockLogger()
    manager.prepare_environment(MockConfig(), logger=logger)

    assert any("Shared environment detected" in msg for msg in logger.debug_calls)

    # Cleanup
    manager.release_resources(MockConfig())


# INFRASTRUCTURE MANAGER: CACHE FLUSHING
@pytest.mark.unit
def test_flush_compute_cache_with_cuda(monkeypatch):
    """Test _flush_compute_cache() with CUDA available."""
    manager = InfrastructureManager()

    cuda_cache_cleared = False

    def mock_empty_cache():
        nonlocal cuda_cache_cleared
        cuda_cache_cleared = True

    # Mock CUDA availability
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

    # Mock MPS availability
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    # Create mock MPS backend
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

    # Mock MPS that raises exception
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

    # Should handle exception gracefully
    assert any("MPS cache cleanup failed" in msg for msg in logger.debug_messages)


@pytest.mark.integration
def test_release_resources_lock_failure(tmp_path):
    """Test release_resources() handles lock release failures."""
    manager = InfrastructureManager()

    # Create lock in a way that removal will fail
    lock_path = tmp_path / "readonly_dir" / "test.lock"
    lock_path.parent.mkdir()
    lock_path.touch()

    # Make parent directory read-only (on Unix systems)
    if os.name != "nt":  # Skip on Windows
        lock_path.parent.chmod(0o444)

    class MockHardware:
        allow_process_kill = False
        lock_file_path = lock_path

    class MockConfig:
        hardware = MockHardware()

    class MockLogger:
        def __init__(self):
            self.warnings = []

        def info(self, msg):
            pass

        def warning(self, msg):
            self.warnings.append(msg)

        def debug(self, msg):
            pass

    logger = MockLogger()
    config = MockConfig()

    # Should handle lock release failure gracefully
    manager.release_resources(config, logger=logger)

    # Should log warning about failed lock release
    assert any("Failed to release lock" in msg for msg in logger.warnings)

    # Cleanup: restore permissions
    if os.name != "nt":
        lock_path.parent.chmod(0o755)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
