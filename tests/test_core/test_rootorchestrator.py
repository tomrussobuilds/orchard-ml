"""
Tests for RootOrchestrator (using refactored version with DI).

Tests all 7 phases, __enter__, __exit__, and edge cases.
Achieves high coverage through dependency injection and mocking.
"""

# =========================================================================== #
#                         Standard Imports                                    #
# =========================================================================== #
from unittest.mock import MagicMock

# =========================================================================== #
#                         Third-Party Imports                                 #
# =========================================================================== #
import pytest

# =========================================================================== #
#                         Internal Imports                                    #
# =========================================================================== #
from orchard.core.orchestrator import RootOrchestrator

# =========================================================================== #
#                    ORCHESTRATOR: INITIALIZATION                             #
# =========================================================================== #


@pytest.mark.unit
def test_orchestrator_init_with_defaults():
    """Test RootOrchestrator initializes with default dependencies."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.effective_num_workers = 4

    orch = RootOrchestrator(cfg=mock_cfg)

    assert orch.cfg == mock_cfg
    assert orch.repro_mode is False
    assert orch.num_workers == 4
    assert orch.paths is None
    assert orch.run_logger is None


@pytest.mark.unit
def test_orchestrator_init_extracts_policies():
    """Test RootOrchestrator extracts policies from config."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = True
    mock_cfg.hardware.effective_num_workers = 8

    orch = RootOrchestrator(cfg=mock_cfg)

    assert orch.repro_mode is True
    assert orch.num_workers == 8


# =========================================================================== #
#                    CONTEXT MANAGER: __ENTER__                               #
# =========================================================================== #


@pytest.mark.unit
def test_context_manager_enter():
    """Test __enter__ calls initialize_core_services."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = False
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
    mock_cfg.hardware.effective_num_workers = 4

    orch = RootOrchestrator(cfg=mock_cfg)
    orch.initialize_core_services = MagicMock(side_effect=RuntimeError("Init failed"))
    orch.cleanup = MagicMock()

    with pytest.raises(RuntimeError, match="Init failed"):
        orch.__enter__()

    orch.cleanup.assert_called_once()


# =========================================================================== #
#                    CONTEXT MANAGER: __EXIT__                                #
# =========================================================================== #


@pytest.mark.unit
def test_context_manager_exit_calls_cleanup():
    """Test __exit__ calls cleanup."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.effective_num_workers = 4

    orch = RootOrchestrator(cfg=mock_cfg)
    orch.cleanup = MagicMock()

    result = orch.__exit__(None, None, None)

    orch.cleanup.assert_called_once()
    assert result is False  # Exception propagation


@pytest.mark.unit
def test_context_manager_exit_propagates_exception():
    """Test __exit__ returns False to propagate exceptions."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.effective_num_workers = 4

    orch = RootOrchestrator(cfg=mock_cfg)
    orch.cleanup = MagicMock()

    # Simulate exception in with block
    result = orch.__exit__(ValueError, ValueError("test"), None)

    assert result is False  # Allows exception to propagate


# =========================================================================== #
#                    GET DEVICE                                               #
# =========================================================================== #


@pytest.mark.unit
def test_get_device_returns_cpu():
    """Test get_device returns CPU device."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.device = "cpu"
    mock_cfg.hardware.use_deterministic_algorithms = False
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
    mock_cfg.hardware.effective_num_workers = 4

    orch = RootOrchestrator(cfg=mock_cfg)

    # First call
    device1 = orch.get_device()
    # Second call
    device2 = orch.get_device()

    # Should be the same object (cached)
    assert device1 is device2


# =========================================================================== #
#                    CLEANUP                                                  #
# =========================================================================== #


@pytest.mark.unit
def test_cleanup_no_infra_manager():
    """Test cleanup handles missing infra manager gracefully."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.effective_num_workers = 4

    orch = RootOrchestrator(cfg=mock_cfg)
    orch.infra = None

    # Should not crash
    orch.cleanup()


@pytest.mark.unit
def test_cleanup_handles_exception_closes_handlers():
    """Test cleanup closes log handlers on exception."""
    mock_cfg = MagicMock()
    mock_cfg.hardware.use_deterministic_algorithms = False
    mock_cfg.hardware.effective_num_workers = 4
    mock_infra = MagicMock()
    mock_infra.release_resources.side_effect = RuntimeError("Release failed")
    mock_logger = MagicMock()
    mock_handler = MagicMock()
    mock_logger.handlers = [mock_handler]

    orch = RootOrchestrator(cfg=mock_cfg)
    orch.infra = mock_infra
    orch.run_logger = mock_logger

    # Should not raise
    orch.cleanup()

    # Should close handlers
    mock_handler.close.assert_called_once()
    mock_logger.removeHandler.assert_called_once_with(mock_handler)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
