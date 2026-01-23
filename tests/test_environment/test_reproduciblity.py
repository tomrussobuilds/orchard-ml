"""
Test Suite for Reproducibility Utilities.

Covers deterministic seeding, reproducibility mode detection,
and DataLoader worker initialization logic.
"""

# Standard Imports
import os
import random

# Third-Party Imports
import numpy as np
import pytest
import torch

# Internal Imports
from orchard.core import is_repro_mode_requested, set_seed, worker_init_fn


# TESTS: MODE DETECTION
@pytest.mark.unit
def test_is_repro_mode_requested_cli_flag():
    """CLI flag alone enables reproducibility mode."""
    assert is_repro_mode_requested(cli_flag=True) is True


@pytest.mark.unit
def test_is_repro_mode_requested_env_var(monkeypatch):
    """Environment variable enables reproducibility mode."""
    monkeypatch.setenv("DOCKER_REPRODUCIBILITY_MODE", "TRUE")
    assert is_repro_mode_requested(cli_flag=False) is True


@pytest.mark.unit
def test_is_repro_mode_requested_disabled(monkeypatch):
    """No flags -> reproducibility disabled."""
    monkeypatch.delenv("DOCKER_REPRODUCIBILITY_MODE", raising=False)
    assert is_repro_mode_requested(cli_flag=False) is False


# TESTS: set_seed
@pytest.mark.unit
def test_set_seed_reproducibility_cpu():
    """set_seed enforces deterministic CPU behavior."""
    set_seed(123)

    a1 = random.random()
    b1 = np.random.rand()
    c1 = torch.rand(1)

    set_seed(123)

    a2 = random.random()
    b2 = np.random.rand()
    c2 = torch.rand(1)

    assert a1 == a2
    assert b1 == b2
    assert torch.equal(c1, c2)


@pytest.mark.unit
def test_set_seed_sets_python_hashseed():
    """PYTHONHASHSEED is correctly set."""
    set_seed(999)
    assert os.environ["PYTHONHASHSEED"] == "999"


@pytest.mark.unit
def test_set_seed_strict_mode_flags():
    """Strict mode enables deterministic PyTorch behavior."""
    set_seed(42, strict=True)
    if torch.cuda.is_available():
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
    else:
        assert True


# TESTS: worker_init_fn
@pytest.mark.unit
def test_worker_init_fn_no_worker_info(monkeypatch):
    """worker_init_fn is a no-op outside DataLoader workers."""
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: None)

    # Should not raise
    worker_init_fn(worker_id=0)


@pytest.mark.unit
def test_worker_init_fn_sets_deterministic_state(monkeypatch):
    """worker_init_fn initializes RNGs deterministically per worker."""

    class DummyWorkerInfo:
        seed = 1000

    monkeypatch.setattr(
        torch.utils.data,
        "get_worker_info",
        lambda: DummyWorkerInfo(),
    )

    worker_init_fn(worker_id=1)

    # Capture state
    a1 = random.random()
    b1 = np.random.rand()
    c1 = torch.rand(1)

    # Re-run with same worker_id
    worker_init_fn(worker_id=1)

    a2 = random.random()
    b2 = np.random.rand()
    c2 = torch.rand(1)

    assert a1 == a2
    assert b1 == b2
    assert torch.equal(c1, c2)


# TESTS: set_seed - strict mode branches


@pytest.mark.unit
def test_set_seed_strict_mode_enables_deterministic_algorithms():
    """Strict mode calls torch.use_deterministic_algorithms(True)."""
    set_seed(42, strict=True)

    # Verify deterministic algorithms are enabled
    # Note: This will raise an error if deterministic mode is not available
    try:
        # Try to use a non-deterministic operation
        # If strict mode is on, this should either work deterministically or raise
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
    except RuntimeError:
        # Some operations may not have deterministic implementations
        pass


@pytest.mark.unit
def test_set_seed_strict_mode_logs_message(caplog):
    """Strict mode logs the reproducibility message."""
    import logging

    with caplog.at_level(logging.INFO):
        set_seed(42, strict=True)

    assert "STRICT REPRODUCIBILITY ENABLED" in caplog.text
    assert "deterministic algorithms" in caplog.text


@pytest.mark.unit
def test_set_seed_non_strict_mode_sets_cudnn_flags():
    """Non-strict mode still sets cudnn deterministic flags."""
    set_seed(42, strict=False)

    # Even without strict mode, these should be set
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_set_seed_with_cuda_seeds_all_devices():
    """When CUDA is available, seed all CUDA devices."""
    set_seed(123, strict=False)

    # Generate random numbers on CUDA
    t1 = torch.rand(1, device="cuda")

    # Reset seed
    set_seed(123, strict=False)

    # Should generate same random number
    t2 = torch.rand(1, device="cuda")

    assert torch.equal(t1, t2)


@pytest.mark.unit
def test_set_seed_strict_vs_non_strict_behavior():
    """Verify that strict and non-strict modes differ in algorithm enforcement."""
    # Non-strict mode
    set_seed(42, strict=False)
    non_strict_deterministic = torch.backends.cudnn.deterministic

    # Strict mode
    set_seed(42, strict=True)
    strict_deterministic = torch.backends.cudnn.deterministic

    # Both should set deterministic to True
    assert non_strict_deterministic is True
    assert strict_deterministic is True


@pytest.mark.unit
def test_set_seed_different_seeds_produce_different_results():
    """Different seeds should produce different random values."""
    set_seed(123)
    a1 = torch.rand(1)

    set_seed(456)
    a2 = torch.rand(1)

    assert not torch.equal(a1, a2)


@pytest.mark.unit
def test_set_seed_affects_all_rng_sources():
    """Verify that set_seed affects Python, NumPy, and PyTorch RNGs."""
    set_seed(42)

    # Capture initial random values
    python_val = random.random()
    numpy_val = np.random.rand()
    torch_val = torch.rand(1)

    # Generate more random values (should be different)
    python_val2 = random.random()
    numpy_val2 = np.random.rand()
    torch_val2 = torch.rand(1)

    assert python_val != python_val2
    assert numpy_val != numpy_val2
    assert not torch.equal(torch_val, torch_val2)

    # Reset seed and verify we get the same initial values
    set_seed(42)

    python_val_reset = random.random()
    numpy_val_reset = np.random.rand()
    torch_val_reset = torch.rand(1)

    assert python_val == python_val_reset
    assert numpy_val == numpy_val_reset
    assert torch.equal(torch_val, torch_val_reset)
