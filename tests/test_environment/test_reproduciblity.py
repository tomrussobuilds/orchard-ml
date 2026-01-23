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


#                         TESTS: Mode Detection                               #
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


#                         TESTS: set_seed                                     #
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


#                         TESTS: worker_init_fn                               #
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
