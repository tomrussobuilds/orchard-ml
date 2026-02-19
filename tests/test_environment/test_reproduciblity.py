"""
Test Suite for Reproducibility Utilities.

Covers deterministic seeding and DataLoader worker initialization logic.
"""

import os
import random
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from orchard.core import set_seed, worker_init_fn


# TESTS: set_seed
@pytest.mark.unit
def test_set_seed_reproducibility_cpu():
    """set_seed enforces deterministic CPU behavior."""

    set_seed(123)

    rng = np.random.default_rng(123)
    a1 = random.random()
    b1 = rng.random()
    c1 = torch.rand(1)

    set_seed(123)

    rng = np.random.default_rng(123)
    a2 = random.random()
    b2 = rng.random()
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
def test_set_seed_strict_mode_with_cuda_available():
    """Strict mode enables deterministic PyTorch behavior when CUDA is available."""
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.use_deterministic_algorithms") as mock_deterministic,
    ):
        set_seed(42, strict=True)
        mock_deterministic.assert_called_once_with(True)

        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"


@pytest.mark.unit
def test_set_seed_non_strict_mode_with_cuda_available():
    """Non-strict mode sets cudnn flags but not deterministic algorithms."""
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.use_deterministic_algorithms") as mock_deterministic,
    ):
        set_seed(42, strict=False)
        mock_deterministic.assert_not_called()

        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False


@pytest.mark.unit
def test_set_seed_strict_mode_without_cuda():
    """Strict mode enables deterministic algorithms even without CUDA."""
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.use_deterministic_algorithms") as mock_deterministic,
    ):
        set_seed(42, strict=True)
        mock_deterministic.assert_called_once_with(True)


@pytest.mark.unit
def test_set_seed_non_strict_without_cuda():
    """Non-strict mode without CUDA skips all backend-specific config."""
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.use_deterministic_algorithms") as mock_deterministic,
    ):
        set_seed(42, strict=False)
        mock_deterministic.assert_not_called()


@pytest.mark.unit
def test_set_seed_cuda_branches_coverage():
    """Ensures all CUDA-related branches are executed in tests."""
    with patch("torch.cuda.is_available", return_value=True):
        mock_cudnn = MagicMock()
        with patch("torch.backends.cudnn", mock_cudnn):
            set_seed(42, strict=True)
            assert mock_cudnn.deterministic is True
            assert mock_cudnn.benchmark is False

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.use_deterministic_algorithms"),
    ):
        set_seed(42, strict=True)
        set_seed(42, strict=False)


# TESTS: MPS support
@pytest.mark.unit
def test_set_seed_with_mps_available():
    """MPS manual seed is called when MPS backend is available."""
    mock_mps_backend = MagicMock()
    mock_mps_backend.is_available.return_value = True
    mock_mps = MagicMock()

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps", mock_mps_backend),
        patch("torch.mps", mock_mps),
        patch("torch.use_deterministic_algorithms"),
    ):
        set_seed(42, strict=False)
        mock_mps.manual_seed.assert_called_once_with(42)


@pytest.mark.unit
def test_set_seed_strict_mode_with_mps_available():
    """Strict mode with MPS seeds MPS and enables deterministic algorithms."""
    mock_mps_backend = MagicMock()
    mock_mps_backend.is_available.return_value = True
    mock_mps = MagicMock()

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps", mock_mps_backend),
        patch("torch.mps", mock_mps),
        patch("torch.use_deterministic_algorithms") as mock_deterministic,
    ):
        set_seed(42, strict=True)
        mock_mps.manual_seed.assert_called_once_with(42)
        mock_deterministic.assert_called_once_with(True)


@pytest.mark.unit
def test_set_seed_mps_not_available():
    """MPS seed is skipped when MPS backend is not available."""
    mock_mps_backend = MagicMock()
    mock_mps_backend.is_available.return_value = False
    mock_mps = MagicMock()

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps", mock_mps_backend),
        patch("torch.mps", mock_mps),
    ):
        set_seed(42, strict=False)
        mock_mps.manual_seed.assert_not_called()


# TESTS: worker_init_fn
@pytest.mark.unit
def test_worker_init_fn_no_worker_info(monkeypatch):
    """worker_init_fn is a no-op outside DataLoader workers."""
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: None)

    worker_init_fn(worker_id=0)


@pytest.mark.unit
def test_worker_init_fn_sets_deterministic_state(monkeypatch):
    """worker_init_fn initializes RNGs deterministically for worker."""

    class DummyWorkerInfo:
        seed = 1000

    monkeypatch.setattr(
        torch.utils.data,
        "get_worker_info",
        lambda: DummyWorkerInfo(),
    )

    worker_init_fn(worker_id=1)

    rng = np.random.default_rng(1000)
    a1 = random.random()
    b1 = rng.random()
    c1 = torch.rand(1)

    worker_init_fn(worker_id=1)

    rng = np.random.default_rng(1000)
    a2 = random.random()
    b2 = rng.random()
    c2 = torch.rand(1)

    assert a1 == a2
    assert b1 == b2
    assert torch.equal(c1, c2)


# INTEGRATION TESTS
@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_set_seed_strict_mode_real_cuda():
    """Integration test: strict mode with real CUDA hardware."""
    set_seed(42, strict=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_set_seed_non_strict_mode_real_cuda():
    """Integration test: non-strict mode with real CUDA hardware."""
    set_seed(42, strict=False)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


_has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


@pytest.mark.unit
@pytest.mark.skipif(not _has_mps, reason="MPS not available")
def test_set_seed_strict_mode_real_mps():
    """Integration test: strict mode with real MPS hardware."""
    set_seed(42, strict=True)


@pytest.mark.unit
@pytest.mark.skipif(not _has_mps, reason="MPS not available")
def test_set_seed_non_strict_mode_real_mps():
    """Integration test: non-strict mode with real MPS hardware."""
    set_seed(42, strict=False)
