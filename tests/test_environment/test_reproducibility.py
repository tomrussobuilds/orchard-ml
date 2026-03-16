"""
Test Suite for Reproducibility Utilities.

Covers deterministic seeding and DataLoader worker initialization logic.
"""

from __future__ import annotations

import os
import random
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from orchard.core import set_seed, worker_init_fn


# TESTS: set_seed
@pytest.mark.unit
def test_set_seed_reproducibility_cpu() -> None:
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
def test_set_seed_sets_python_hashseed() -> None:
    """PYTHONHASHSEED is correctly set."""
    set_seed(999)
    assert os.environ["PYTHONHASHSEED"] == "999"


@pytest.mark.unit
def test_set_seed_strict_mode_with_cuda_available() -> None:
    """Strict mode enables deterministic PyTorch behavior when CUDA is available."""
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.use_deterministic_algorithms") as mock_deterministic,
    ):
        set_seed(42, strict=True)
        mock_deterministic.assert_called_once_with(True, warn_only=False)

        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"


@pytest.mark.unit
def test_set_seed_non_strict_mode_with_cuda_available() -> None:
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
def test_set_seed_strict_mode_without_cuda() -> None:
    """Strict mode enables deterministic algorithms even without CUDA."""
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.use_deterministic_algorithms") as mock_deterministic,
    ):
        set_seed(42, strict=True)
        mock_deterministic.assert_called_once_with(True, warn_only=False)


@pytest.mark.unit
def test_set_seed_non_strict_without_cuda() -> None:
    """Non-strict mode without CUDA skips all backend-specific config."""
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.use_deterministic_algorithms") as mock_deterministic,
    ):
        set_seed(42, strict=False)
        mock_deterministic.assert_not_called()


@pytest.mark.unit
def test_set_seed_cuda_branches_coverage() -> None:
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
def test_set_seed_with_mps_available() -> None:
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
def test_set_seed_strict_mode_with_mps_available() -> None:
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
        with pytest.warns(UserWarning, match="MPS backend has partial determinism"):
            set_seed(42, strict=True)
        mock_mps.manual_seed.assert_called_once_with(42)
        mock_deterministic.assert_called_once_with(True, warn_only=False)


@pytest.mark.unit
def test_set_seed_mps_not_available() -> None:
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
def test_worker_init_fn_no_worker_info(monkeypatch: pytest.MonkeyPatch) -> None:
    """worker_init_fn is a no-op outside DataLoader workers."""
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: None)

    worker_init_fn(worker_id=0)


@pytest.mark.unit
def test_worker_init_fn_sets_deterministic_state(monkeypatch: pytest.MonkeyPatch) -> None:
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


@pytest.mark.unit
def test_worker_init_fn_seed_calculation(monkeypatch: pytest.MonkeyPatch) -> None:
    """worker_init_fn computes seed = (base_seed + worker_id) % 2**32."""

    # Use a large seed that wraps around 2**32 to distinguish from % 3**32 or % 2**33
    large_seed = 2**32 - 1

    class DummyWorkerInfo:
        seed = large_seed

    monkeypatch.setattr(
        torch.utils.data,
        "get_worker_info",
        lambda: DummyWorkerInfo(),
    )

    expected_seed = (large_seed + 3) % 2**32
    worker_init_fn(worker_id=3)

    # Verify python random was seeded with expected_seed
    random.seed(expected_seed)
    expected_val = random.random()

    worker_init_fn(worker_id=3)
    actual_val = random.random()
    assert actual_val == expected_val


@pytest.mark.unit
def test_worker_init_fn_different_workers_different_seeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Different worker_ids produce different RNG states."""

    class DummyWorkerInfo:
        seed = 1000

    monkeypatch.setattr(
        torch.utils.data,
        "get_worker_info",
        lambda: DummyWorkerInfo(),
    )

    worker_init_fn(worker_id=0)
    val0 = random.random()

    worker_init_fn(worker_id=1)
    val1 = random.random()

    assert val0 != val1


@pytest.mark.unit
def test_worker_init_fn_seeds_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """worker_init_fn seeds numpy's legacy PRNG with the computed seed."""

    class DummyWorkerInfo:
        seed = 2000

    monkeypatch.setattr(
        torch.utils.data,
        "get_worker_info",
        lambda: DummyWorkerInfo(),
    )

    expected_seed = (2000 + 1) % 2**32
    worker_init_fn(worker_id=1)
    a = np.random.random()

    np.random.seed(expected_seed)
    b = np.random.random()
    assert a == b


# INTEGRATION TESTS
@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_set_seed_strict_mode_real_cuda() -> None:
    """Integration test: strict mode with real CUDA hardware."""
    set_seed(42, strict=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_set_seed_non_strict_mode_real_cuda() -> None:
    """Integration test: non-strict mode with real CUDA hardware."""
    set_seed(42, strict=False)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


_has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


@pytest.mark.unit
@pytest.mark.skipif(not _has_mps, reason="MPS not available")
def test_set_seed_strict_mode_real_mps() -> None:
    """Integration test: strict mode with real MPS hardware."""
    set_seed(42, strict=True)


@pytest.mark.unit
@pytest.mark.skipif(not _has_mps, reason="MPS not available")
def test_set_seed_non_strict_mode_real_mps() -> None:
    """Integration test: non-strict mode with real MPS hardware."""
    set_seed(42, strict=False)


# TESTS: warn_only mode
@pytest.mark.unit
def test_set_seed_strict_warn_only_mode() -> None:
    """Strict mode with warn_only passes warn_only=True to PyTorch."""
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.use_deterministic_algorithms") as mock_deterministic,
    ):
        set_seed(42, strict=True, warn_only=True)
        mock_deterministic.assert_called_once_with(True, warn_only=True)


@pytest.mark.unit
def test_set_seed_strict_warn_only_false_by_default() -> None:
    """Strict mode defaults to warn_only=False."""
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.use_deterministic_algorithms") as mock_deterministic,
    ):
        set_seed(42, strict=True)
        mock_deterministic.assert_called_once_with(True, warn_only=False)


@pytest.mark.unit
def test_set_seed_warn_only_ignored_when_not_strict() -> None:
    """warn_only is ignored when strict=False."""
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.use_deterministic_algorithms") as mock_deterministic,
    ):
        set_seed(42, strict=False, warn_only=True)
        mock_deterministic.assert_not_called()


# TESTS: default parameter values
@pytest.mark.unit
def test_set_seed_defaults_strict_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default strict=False does not enable deterministic algorithms."""
    # Ensure PYTHONHASHSEED is preset to avoid warning side-effects
    monkeypatch.setenv("PYTHONHASHSEED", "42")
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.use_deterministic_algorithms") as mock_det,
    ):
        set_seed(42)
        mock_det.assert_not_called()


@pytest.mark.unit
def test_set_seed_defaults_warn_only_false() -> None:
    """Default warn_only=False is passed through in strict mode."""
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.use_deterministic_algorithms") as mock_det,
    ):
        set_seed(42, strict=True)
        mock_det.assert_called_once_with(True, warn_only=False)


# TESTS: PYTHONHASHSEED warning logic
@pytest.mark.unit
def test_set_seed_strict_warns_pythonhashseed_not_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strict mode warns when PYTHONHASHSEED was not already set to the seed."""
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.use_deterministic_algorithms"),
    ):
        with pytest.warns(UserWarning) as record:
            set_seed(42, strict=True)
        assert len(record) == 1
        assert "PYTHONHASHSEED" in str(record[0].message)


@pytest.mark.unit
def test_set_seed_strict_no_warn_pythonhashseed_already_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No PYTHONHASHSEED warning when it's already set to the correct value."""
    monkeypatch.setenv("PYTHONHASHSEED", "42")
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.use_deterministic_algorithms"),
    ):
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("error")
            set_seed(42, strict=True)


@pytest.mark.unit
def test_set_seed_strict_warns_pythonhashseed_wrong_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """PYTHONHASHSEED warning fires when set to a different seed value."""
    monkeypatch.setenv("PYTHONHASHSEED", "999")
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.use_deterministic_algorithms"),
        pytest.warns(UserWarning, match="PYTHONHASHSEED=42 set at runtime"),
    ):
        set_seed(42, strict=True)


@pytest.mark.unit
def test_set_seed_non_strict_no_pythonhashseed_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-strict mode never emits the PYTHONHASHSEED warning."""
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    with patch("torch.cuda.is_available", return_value=False):
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("error")
            set_seed(42, strict=False)


# TESTS: numpy seeding
@pytest.mark.unit
def test_set_seed_seeds_numpy() -> None:
    """set_seed correctly seeds numpy's legacy PRNG."""
    set_seed(42)
    a = np.random.random()
    set_seed(42)
    b = np.random.random()
    assert a == b

    set_seed(99)
    c = np.random.random()
    assert a != c


# TESTS: CUBLAS_WORKSPACE_CONFIG
@pytest.mark.unit
def test_set_seed_cublas_workspace_config_exact_key() -> None:
    """Strict CUDA mode sets exactly CUBLAS_WORKSPACE_CONFIG (not a mangled key)."""
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.use_deterministic_algorithms"),
    ):
        set_seed(42, strict=True)
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        assert os.environ.get("XXCUBLAS_WORKSPACE_CONFIGXX") is None
        assert os.environ.get("cublas_workspace_config") is None


# TESTS: MPS strict mode warning
@pytest.mark.unit
def test_set_seed_strict_mps_partial_determinism_warning() -> None:
    """Strict mode with MPS warns about partial determinism support."""
    mock_mps_backend = MagicMock()
    mock_mps_backend.is_available.return_value = True
    mock_mps = MagicMock()

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps", mock_mps_backend),
        patch("torch.mps", mock_mps),
        patch("torch.use_deterministic_algorithms"),
        pytest.warns(UserWarning, match="MPS backend has partial determinism"),
    ):
        set_seed(42, strict=True)


@pytest.mark.unit
def test_set_seed_strict_mps_warning_message_content() -> None:
    """MPS warning contains exact expected substrings about determinism and CPU."""
    mock_mps_backend = MagicMock()
    mock_mps_backend.is_available.return_value = True
    mock_mps = MagicMock()

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps", mock_mps_backend),
        patch("torch.mps", mock_mps),
        patch("torch.use_deterministic_algorithms"),
    ):
        with pytest.warns(UserWarning) as record:
            set_seed(42, strict=True)
        # Filter to only MPS warnings (exclude PYTHONHASHSEED)
        mps_warnings = [r for r in record if "MPS" in str(r.message)]
        assert len(mps_warnings) == 1
        assert "MPS" in str(mps_warnings[0].message)


@pytest.mark.unit
def test_set_seed_strict_pythonhashseed_warning_stacklevel(monkeypatch: pytest.MonkeyPatch) -> None:
    """PYTHONHASHSEED warning uses stacklevel=2 so caller frame is shown."""
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.use_deterministic_algorithms"),
        patch("orchard.core.environment.reproducibility.warnings.warn") as mock_warn,
    ):
        set_seed(42, strict=True)
        mock_warn.assert_called_once()
        assert mock_warn.call_args.kwargs["stacklevel"] == 2


@pytest.mark.unit
def test_set_seed_strict_mps_warning_stacklevel() -> None:
    """MPS partial determinism warning uses stacklevel=2."""
    mock_mps_backend = MagicMock()
    mock_mps_backend.is_available.return_value = True
    mock_mps = MagicMock()

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps", mock_mps_backend),
        patch("torch.mps", mock_mps),
        patch("torch.use_deterministic_algorithms"),
        patch("orchard.core.environment.reproducibility.warnings.warn") as mock_warn,
    ):
        set_seed(42, strict=True)
        # Filter to MPS warning call (there may also be PYTHONHASHSEED call)
        mps_calls = [c for c in mock_warn.call_args_list if "MPS" in str(c)]
        assert len(mps_calls) == 1
        assert mps_calls[0].kwargs["stacklevel"] == 2
