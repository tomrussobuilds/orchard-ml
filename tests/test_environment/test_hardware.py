"""
Test Suite for Hardware Acceleration & Computing Environment.

Tests hardware detection, device configuration, CUDA utilities,
and CPU thread management.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import matplotlib
import pytest
import torch

from orchard.core.environment import (
    apply_cpu_threads,
    configure_system_libraries,
    detect_best_device,
    get_accelerator_name,
    get_num_workers,
    get_vram_info,
    to_device_obj,
)
from tests.conftest import mutmut_safe_env


# SYSTEM CONFIGURATION
@pytest.mark.unit
@patch("platform.system", return_value="Linux")
def test_configure_system_libraries_linux(mock_platform: MagicMock) -> None:
    """Test configure_system_libraries sets Agg backend on Linux."""
    with patch.dict(os.environ, mutmut_safe_env(), clear=True):
        configure_system_libraries()

        assert matplotlib.get_backend() == "Agg"
        assert matplotlib.rcParams["pdf.fonttype"] == 42
        assert matplotlib.rcParams["ps.fonttype"] == 42


@pytest.mark.unit
@patch("platform.system", return_value="Linux")
@patch("orchard.core.environment.hardware.Path")
def test_configure_system_libraries_docker(mock_path: MagicMock, mock_platform: MagicMock) -> None:
    """Test configure_system_libraries detects Docker environment."""
    mock_path.return_value.exists.return_value = True
    with patch.dict(os.environ, mutmut_safe_env(), clear=True):
        configure_system_libraries()

        assert matplotlib.get_backend() == "Agg"


@pytest.mark.unit
@patch("platform.system", return_value="Linux")
def test_configure_system_libraries_docker_env_var(mock_platform: MagicMock) -> None:
    """Test configure_system_libraries uses IN_DOCKER environment variable."""
    with patch.dict(os.environ, {"IN_DOCKER": "TRUE"}):
        configure_system_libraries()

        assert matplotlib.get_backend() == "Agg"


@pytest.mark.unit
@patch("platform.system", return_value="Windows")
def test_configure_system_libraries_windows(mock_platform: MagicMock) -> None:
    """Test configure_system_libraries skips Agg backend on Windows."""
    with patch.dict(os.environ, mutmut_safe_env(), clear=True):
        original_backend = matplotlib.get_backend()

        configure_system_libraries()

        assert matplotlib.get_backend() == original_backend


@pytest.mark.unit
@patch("orchard.core.environment.hardware.Path")
@patch("platform.system", return_value="Windows")
def test_configure_system_libraries_non_linux_non_docker_skips(
    mock_platform: MagicMock, mock_path: MagicMock
) -> None:
    """Test non-Linux, non-Docker environment skips all configuration."""
    mock_path.return_value.exists.return_value = False
    with patch.dict(os.environ, mutmut_safe_env(), clear=True):
        matplotlib.rcParams["pdf.fonttype"] = 3
        matplotlib.rcParams["ps.fonttype"] = 3

        configure_system_libraries()

        assert matplotlib.rcParams["pdf.fonttype"] == 3
        assert matplotlib.rcParams["ps.fonttype"] == 3


@pytest.mark.unit
@patch("orchard.core.environment.hardware.Path")
@patch("platform.system", return_value="Linux")
def test_configure_system_libraries_linux_only_no_docker(
    mock_platform: MagicMock, mock_path: MagicMock
) -> None:
    """Test is_linux alone triggers configuration (kills or→and mutant)."""
    mock_path.return_value.exists.return_value = False
    with patch.dict(os.environ, mutmut_safe_env(), clear=True):
        matplotlib.rcParams["pdf.fonttype"] = 3
        matplotlib.rcParams["ps.fonttype"] = 3

        configure_system_libraries()

        assert matplotlib.get_backend() == "Agg"
        assert matplotlib.rcParams["pdf.fonttype"] == 42
        assert matplotlib.rcParams["ps.fonttype"] == 42


@pytest.mark.unit
@patch("orchard.core.environment.hardware.Path")
@patch("platform.system", return_value="Darwin")
def test_configure_system_libraries_docker_env_only_no_linux(
    mock_platform: MagicMock, mock_path: MagicMock
) -> None:
    """Test is_docker via IN_DOCKER env var alone triggers config (kills is_linux mutants)."""
    mock_path.return_value.exists.return_value = False
    with patch.dict(os.environ, mutmut_safe_env(IN_DOCKER="TRUE"), clear=True):
        matplotlib.rcParams["pdf.fonttype"] = 3
        matplotlib.rcParams["ps.fonttype"] = 3

        configure_system_libraries()

        assert matplotlib.get_backend() == "Agg"
        assert matplotlib.rcParams["pdf.fonttype"] == 42
        assert matplotlib.rcParams["ps.fonttype"] == 42


@pytest.mark.unit
@patch("platform.system", return_value="Darwin")
def test_configure_system_libraries_dockerenv_file_only(mock_platform: MagicMock) -> None:
    """Test is_docker via /.dockerenv file alone triggers config (exact path)."""
    with patch.dict(os.environ, mutmut_safe_env(), clear=True):
        matplotlib.rcParams["pdf.fonttype"] = 3

        with patch("orchard.core.environment.hardware.Path") as MockPath:
            MockPath.return_value.exists.return_value = True
            configure_system_libraries()

            # Verify exact dockerenv path (kills path-string mutants)
            MockPath.assert_called_once_with("/.dockerenv")
            assert matplotlib.get_backend() == "Agg"
            assert matplotlib.rcParams["pdf.fonttype"] == 42


@pytest.mark.unit
@patch("orchard.core.environment.hardware.Path")
@patch("platform.system", return_value="Darwin")
def test_configure_system_libraries_docker_env_wrong_value(
    mock_platform: MagicMock, mock_path: MagicMock
) -> None:
    """Test IN_DOCKER with wrong value still works via dockerenv file."""
    mock_path.return_value.exists.return_value = True
    with patch.dict(os.environ, mutmut_safe_env(IN_DOCKER="false"), clear=True):
        matplotlib.rcParams["pdf.fonttype"] = 3

        configure_system_libraries()

        assert matplotlib.get_backend() == "Agg"
        assert matplotlib.rcParams["pdf.fonttype"] == 42


@pytest.mark.unit
@patch("orchard.core.environment.hardware.Path")
@patch("platform.system", return_value="Darwin")
def test_configure_system_libraries_no_docker_no_linux_skips(
    mock_platform: MagicMock, mock_path: MagicMock
) -> None:
    """Test non-Linux non-Docker truly skips (kills or→and and string mutants)."""
    mock_path.return_value.exists.return_value = False
    with patch.dict(os.environ, mutmut_safe_env(), clear=True):
        matplotlib.rcParams["pdf.fonttype"] = 3
        matplotlib.rcParams["ps.fonttype"] = 3

        configure_system_libraries()

        assert matplotlib.rcParams["pdf.fonttype"] == 3


# DEVICE DETECTION
@pytest.mark.unit
@patch("torch.cuda.is_available", return_value=True)
def test_detect_best_device_cuda(mock_cuda: MagicMock) -> None:
    """Test detect_best_device prioritizes CUDA when available."""
    device = detect_best_device()
    assert device == "cuda"


@pytest.mark.unit
@patch("torch.cuda.is_available", return_value=False)
def test_detect_best_device_mps(mock_cuda: MagicMock) -> None:
    """Test detect_best_device falls back to MPS when CUDA unavailable."""
    if hasattr(torch.backends, "mps"):
        with patch.object(torch.backends.mps, "is_available", return_value=True):
            device = detect_best_device()
            assert device == "mps"


@pytest.mark.unit
@patch("torch.cuda.is_available", return_value=False)
def test_detect_best_device_cpu(mock_cuda: MagicMock) -> None:
    """Test detect_best_device falls back to CPU when no accelerators."""
    if hasattr(torch.backends, "mps"):
        with patch.object(torch.backends.mps, "is_available", return_value=False):
            device = detect_best_device()
            assert device == "cpu"
    else:
        device = detect_best_device()
        assert device == "cpu"


# DEVICE OBJECT CONVERSION
@pytest.mark.unit
def test_to_device_obj_cpu() -> None:
    """Test to_device_obj converts 'cpu' string to torch.device."""
    device = to_device_obj("cpu")
    assert isinstance(device, torch.device)
    assert device.type == "cpu"


@pytest.mark.unit
@patch("torch.cuda.set_device")
@patch("torch.cuda.is_available", return_value=True)
def test_to_device_obj_cuda(mock_cuda: MagicMock, mock_set_device: MagicMock) -> None:
    """Test to_device_obj converts 'cuda' to default device (no index)."""
    device = to_device_obj("cuda")
    assert isinstance(device, torch.device)
    assert device == torch.device("cuda")
    mock_set_device.assert_not_called()


@pytest.mark.unit
@patch("torch.cuda.is_available", return_value=False)
def test_to_device_obj_cuda_unavailable(mock_cuda: MagicMock) -> None:
    """Test to_device_obj raises ValueError when CUDA requested but unavailable."""
    with pytest.raises(ValueError, match="CUDA requested but not available"):
        to_device_obj("cuda")


@pytest.mark.unit
def test_to_device_obj_mps() -> None:
    """Test to_device_obj converts 'mps' string to torch.device."""
    device = to_device_obj("mps")
    assert isinstance(device, torch.device)
    assert device.type == "mps"


@pytest.mark.unit
@patch("torch.cuda.is_available", return_value=True)
def test_to_device_obj_auto_cuda(mock_cuda: MagicMock) -> None:
    """Test to_device_obj auto-selects CUDA when available."""
    device = to_device_obj("auto")
    assert isinstance(device, torch.device)
    assert device.type == "cuda"


@pytest.mark.unit
@patch("torch.cuda.is_available", return_value=False)
def test_to_device_obj_auto_cpu(mock_cuda: MagicMock) -> None:
    """Test to_device_obj auto-selects CPU when no accelerators."""
    if hasattr(torch.backends, "mps"):
        with patch.object(torch.backends.mps, "is_available", return_value=False):
            device = to_device_obj("auto")
            assert device.type == "cpu"
    else:
        device = to_device_obj("auto")
        assert device.type == "cpu"


@pytest.mark.unit
def test_to_device_obj_invalid_device() -> None:
    """Test to_device_obj raises ValueError for unsupported device."""
    with pytest.raises(ValueError, match="Unsupported device"):
        to_device_obj("invalid_device")


@pytest.mark.unit
def test_to_device_obj_case_sensitivity() -> None:
    """Test to_device_obj is case-sensitive."""
    device = to_device_obj("cpu")
    assert device.type == "cpu"

    with pytest.raises(ValueError, match="Unsupported device"):
        to_device_obj("CPU")


# ACCELERATOR NAME
@pytest.mark.unit
@patch("torch.cuda.is_available", return_value=False)
def test_get_accelerator_name_cpu(mock_cuda: MagicMock) -> None:
    """Test get_accelerator_name returns empty string when no accelerator."""
    if hasattr(torch.backends, "mps"):
        with patch.object(torch.backends.mps, "is_available", return_value=False):
            name = get_accelerator_name()
            assert name == ""
    else:
        name = get_accelerator_name()
        assert name == ""


@pytest.mark.unit
@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 3090")
def test_get_accelerator_name_cuda(mock_name: MagicMock, mock_cuda: MagicMock) -> None:
    """Test get_accelerator_name returns GPU model name when CUDA available."""
    name = get_accelerator_name()
    assert name == "NVIDIA GeForce RTX 3090"


@pytest.mark.unit
@patch("torch.cuda.is_available", return_value=False)
def test_get_accelerator_name_mps(mock_cuda: MagicMock) -> None:
    """Test get_accelerator_name returns Apple Silicon string when MPS available."""
    if hasattr(torch.backends, "mps"):
        with patch.object(torch.backends.mps, "is_available", return_value=True):
            name = get_accelerator_name()
            assert "Apple Silicon" in name


@pytest.mark.unit
@patch("torch.cuda.is_available", return_value=False)
def test_get_vram_info_unavailable(mock_cuda: MagicMock) -> None:
    """Test get_vram_info returns N/A when CUDA unavailable."""
    info = get_vram_info()
    assert info == "N/A"


@pytest.mark.unit
@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.device_count", return_value=1)
@patch("torch.cuda.mem_get_info", return_value=(8 * 1024**3, 16 * 1024**3))
def test_get_vram_info_available(
    mock_mem_info: MagicMock, mock_device_count: MagicMock, mock_cuda: MagicMock
) -> None:
    """Test get_vram_info returns formatted VRAM string when CUDA available."""
    info = get_vram_info(device_idx=0)

    assert "GB" in info
    assert "/" in info
    assert "8.00 GB" in info
    assert "16.00 GB" in info


@pytest.mark.unit
@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.device_count", return_value=1)
def test_get_vram_info_invalid_device_index(
    mock_device_count: MagicMock, mock_cuda: MagicMock
) -> None:
    """Test get_vram_info handles invalid device index."""
    info = get_vram_info(device_idx=5)
    assert info == "Invalid Device Index"


@pytest.mark.unit
@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.device_count", return_value=1)
@patch("torch.cuda.mem_get_info", side_effect=RuntimeError("CUDA error"))
def test_get_vram_info_query_failed(
    mock_mem_info: MagicMock, mock_device_count: MagicMock, mock_cuda: MagicMock
) -> None:
    """Test get_vram_info handles CUDA query failures gracefully."""
    info = get_vram_info(device_idx=0)
    assert info == "Query Failed"


# CPU THREAD MANAGEMENT
@pytest.mark.unit
@patch("os.cpu_count", return_value=8)
def test_get_num_workers_standard(mock_cpu_count: MagicMock) -> None:
    """Test get_num_workers returns half of CPU count for standard systems."""
    num_workers = get_num_workers()
    assert num_workers == 4


@pytest.mark.unit
@patch("os.cpu_count", return_value=16)
def test_get_num_workers_capped(mock_cpu_count: MagicMock) -> None:
    """Test get_num_workers caps at 8 workers for high-core systems."""
    num_workers = get_num_workers()
    assert num_workers == 8


@pytest.mark.unit
@patch("os.cpu_count", return_value=4)
def test_get_num_workers_low_cores(mock_cpu_count: MagicMock) -> None:
    """Test get_num_workers returns 2 for low-core systems (exactly 4)."""
    num_workers = get_num_workers()
    assert num_workers == 2


@pytest.mark.unit
@patch("os.cpu_count", return_value=5)
def test_get_num_workers_boundary_five_cores(mock_cpu_count: MagicMock) -> None:
    """Test get_num_workers with exactly 5 cores (kills <=4 vs <=5 mutant)."""
    num_workers = get_num_workers()
    assert num_workers == 2


@pytest.mark.unit
@patch("os.cpu_count", return_value=2)
def test_get_num_workers_very_low_cores(mock_cpu_count: MagicMock) -> None:
    """Test get_num_workers returns 2 for very low-core systems."""
    num_workers = get_num_workers()
    assert num_workers == 2


@pytest.mark.unit
@patch("os.cpu_count", return_value=3)
def test_get_num_workers_three_cores(mock_cpu_count: MagicMock) -> None:
    """Test get_num_workers returns 2 for 3-core systems (kills <4 mutant)."""
    num_workers = get_num_workers()
    assert num_workers == 2


@pytest.mark.unit
@patch("os.cpu_count", return_value=None)
def test_get_num_workers_fallback(mock_cpu_count: MagicMock) -> None:
    """Test get_num_workers falls back to 2 when cpu_count is None."""
    num_workers = get_num_workers()
    assert num_workers == 2


@pytest.mark.unit
@patch("os.cpu_count", return_value=8)
def test_apply_cpu_threads_standard(mock_cpu_count: MagicMock) -> None:
    """Test apply_cpu_threads sets optimal thread count."""
    num_workers = 4
    threads = apply_cpu_threads(num_workers)

    assert threads == 4
    assert torch.get_num_threads() == threads
    assert os.environ["OMP_NUM_THREADS"] == str(threads)
    assert os.environ["MKL_NUM_THREADS"] == str(threads)


@pytest.mark.unit
@patch("os.cpu_count", return_value=4)
def test_apply_cpu_threads_minimum(mock_cpu_count: MagicMock) -> None:
    """Test apply_cpu_threads clamps to exactly 2 threads (kills max(2→3) mutant)."""
    num_workers = 8
    threads = apply_cpu_threads(num_workers)

    assert threads == 2
    assert torch.get_num_threads() == 2


@pytest.mark.unit
@patch("os.cpu_count", return_value=None)
def test_apply_cpu_threads_fallback(mock_cpu_count: MagicMock) -> None:
    """Test apply_cpu_threads handles None cpu_count gracefully."""
    num_workers = 4
    threads = apply_cpu_threads(num_workers)

    assert threads == 2


@pytest.mark.unit
@patch("os.cpu_count", return_value=16)
def test_apply_cpu_threads_high_core_system(mock_cpu_count: MagicMock) -> None:
    """Test apply_cpu_threads on high-core system."""
    num_workers = 4
    threads = apply_cpu_threads(num_workers)

    assert threads == 12
    assert torch.get_num_threads() == 12


# INTEGRATION TESTS
@pytest.mark.integration
@patch("os.cpu_count", return_value=8)
def test_full_hardware_workflow(mock_cpu_count: MagicMock) -> None:
    """Test complete hardware configuration workflow."""
    configure_system_libraries()

    device_str = detect_best_device()
    assert device_str in ["cuda", "mps", "cpu"]

    if device_str != "cuda" or torch.cuda.is_available():
        device = to_device_obj(device_str)
        assert isinstance(device, torch.device)

    num_workers = get_num_workers()
    assert 2 <= num_workers <= 8

    threads = apply_cpu_threads(num_workers)
    assert threads >= 2


# DEVICE OBJECT: LOCAL_RANK SUPPORT
@pytest.mark.unit
@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.set_device")
def test_to_device_obj_cuda_with_local_rank(
    mock_set_device: MagicMock, mock_cuda: MagicMock
) -> None:
    """to_device_obj assigns correct GPU for local_rank > 0."""
    device = to_device_obj("cuda", local_rank=1)

    mock_set_device.assert_called_once_with(1)
    assert device == torch.device("cuda:1")


@pytest.mark.unit
@patch("torch.cuda.is_available", return_value=True)
def test_to_device_obj_cuda_local_rank_zero(mock_cuda: MagicMock) -> None:
    """to_device_obj returns standard cuda device for local_rank=0."""
    device = to_device_obj("cuda", local_rank=0)

    assert device == torch.device("cuda")


@pytest.mark.unit
def test_to_device_obj_cpu_ignores_local_rank() -> None:
    """to_device_obj ignores local_rank for CPU device."""
    device = to_device_obj("cpu", local_rank=3)

    assert device == torch.device("cpu")


@pytest.mark.unit
def test_to_device_obj_default_local_rank_is_zero() -> None:
    """to_device_obj defaults local_rank to 0 (kills default=1 mutant)."""
    import inspect

    sig = inspect.signature(to_device_obj)
    assert sig.parameters["local_rank"].default == 0


@pytest.mark.unit
@patch("torch.cuda.is_available", return_value=False)
def test_to_device_obj_cuda_error_message_exact(mock_cuda: MagicMock) -> None:
    """to_device_obj error message is exact (kills string mutant)."""
    with pytest.raises(ValueError, match="^CUDA requested but not available$"):
        to_device_obj("cuda")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
