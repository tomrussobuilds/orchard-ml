"""
Test Suite for Hardware Acceleration & Computing Environment.

Tests hardware detection, device configuration, CUDA utilities,
and CPU thread management.
"""

# Standard Imports
import os
from unittest.mock import patch

# Third-Party Imports
import matplotlib
import pytest
import torch

# Internal Imports
from orchard.core.environment import (
    apply_cpu_threads,
    configure_system_libraries,
    detect_best_device,
    get_cuda_name,
    get_num_workers,
    get_vram_info,
    to_device_obj,
)


# SYSTEM CONFIGURATION
@pytest.mark.unit
@patch("platform.system", return_value="Linux")
def test_configure_system_libraries_linux(mock_platform):
    """Test configure_system_libraries sets Agg backend on Linux."""
    with patch.dict(os.environ, {}, clear=True):
        configure_system_libraries()

        assert matplotlib.get_backend() == "Agg"
        assert matplotlib.rcParams["pdf.fonttype"] == 42
        assert matplotlib.rcParams["ps.fonttype"] == 42


@pytest.mark.unit
@patch("platform.system", return_value="Linux")
@patch("os.path.exists", return_value=True)
def test_configure_system_libraries_docker(mock_exists, mock_platform):
    """Test configure_system_libraries detects Docker environment."""
    with patch.dict(os.environ, {}, clear=True):
        configure_system_libraries()

        assert matplotlib.get_backend() == "Agg"


@pytest.mark.unit
@patch("platform.system", return_value="Linux")
def test_configure_system_libraries_docker_env_var(mock_platform):
    """Test configure_system_libraries uses IN_DOCKER environment variable."""
    with patch.dict(os.environ, {"IN_DOCKER": "TRUE"}):
        configure_system_libraries()

        assert matplotlib.get_backend() == "Agg"


@pytest.mark.unit
@patch("platform.system", return_value="Windows")
def test_configure_system_libraries_windows(mock_platform):
    """Test configure_system_libraries skips Agg backend on Windows."""
    with patch.dict(os.environ, {}, clear=True):
        # Store original backend
        original_backend = matplotlib.get_backend()

        configure_system_libraries()

        # Backend should remain unchanged on Windows
        # (unless Docker env is set)
        assert matplotlib.get_backend() == original_backend


# DEVICE DETECTION
@pytest.mark.unit
@patch("torch.cuda.is_available", return_value=True)
def test_detect_best_device_cuda(mock_cuda):
    """Test detect_best_device prioritizes CUDA when available."""
    device = detect_best_device()
    assert device == "cuda"


@pytest.mark.unit
@patch("torch.cuda.is_available", return_value=False)
def test_detect_best_device_mps(mock_cuda):
    """Test detect_best_device falls back to MPS when CUDA unavailable."""
    # Mock MPS availability
    if hasattr(torch.backends, "mps"):
        with patch.object(torch.backends.mps, "is_available", return_value=True):
            device = detect_best_device()
            assert device == "mps"


@pytest.mark.unit
@patch("torch.cuda.is_available", return_value=False)
def test_detect_best_device_cpu(mock_cuda):
    """Test detect_best_device falls back to CPU when no accelerators."""
    # Mock MPS unavailable
    if hasattr(torch.backends, "mps"):
        with patch.object(torch.backends.mps, "is_available", return_value=False):
            device = detect_best_device()
            assert device == "cpu"
    else:
        # If MPS backend doesn't exist (older PyTorch)
        device = detect_best_device()
        assert device == "cpu"


# DEVICE OBJECT CONVERSION
@pytest.mark.unit
def test_to_device_obj_cpu():
    """Test to_device_obj converts 'cpu' string to torch.device."""
    device = to_device_obj("cpu")
    assert isinstance(device, torch.device)
    assert device.type == "cpu"


@pytest.mark.unit
@patch("torch.cuda.is_available", return_value=True)
def test_to_device_obj_cuda(mock_cuda):
    """Test to_device_obj converts 'cuda' string when CUDA available."""
    device = to_device_obj("cuda")
    assert isinstance(device, torch.device)
    assert device.type == "cuda"


@pytest.mark.unit
@patch("torch.cuda.is_available", return_value=False)
def test_to_device_obj_cuda_unavailable(mock_cuda):
    """Test to_device_obj raises ValueError when CUDA requested but unavailable."""
    with pytest.raises(ValueError, match="CUDA requested but not available"):
        to_device_obj("cuda")


@pytest.mark.unit
def test_to_device_obj_mps():
    """Test to_device_obj converts 'mps' string to torch.device."""
    device = to_device_obj("mps")
    assert isinstance(device, torch.device)
    assert device.type == "mps"


@pytest.mark.unit
@patch("torch.cuda.is_available", return_value=True)
def test_to_device_obj_auto_cuda(mock_cuda):
    """Test to_device_obj auto-selects CUDA when available."""
    device = to_device_obj("auto")
    assert isinstance(device, torch.device)
    assert device.type == "cuda"


@pytest.mark.unit
@patch("torch.cuda.is_available", return_value=False)
def test_to_device_obj_auto_cpu(mock_cuda):
    """Test to_device_obj auto-selects CPU when no accelerators."""
    if hasattr(torch.backends, "mps"):
        with patch.object(torch.backends.mps, "is_available", return_value=False):
            device = to_device_obj("auto")
            assert device.type == "cpu"
    else:
        device = to_device_obj("auto")
        assert device.type == "cpu"


@pytest.mark.unit
def test_to_device_obj_invalid_device():
    """Test to_device_obj raises ValueError for unsupported device."""
    with pytest.raises(ValueError, match="Unsupported device"):
        to_device_obj("invalid_device")


@pytest.mark.unit
def test_to_device_obj_case_sensitivity():
    """Test to_device_obj is case-sensitive."""
    # Valid lowercase
    device = to_device_obj("cpu")
    assert device.type == "cpu"

    # Invalid uppercase should raise error
    with pytest.raises(ValueError, match="Unsupported device"):
        to_device_obj("CPU")


# CUDA UTILITIES
@pytest.mark.unit
@patch("torch.cuda.is_available", return_value=False)
def test_get_cuda_name_unavailable(mock_cuda):
    """Test get_cuda_name returns empty string when CUDA unavailable."""
    name = get_cuda_name()
    assert name == ""


@pytest.mark.unit
@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 3090")
def test_get_cuda_name_available(mock_name, mock_cuda):
    """Test get_cuda_name returns GPU model name when CUDA available."""
    name = get_cuda_name()
    assert name == "NVIDIA GeForce RTX 3090"


@pytest.mark.unit
@patch("torch.cuda.is_available", return_value=False)
def test_get_vram_info_unavailable(mock_cuda):
    """Test get_vram_info returns N/A when CUDA unavailable."""
    info = get_vram_info()
    assert info == "N/A"


@pytest.mark.unit
@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.device_count", return_value=1)
@patch("torch.cuda.mem_get_info", return_value=(8 * 1024**3, 16 * 1024**3))
def test_get_vram_info_available(mock_mem_info, mock_device_count, mock_cuda):
    """Test get_vram_info returns formatted VRAM string when CUDA available."""
    info = get_vram_info(device_idx=0)

    # Should contain both free and total memory
    assert "GB" in info
    assert "/" in info
    assert "8.00 GB" in info
    assert "16.00 GB" in info


@pytest.mark.unit
@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.device_count", return_value=1)
def test_get_vram_info_invalid_device_index(mock_device_count, mock_cuda):
    """Test get_vram_info handles invalid device index."""
    info = get_vram_info(device_idx=5)
    assert info == "Invalid Device Index"


@pytest.mark.unit
@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.device_count", return_value=1)
@patch("torch.cuda.mem_get_info", side_effect=RuntimeError("CUDA error"))
def test_get_vram_info_query_failed(mock_mem_info, mock_device_count, mock_cuda):
    """Test get_vram_info handles CUDA query failures gracefully."""
    info = get_vram_info(device_idx=0)
    assert info == "Query Failed"


# CPU THREAD MANAGEMENT
@pytest.mark.unit
@patch("os.cpu_count", return_value=8)
def test_get_num_workers_standard(mock_cpu_count):
    """Test get_num_workers returns half of CPU count for standard systems."""
    num_workers = get_num_workers()
    assert num_workers == 4


@pytest.mark.unit
@patch("os.cpu_count", return_value=16)
def test_get_num_workers_capped(mock_cpu_count):
    """Test get_num_workers caps at 8 workers for high-core systems."""
    num_workers = get_num_workers()
    assert num_workers == 8


@pytest.mark.unit
@patch("os.cpu_count", return_value=4)
def test_get_num_workers_low_cores(mock_cpu_count):
    """Test get_num_workers returns 2 for low-core systems."""
    num_workers = get_num_workers()
    assert num_workers == 2


@pytest.mark.unit
@patch("os.cpu_count", return_value=2)
def test_get_num_workers_very_low_cores(mock_cpu_count):
    """Test get_num_workers returns 2 for very low-core systems."""
    num_workers = get_num_workers()
    assert num_workers == 2


@pytest.mark.unit
@patch("os.cpu_count", return_value=None)
def test_get_num_workers_fallback(mock_cpu_count):
    """Test get_num_workers falls back to 2 when cpu_count is None."""
    num_workers = get_num_workers()
    assert num_workers == 2


@pytest.mark.unit
@patch("os.cpu_count", return_value=8)
def test_apply_cpu_threads_standard(mock_cpu_count):
    """Test apply_cpu_threads sets optimal thread count."""
    num_workers = 4
    threads = apply_cpu_threads(num_workers)

    # Should reserve threads for workers
    assert threads == 4  # 8 total - 4 workers
    assert torch.get_num_threads() == threads
    assert os.environ["OMP_NUM_THREADS"] == str(threads)
    assert os.environ["MKL_NUM_THREADS"] == str(threads)


@pytest.mark.unit
@patch("os.cpu_count", return_value=4)
def test_apply_cpu_threads_minimum(mock_cpu_count):
    """Test apply_cpu_threads maintains minimum of 2 threads."""
    num_workers = 8  # More workers than cores
    threads = apply_cpu_threads(num_workers)

    # Should never go below 2
    assert threads >= 2
    assert torch.get_num_threads() == threads


@pytest.mark.unit
@patch("os.cpu_count", return_value=None)
def test_apply_cpu_threads_fallback(mock_cpu_count):
    """Test apply_cpu_threads handles None cpu_count gracefully."""
    num_workers = 4
    threads = apply_cpu_threads(num_workers)

    # Should use fallback value (1 total - 4 workers = max(2, -3) = 2)
    assert threads >= 2


@pytest.mark.unit
@patch("os.cpu_count", return_value=16)
def test_apply_cpu_threads_high_core_system(mock_cpu_count):
    """Test apply_cpu_threads on high-core system."""
    num_workers = 4
    threads = apply_cpu_threads(num_workers)

    assert threads == 12  # 16 - 4
    assert torch.get_num_threads() == 12


# INTEGRATION TESTS
@pytest.mark.integration
@patch("os.cpu_count", return_value=8)
def test_full_hardware_workflow(mock_cpu_count):
    """Test complete hardware configuration workflow."""
    # Configure system libraries
    configure_system_libraries()

    # Detect best device
    device_str = detect_best_device()
    assert device_str in ["cuda", "mps", "cpu"]

    # Convert to device object (skip if cuda not available)
    if device_str != "cuda" or torch.cuda.is_available():
        device = to_device_obj(device_str)
        assert isinstance(device, torch.device)

    # Get optimal workers
    num_workers = get_num_workers()
    assert 2 <= num_workers <= 8

    # Apply CPU threading
    threads = apply_cpu_threads(num_workers)
    assert threads >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
