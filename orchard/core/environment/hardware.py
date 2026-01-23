"""
Hardware Acceleration & Computing Environment.

This module provides high-level abstractions for hardware discovery (CUDA/MPS),
and compute resource optimization. It manages the detection of available
accelerators and synchronizes PyTorch threading with system capabilities.
"""

import logging

# Standard Imports
import os
import platform

import matplotlib

# Third-Party Imports
import torch


# SYSTEM CONFIGURATION
def configure_system_libraries() -> None:
    """
    Configures libraries for headless environments and reduces logging noise.

    - Sets Matplotlib to 'Agg' backend on Linux/Docker (no GUI)
    - Configures font embedding for PDF/PS exports
    - Suppresses verbose Matplotlib warnings
    """
    is_linux = platform.system() == "Linux"
    is_docker = os.environ.get("IN_DOCKER") == "TRUE" or os.path.exists("/.dockerenv")

    if is_linux or is_docker:
        matplotlib.use("Agg")
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        logging.getLogger("matplotlib").setLevel(logging.WARNING)


# HARDWARE DETECTION
def detect_best_device() -> str:
    """
    Detects the most performant accelerator (CUDA > MPS > CPU).

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def to_device_obj(device_str: str) -> torch.device:
    """
    Converts device string to PyTorch device object.

    Args:
        device_str: 'cuda', 'cpu', or 'auto' (auto-selects best available)

    Returns:
        torch.device object

    Raises:
        ValueError: If CUDA requested but unavailable, or invalid device string
    """
    if device_str == "auto":
        device_str = detect_best_device()

    if device_str == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available")

    if device_str not in ("cuda", "cpu", "mps"):
        raise ValueError(f"Unsupported device: {device_str}")

    return torch.device(device_str)


def get_cuda_name() -> str:
    """Returns GPU model name or empty string if unavailable."""
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""


def get_vram_info(device_idx: int = 0) -> str:
    """
    Retrieves VRAM availability for a CUDA device.

    Args:
        device_idx: GPU index to query

    Returns:
        Formatted string 'X.XX GB / Y.YY GB' or status message
    """
    if not torch.cuda.is_available():
        return "N/A"

    try:
        if device_idx >= torch.cuda.device_count():
            return "Invalid Device Index"

        free, total = torch.cuda.mem_get_info(device_idx)
        return f"{free / 1024**3:.2f} GB / {total / 1024**3:.2f} GB"
    except Exception as e:
        logging.debug(f"VRAM query failed: {e}")
        return "Query Failed"


# CPU MANAGEMENT
def get_num_workers() -> int:
    """
    Determines optimal DataLoader workers with RAM stability cap.

    Returns:
        Recommended number of subprocesses (2-8 range)
    """
    total_cores = os.cpu_count() or 2
    if total_cores <= 4:
        return 2
    return min(total_cores // 2, 8)


def apply_cpu_threads(num_workers: int) -> int:
    """
    Sets optimal compute threads to avoid resource contention.
    Synchronizes PyTorch, OMP, and MKL thread counts.

    Args:
        num_workers: Active DataLoader workers

    Returns:
        Number of threads assigned to compute operations
    """
    total_cores = os.cpu_count() or 1
    optimal_threads = max(2, total_cores - num_workers)

    torch.set_num_threads(optimal_threads)
    os.environ["OMP_NUM_THREADS"] = str(optimal_threads)
    os.environ["MKL_NUM_THREADS"] = str(optimal_threads)

    return optimal_threads
