"""
Hardware Acceleration & Computing Environment.

This module provides high-level abstractions for hardware discovery (CUDA/MPS),
and compute resource optimization. It manages the detection of available 
accelerators and synchronizes PyTorch threading with system capabilities.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import os
import platform
import logging

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import torch
import matplotlib

# =========================================================================== #
#                               System Utilities                              #
# =========================================================================== #

def configure_system_libraries() -> None:
    """
    Configures third-party libraries for headless environments.
    Sets Matplotlib to 'Agg' backend on Linux/Docker to avoid GUI issues.
    Also sets logging level for Matplotlib to WARNING to reduce verbosity.
    """
    is_linux = platform.system() == "Linux"
    is_docker = any([
        os.environ.get("IN_DOCKER") == "TRUE",
        os.path.exists("/.dockerenv")
    ])
    
    if is_linux or is_docker:
        matplotlib.use("Agg")  
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        
    if platform.system() == "Windows":
        logging.debug("Windows environment detected: fcntl locking is unavailable.")

# =========================================================================== #
#                              Hardware Utilities                             #
# =========================================================================== #

def get_num_workers() -> int:
    """
    Determines optimal DataLoader workers with a safe cap for RAM stability.

    Returns:
        int: Recommended number of subprocesses for data loading.
    """
    total_cores = os.cpu_count() or 2
    if total_cores <= 4:
        return 2
    return min(total_cores // 2, 8)

def apply_cpu_threads(num_workers: int) -> int:
    """
    Calculates and sets optimal compute threads to avoid resource contention.
    Synchronizes PyTorch intra-op parallelism with OMP/MKL environment variables.
    
    Args:
        num_workers (int): Number of active DataLoader workers.

    Returns:
        int: The number of threads applied to the system.
    """
    total_cores = os.cpu_count() or 1
    optimal_threads = max(2, total_cores - num_workers)
    
    torch.set_num_threads(optimal_threads)
    os.environ["OMP_NUM_THREADS"] = str(optimal_threads)
    os.environ["MKL_NUM_THREADS"] = str(optimal_threads)

    return optimal_threads

def detect_best_device() -> str:
    """
    Detects the most performant hardware accelerator available (CUDA > MPS > CPU).
    
    Returns:
        str: The best available device string.
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_cuda_name() -> str:
    """
    Returns the human-readable name of the primary GPU device.
    
    Returns:
        str: GPU model name or empty string if unavailable.
    """
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""

def to_device_obj(device_str: str) -> torch.device:
    """
    Converts a device string into a live torch.device object.
    
    Args:
        device_str (str): Target device ('cuda', 'cpu', 'mps').

    Returns:
        torch.device: The active computing device object.
    """
    return torch.device(device_str)

def determine_tta_mode(use_tta: bool, device_type: str) -> str:
    """
    Defines TTA complexity based on hardware acceleration availability.
    
    Args:
        use_tta (bool): Whether Test-Time Augmentation is enabled.
        device_type (str): The type of active device ('cpu', 'cuda', 'mps').

    Returns:
        str: Descriptive string of the TTA operation mode.
    """
    if not use_tta:
        return "DISABLED"

    return f"FULL ({device_type.upper()})" if device_type != "cpu" else "LIGHT (CPU Optimized)"