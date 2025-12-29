"""
Hardware Acceleration & Reproducibility Environment.

This module provides high-level abstractions for hardware discovery (CUDA/MPS),
deterministic seeding across libraries, and compute resource optimization. 
It ensures that the execution context is synchronized between PyTorch, NumPy, 
and the underlying system libraries.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import os
import random
import platform
import logging

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import numpy as np
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
        os.environ.get("IS_DOCKER") == "1",
        os.path.exists("/.dockerenv")
    ])
    
    if is_linux or is_docker:
        matplotlib.use("Agg")  
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        
    # Note: The fcntl check is now partially handled by the 'processes' module,
    # but we keep the system-level warning here for environment awareness.
    if platform.system() == "Windows":
        logging.debug("Windows environment detected: fcntl locking is unavailable.")


# =========================================================================== #
#                              Hardware Utilities                             #
# =========================================================================== #

def set_seed(
        seed: int
) -> None:
    """
    Ensures deterministic behavior across Python, NumPy, and PyTorch.
    
    Args:
        seed (int): The seed value to set for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def worker_init_fn(worker_id: int) -> None:
    """
    Initializes random number generators for DataLoader workers to ensure 
    augmentation diversity and reproducibility.
    
    This function bridges the gap between PyTorch's multi-processing and 
    Python/NumPy's random states, preventing 'seed leakage' where different 
    workers produce identical augmentations.
    """
    # 1. Get the base seed from the parent process
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return

    # 2. Combine base seed with worker ID for a unique sub-seed
    base_seed = worker_info.seed 
    seed = (base_seed + worker_id) % 2**32

    # 3. Synchronize all major PRNGs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def apply_cpu_threads(
        num_workers: int
) -> int:
    """
    Calculates and sets optimal compute threads to avoid resource contention.
    Synchronizes PyTorch intra-op parallelism with OMP/MKL environment variables.
    
    Args:
        num_workers (int): Number of active DataLoader workers.

    Returns:
        int: The number of threads applied to the system.
    """
    total_cores = os.cpu_count() or 1
    # Balance: Leave cores for workers, but keep at least 2 for tensor math
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


def to_device_obj(
        device_str: str
) -> torch.device:
    """
    Converts a device string into a live torch.device object.
    
    Args:
        device_str (str): Target device ('cuda', 'cpu', 'mps').

    Returns:
        torch.device: The active computing device object.
    """
    return torch.device(device_str)


def determine_tta_mode(
        use_tta: bool,
        device_type: str
) -> str:
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