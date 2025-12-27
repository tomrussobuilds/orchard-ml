"""
System and Hardware Utilities Module

This module provides low-level abstractions for hardware acceleration (CUDA/MPS),
environment-aware reproducibility (seeding, worker allocation), and OS-level 
process management (exclusive locking, duplicate termination).
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import os
import sys
import platform
import random
import time
import logging
from pathlib import Path
from typing import Optional, IO

try:
    import fcntl  # Unix-only
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

# =========================================================================== #
#                                Third-Party Imports                           #
# =========================================================================== #
import numpy as np
import torch
import psutil
import matplotlib

# =========================================================================== #
#                               SYSTEM UTILITIES                              #
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
        
    if not HAS_FCNTL and platform.system() != "Windows":
        logging.warning("fcntl module not available; exclusive locking disabled.")


# Global lock FD to prevent GC cleanup
_lock_fd: Optional[IO] = None

def ensure_single_instance(
        lock_file: Path,
        logger: logging.Logger
) -> None:
    """
    Uses flock (Unix) to ensure only one instance of the script runs.
    Aborts execution if another instance is detected.
    """
    global _lock_fd
    if platform.system() in ("Linux", "Darwin") and HAS_FCNTL:
        try:
            lock_file.parent.mkdir(parents=True, exist_ok=True)
            f = open(lock_file, 'a')
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _lock_fd = f 
            logger.info("Exclusive system lock acquired.")
        except (IOError, BlockingIOError):
            logger.error("CRITICAL: Another instance is already running. Aborting.")
            sys.exit(1)


def release_single_instance(
        lock_file: Path
) -> None:
    """
    Releases the exclusive system lock and performs cleanup of the lock file.
    """
    global _lock_fd
    if _lock_fd:
        try:
            if HAS_FCNTL:
                fcntl.flock(_lock_fd, fcntl.LOCK_UN)
            _lock_fd.close()
        finally:
            _lock_fd = None
    if lock_file.exists():
        try:
            lock_file.unlink()
        except OSError:
            pass


def kill_duplicate_processes(
        logger: logging.Logger,
        script_name: Optional[str] = None
) -> None:
    """
    Terminates other instances of the same script to prevent resource contention.
    Includes a cooldown period to allow OS to release resources.
    """
    if script_name is None:
        script_name = os.path.basename(sys.argv[0])
    
    current_pid = os.getpid()
    killed = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            p = proc.info
            if not p['cmdline'] or p['pid'] == current_pid:
                continue
            if any(script_name in arg for arg in p['cmdline']) and 'python' in p['name'].lower():
                proc.terminate()
                killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    
    if killed:
        logger.info(f"Cleaned {killed} duplicate(s). Cooling down...")
        time.sleep(1.5)


# =========================================================================== #
#                                HARDWARE UTILITIES                           #
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
    if os.environ.get("DOCKER_REPRODUCIBILITY_MODE", "0").upper() in ("1", "TRUE"):
        return 0
    return min(os.cpu_count() or 2, 8)


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
    optimal_threads = total_cores if num_workers == 0 else max(2, total_cores - num_workers)
    
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


def load_model_weights(
        model: torch.nn.Module, 
        path: Path, 
        device: torch.device
) -> None:
    """
    Restores model state from a checkpoint using secure weight-only loading.
    
    Args:
        model (torch.nn.Module): The model instance to populate.
        path (Path): Filesystem path to the checkpoint file.
        device (torch.device): Target device for mapping the tensors.
    """
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)


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