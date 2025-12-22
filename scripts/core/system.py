"""
System and Hardware Utilities Module

This module provides low-level utilities for hardware abstraction (device selection),
reproducibility (seeding), file integrity (checksums), and process management.
"""

# =========================================================================== #
#                                Standard Imports
# =========================================================================== #
import os
# fcntl is a Unix-specific module for file locking. 
# This utility module currently supports Linux/macOS only.
import fcntl
import sys
import random
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional

# =========================================================================== #
#                                Third-Party Imports
# =========================================================================== #
import numpy as np
import torch
import psutil

# =========================================================================== #
#                               SYSTEM UTILITIES
# =========================================================================== #
# Variable to keep the lock file descriptor alive
_lock_fd: Optional[int] = None

def set_seed(seed: int) -> None:
    """
    Sets random seeds for reproducibility across NumPy, Python, and PyTorch.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Guarantees deterministic convolution algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(logger: logging.Logger) -> torch.device:
    """
    Determines the appropriate device (CUDA or CPU) for computation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Computing on: {device}")
    return device 


def md5_checksum(path: Path) -> str:
    """
    Calculates the MD5 checksum of a file in chunks for efficiency.
    """
    hash_md5 = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def validate_npz_keys(data: np.lib.npyio.NpzFile) -> None:
    """
    Validates that the loaded NPZ dataset contains all expected keys.

    Args:
        data (np.lib.npyio.NpzFile): The loaded NPZ file object.

    Raises:
        ValueError: If any required key is missing from the NPZ file.
    """
    required_keys = {
        "train_images", "train_labels",
        "val_images", "val_labels",
        "test_images", "test_labels",
    }

    missing = required_keys - set(data.files)
    if missing:
        raise ValueError(f"NPZ file is missing required keys: {missing}")

def kill_duplicate_processes(logger: logging.Logger, script_name: Optional[str] = None) -> None:
    """
    Kills duplicate Python processes to prevent resource or directory conflicts.
    """
    # Use the entry-point script name as default (usually main.py)
    if script_name is None:
        script_name = os.path.basename(sys.argv[0])
    
    current_pid = os.getpid()
    killed = 0
    # Common python executable names across platforms
    python_names = {'python', 'python3', 'python.exe', 'python3.exe'}

    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            pinfo = proc.info
            if pinfo['name'] not in python_names or pinfo['pid'] == current_pid:
                continue
            
            cmdline = pinfo['cmdline']
            if cmdline and any(script_name in arg for arg in cmdline):
                proc.terminate()
                killed += 1
                logger.warning(f"Terminated duplicate process: PID {pinfo['pid']}")
        
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if killed:
        logger.info(f"Cleaned up {killed} duplicate process(es). Cooling down...")
        time.sleep(1.5)


def ensure_single_instance(lock_file: Path, logger: logging.Logger) -> None:
    """
    Uses flock to ensure only one instance of the pipeline runs at a time.
    """
    global _lock_fd
    try:
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        # Open for writing and keep the file descriptor in a global variable
        # so the garbage collector doesn't close it and release the lock.
        f = open(lock_file, 'w')
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_fd = f 
        logger.info("Exclusive lock acquired.")
    except (IOError, BlockingIOError):
        logger.error("PROCESS ABORTED: Another instance is already running.")
        sys.exit(1)