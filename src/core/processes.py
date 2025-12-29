"""
Process & Resource Guarding Utilities.

This module provides low-level OS abstractions for managing script execution 
lifecycle. It ensures system stability through exclusive file locking (flock) 
and duplicate process termination, preventing resource contention in multi-user 
or automated environments.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import os
import sys
import time
import platform
import logging
from pathlib import Path
from typing import Optional, IO

# Tentative import for Unix-specific file locking
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

# =========================================================================== #
#                                Third-Party Imports                             #
# =========================================================================== #
import psutil

# =========================================================================== #
#                               Global State                                  #
# =========================================================================== #
# Persistent file descriptor to prevent garbage collection from releasing locks
_lock_fd: Optional[IO] = None

# =========================================================================== #
#                              Process Management                             #
# =========================================================================== #

def ensure_single_instance(
    lock_file: Path,
    logger: logging.Logger
) -> None:
    """
    Implements a cooperative advisory lock to guarantee singleton execution.
    
    Leverages Unix 'flock' to create an exclusive lock on a sentinel file.
    If the lock cannot be acquired immediately, it indicates another instance 
    is active, and the process will abort to prevent filesystem or GPU 
    race conditions.

    Args:
        lock_file (Path): Filesystem path where the lock sentinel will reside.
        logger (logging.Logger): Active logger for reporting acquisition status.

    Raises:
        SystemExit: If an existing lock is detected on the system.
    """
    global _lock_fd
    
    # Locking is currently only supported on Unix-like systems via fcntl
    if platform.system() in ("Linux", "Darwin") and HAS_FCNTL:
        try:
            lock_file.parent.mkdir(parents=True, exist_ok=True)
            f = open(lock_file, 'a')
            
            # Attempt to acquire an exclusive lock without blocking
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _lock_fd = f 
            logger.info("Exclusive system lock acquired.")
            
        except (IOError, BlockingIOError):
            logger.error(" [!] CRITICAL: Another instance is already running. Aborting.")
            sys.exit(1)


def release_single_instance(
    lock_file: Path
) -> None:
    """
    Safely releases the system lock and unlinks the sentinel file.

    Guarantees that the file descriptor is closed and the lock is returned 
    to the OS. Designed to be called during normal shutdown or within 
    exception handling blocks.

    Args:
        lock_file (Path): Filesystem path to the sentinel file to be removed.
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
            # Silence errors if the file was already removed by another process
            pass


def kill_duplicate_processes(
    logger: Optional[logging.Logger] = None,
    script_name: Optional[str] = None
) -> None:
    """
    Proactively identifies and terminates ghost instances of the pipeline.
    
    Scans the system process table for Python instances executing the same 
    script name. Useful in shared environments to clear orphaned processes 
    before starting a new experimental run.

    Args:
        logger (Optional[logging.Logger]): Logger for reporting terminated PIDs.
        script_name (Optional[str]): Target script name. Defaults to current file.
    """
    if script_name is None:
        script_name = os.path.basename(sys.argv[0])
    
    current_pid = os.getpid()
    killed = 0
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            p = proc.info
            # Skip empty cmdlines or the current process itself
            if not p['cmdline'] or p['pid'] == current_pid:
                continue
            
            # Match script name in cmdline arguments and ensure it's a python process
            is_python = 'python' in p['name'].lower()
            is_target = any(script_name in arg for arg in p['cmdline'])
            
            if is_python and is_target:
                proc.terminate()
                killed += 1
                
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    
    if killed and logger:
        logger.info(f" » Cleaned {killed} duplicate process(es). Cooling down...")
        time.sleep(1.5)