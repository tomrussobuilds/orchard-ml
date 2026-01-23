"""
Process & Resource Guarding Utilities.

Provides low-level OS abstractions to manage Python script execution
in multi-user or shared environments. It ensures system stability
and safe resource usage by offering:

- **Exclusive filesystem locking** (`flock`) to prevent concurrent runs
  and protect against disk or GPU/MPS conflicts.
- **Duplicate process detection and optional termination** to free
  resources and avoid interference.

These utilities ensure each run is isolated, reproducible, and safe
even on clusters or shared systems.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import IO, Optional

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


def ensure_single_instance(lock_file: Path, logger: logging.Logger) -> None:
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
            f = open(lock_file, "a")

            # Attempt to acquire an exclusive lock without blocking
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _lock_fd = f
            logger.info("Exclusive system lock acquired.")

        except (IOError, BlockingIOError):
            logger.error(" [!] CRITICAL: Another instance is already running. Aborting.")
            sys.exit(1)


def release_single_instance(lock_file: Path) -> None:
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
                try:
                    fcntl.flock(_lock_fd, fcntl.LOCK_UN)
                except (OSError, IOError):
                    # Unlock may fail if process is already terminated
                    pass

            try:
                _lock_fd.close()
            except (OSError, IOError):
                # Close may fail if fd is already closed
                pass
        finally:
            _lock_fd = None

    if lock_file.exists():
        try:
            lock_file.unlink()
        except OSError:
            # Silence errors if the file was already removed by another process
            pass


class DuplicateProcessCleaner:
    """
    Scans and optionally terminates duplicate instances of the current script.

    Attributes:
        script_path (str): Absolute path of the script to match against running processes.
        current_pid (int): PID of the current process.
    """

    def __init__(self, script_name: Optional[str] = None):
        self.script_path = os.path.realpath(script_name or sys.argv[0])
        self.current_pid = os.getpid()

    def detect_duplicates(self) -> list[psutil.Process]:
        """
        Detects other Python processes running the same script.

        Returns:
            List of psutil.Process instances representing duplicates.
        """
        duplicates = []

        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                info = proc.info
                if not info["cmdline"] or info["pid"] == self.current_pid:
                    continue

                # Check if process is Python
                cmd0 = os.path.basename(info["cmdline"][0]).lower()
                if "python" not in cmd0:
                    continue

                # Match exact script path in cmdline
                cmdline_paths = [os.path.realpath(arg) for arg in info["cmdline"][1:]]
                if self.script_path in cmdline_paths:
                    duplicates.append(proc)

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        return duplicates

    def terminate_duplicates(self, logger: Optional[logging.Logger] = None) -> int:
        """
        Terminates detected duplicate processes.

        Args:
            logger (Optional[logging.Logger]): Logger for reporting terminated PIDs.

        Returns:
            Number of terminated duplicate processes.
        """
        duplicates = self.detect_duplicates()
        count = 0

        for proc in duplicates:
            try:
                proc.terminate()
                proc.wait(timeout=1)
                count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        if count and logger:
            logger.info(f" » Cleaned {count} duplicate process(es). Cooling down...")
            time.sleep(1.5)

        return count
