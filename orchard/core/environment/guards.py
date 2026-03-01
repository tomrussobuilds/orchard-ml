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

from __future__ import annotations

import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import IO

# Tentative import for Unix-specific file locking
try:
    import fcntl

    HAS_FCNTL = True
except ImportError:  # pragma: no cover
    HAS_FCNTL = False

import psutil

from ..paths.constants import LogStyle
from .distributed import is_distributed, is_main_process

# Global State
# Persistent file descriptor to prevent garbage collection from releasing locks
_lock_fd: IO | None = None


# PROCESS MANAGEMENT
def ensure_single_instance(lock_file: Path, logger: logging.Logger) -> None:
    """
    Implements a cooperative advisory lock to guarantee singleton execution.

    Leverages Unix 'flock' to create an exclusive lock on a sentinel file.
    If the lock cannot be acquired immediately, it indicates another instance
    is active, and the process will abort to prevent filesystem or GPU
    race conditions.

    In distributed mode (torchrun / DDP), only the main process (rank 0)
    acquires the lock.  Non-main ranks skip locking entirely to avoid
    deadlocking against the rank-0 held lock.

    Args:
        lock_file (Path): Filesystem path where the lock sentinel will reside.
        logger (logging.Logger): Active logger for reporting acquisition status.

    Raises:
        SystemExit: If an existing lock is detected on the system.
    """
    global _lock_fd

    # In distributed mode, only rank 0 manages the lock
    if not is_main_process():
        logger.debug(  # pragma: no mutant
            "Rank %d: skipping lock acquisition (non-main process).", os.getpid()
        )
        return

    # Locking is currently only supported on Unix-like systems via fcntl
    if platform.system() in ("Linux", "Darwin") and HAS_FCNTL:
        f: IO | None = None
        try:
            lock_file.parent.mkdir(parents=True, exist_ok=True)
            f = open(lock_file, "a")

            # Attempt to acquire an exclusive lock without blocking
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _lock_fd = f
            logger.info(f"  {LogStyle.ARROW} System lock acquired")  # pragma: no mutant

        except (IOError, BlockingIOError):
            if f is not None:
                f.close()
            logger.error(
                f" {LogStyle.WARNING} CRITICAL: Another instance is already running. Aborting."
            )
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
            except (OSError, IOError):  # pragma: no cover
                # Close may fail if fd is already closed
                pass
        finally:
            _lock_fd = None

    # Attempt unlink directly to avoid TOCTOU race condition
    # (file could be deleted between exists() check and unlink() call)
    try:
        lock_file.unlink()
    except FileNotFoundError:
        # File was already removed by another process - expected in race conditions
        pass
    except OSError:  # pragma: no cover
        # Other OS errors (permissions, etc.) - safe to ignore during cleanup
        pass


class DuplicateProcessCleaner:
    """
    Scans and optionally terminates duplicate instances of the current script.

    Attributes:
        script_path (str): Absolute path of the script to match against running processes.
        current_pid (int): PID of the current process.
    """

    def __init__(self, script_name: str | None = None) -> None:
        self.script_path = os.path.realpath(script_name or sys.argv[0])
        self.current_pid = os.getpid()

    def detect_duplicates(self) -> list[psutil.Process]:
        """
        Detects other Python processes running the same script.

        Returns:
            list of psutil.Process instances representing duplicates.
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

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return duplicates

    def terminate_duplicates(self, logger: logging.Logger | None = None) -> int:
        """
        Terminates detected duplicate processes.

        In distributed mode (torchrun / DDP), termination is skipped entirely
        because sibling rank processes are intentional, not duplicates.

        Args:
            logger (logging.Logger | None): Logger for reporting terminated PIDs.

        Returns:
            Number of terminated duplicate processes (0 in distributed mode).
        """
        if is_distributed():
            if logger:
                logger.debug(  # pragma: no mutant
                    "Distributed mode: skipping duplicate process cleanup."
                )
            return 0

        duplicates = self.detect_duplicates()
        count = 0

        for proc in duplicates:
            try:
                proc.terminate()
                proc.wait(timeout=1)
                count += 1
                continue
            except psutil.TimeoutExpired:
                # Graceful SIGTERM timed out — fall through to SIGKILL escalation
                pass
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

            # If terminate failed, try kill
            try:
                proc.kill()
                proc.wait(timeout=1)
                count += 1
            except (psutil.TimeoutExpired, psutil.NoSuchProcess, psutil.AccessDenied):
                # SIGKILL also failed or process vanished — nothing more we can do
                continue

        if count and logger:
            logger.info(  # pragma: no mutant
                f" {LogStyle.ARROW} Cleaned {count} duplicate process(es). Cooling down..."
            )
            time.sleep(1.5)

        return count
