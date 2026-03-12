"""
Test Suite for Process & Resource Guarding Utilities.

Tests filesystem locking, duplicate process detection,
and process termination utilities.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import psutil
import pytest

from orchard.core.environment import (
    DuplicateProcessCleaner,
    ensure_single_instance,
    release_single_instance,
)


# SINGLE INSTANCE LOCKING
@pytest.mark.unit
@patch("platform.system", return_value="Linux")
@patch("orchard.core.environment.guards.HAS_FCNTL", True)
def test_ensure_single_instance_success(mock_platform, tmp_path):
    """Test ensure_single_instance acquires lock successfully."""
    import fcntl

    from orchard.core.environment import guards

    lock_file = tmp_path / "test.lock"
    logger = logging.getLogger("test")

    try:
        with patch("fcntl.flock") as mock_flock:
            ensure_single_instance(lock_file, logger)

            mock_flock.assert_called_once()
            # Verify exclusive non-blocking flags
            call_args = mock_flock.call_args
            assert call_args[0][1] == (fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert lock_file.parent.exists()

        # Verify global _lock_fd was set
        assert guards._lock_fd is not None
    finally:
        if guards._lock_fd is not None:
            guards._lock_fd.close()
            guards._lock_fd = None


@pytest.mark.unit
@patch("platform.system", return_value="Linux")
@patch("orchard.core.environment.guards.HAS_FCNTL", True)
def test_ensure_single_instance_already_locked(mock_platform, tmp_path):
    """Test ensure_single_instance exits when lock already held and closes fd."""
    lock_file = tmp_path / "test.lock"
    logger = logging.getLogger("test")
    mock_file = MagicMock()

    with patch("builtins.open", return_value=mock_file):
        with patch("fcntl.flock", side_effect=BlockingIOError):
            with pytest.raises(SystemExit) as exc_info:
                ensure_single_instance(lock_file, logger)

            assert exc_info.value.code == 1
            # Verify fd is closed on lock failure
            mock_file.close.assert_called_once()


@pytest.mark.unit
@patch("platform.system", return_value="Linux")
@patch("orchard.core.environment.guards.HAS_FCNTL", True)
def test_ensure_single_instance_io_error(mock_platform, tmp_path):
    """Test ensure_single_instance handles IOError."""
    lock_file = tmp_path / "test.lock"
    logger = logging.getLogger("test")

    with patch("fcntl.flock", side_effect=IOError):
        with pytest.raises(SystemExit) as exc_info:
            ensure_single_instance(lock_file, logger)

        assert exc_info.value.code == 1


@pytest.mark.unit
@patch("platform.system", return_value="Windows")
def test_ensure_single_instance_windows(mock_platform, tmp_path):
    """Test ensure_single_instance skips locking on Windows."""
    lock_file = tmp_path / "test.lock"
    logger = logging.getLogger("test")

    ensure_single_instance(lock_file, logger)


@pytest.mark.unit
@patch("platform.system", return_value="Darwin")
@patch("orchard.core.environment.guards.HAS_FCNTL", True)
def test_ensure_single_instance_macos(mock_platform, tmp_path):
    """Test ensure_single_instance works on macOS."""
    from orchard.core.environment import guards

    lock_file = tmp_path / "test.lock"
    logger = logging.getLogger("test")

    try:
        with patch("fcntl.flock") as mock_flock:
            ensure_single_instance(lock_file, logger)
            mock_flock.assert_called_once()
    finally:
        if guards._lock_fd is not None:
            guards._lock_fd.close()
            guards._lock_fd = None


@pytest.mark.unit
@patch("platform.system", return_value="Linux")
@patch("orchard.core.environment.guards.HAS_FCNTL", False)
def test_ensure_single_instance_no_fcntl(mock_platform, tmp_path):
    """Test ensure_single_instance when fcntl not available."""
    lock_file = tmp_path / "test.lock"
    logger = logging.getLogger("test")

    ensure_single_instance(lock_file, logger)


# LOCK RELEASE
@pytest.mark.unit
@patch("orchard.core.environment.guards.HAS_FCNTL", True)
def test_release_single_instance_with_lock(tmp_path):
    """Test release_single_instance releases lock, closes fd, resets global, and removes file."""
    import fcntl

    from orchard.core.environment import guards

    lock_file = tmp_path / "test.lock"
    lock_file.touch()

    mock_fd = MagicMock()
    with patch("orchard.core.environment.guards._lock_fd", mock_fd):
        with patch("fcntl.flock") as mock_flock:
            release_single_instance(lock_file)

            # Verify unlock with correct flag
            mock_flock.assert_called_once_with(mock_fd, fcntl.LOCK_UN)
            mock_fd.close.assert_called_once()

        # Global must be reset to None (check inside patch scope)
        assert guards._lock_fd is None

    assert not lock_file.exists()


@pytest.mark.unit
def test_release_single_instance_no_lock(tmp_path):
    """Test release_single_instance when no lock is held."""
    lock_file = tmp_path / "test.lock"
    lock_file.touch()

    with patch("orchard.core.environment.guards._lock_fd", None):
        release_single_instance(lock_file)

    assert not lock_file.exists()


@pytest.mark.unit
def test_release_single_instance_file_not_exists(tmp_path):
    """Test release_single_instance when lock file doesn't exist."""
    lock_file = tmp_path / "nonexistent.lock"

    with patch("orchard.core.environment.guards._lock_fd", None):
        release_single_instance(lock_file)


@pytest.mark.unit
@patch("orchard.core.environment.guards.HAS_FCNTL", True)
def test_release_single_instance_oserror(tmp_path):
    """Test release_single_instance handles OSError gracefully."""
    lock_file = tmp_path / "test.lock"

    mock_fd = MagicMock()
    with patch("orchard.core.environment.guards._lock_fd", mock_fd):
        with patch("fcntl.flock"):
            with patch.object(Path, "unlink", side_effect=OSError):
                release_single_instance(lock_file)


# DUPLICATE PROCESS CLEANER: INITIALIZATION
@pytest.mark.unit
def test_duplicate_process_cleaner_init_default():
    """Test DuplicateProcessCleaner initializes with default script name."""
    cleaner = DuplicateProcessCleaner()

    assert cleaner.script_path == str(Path(sys.argv[0]).resolve())
    assert cleaner.current_pid == os.getpid()


@pytest.mark.unit
def test_duplicate_process_cleaner_init_custom_script():
    """Test DuplicateProcessCleaner initializes with custom script name."""
    custom_script = "/path/to/custom_script.py"

    cleaner = DuplicateProcessCleaner(script_name=custom_script)

    assert cleaner.script_path == custom_script


# DUPLICATE PROCESS CLEANER: DETECTION
@pytest.mark.unit
def test_detect_duplicates_no_duplicates():
    """Test detect_duplicates returns empty list when no duplicates."""
    cleaner = DuplicateProcessCleaner()

    mock_procs = []
    with patch("psutil.process_iter", return_value=mock_procs):
        duplicates = cleaner.detect_duplicates()

    assert duplicates == []


@pytest.mark.unit
def test_detect_duplicates_skips_self():
    """Test detect_duplicates skips current process."""
    cleaner = DuplicateProcessCleaner(script_name="test.py")

    mock_proc = MagicMock()
    mock_proc.info = {
        "pid": cleaner.current_pid,
        "name": "python",
        "cmdline": ["python", str(Path("test.py").resolve())],
    }

    with patch("psutil.process_iter", return_value=[mock_proc]):
        duplicates = cleaner.detect_duplicates()

    assert duplicates == []


@pytest.mark.unit
def test_detect_duplicates_skips_non_python():
    """Test detect_duplicates skips non-Python processes (python not in cmd[0])."""
    cleaner = DuplicateProcessCleaner(script_name="test.py")

    mock_proc = MagicMock()
    mock_proc.info = {
        "pid": 9999,
        "name": "bash",
        "cmdline": ["bash", str(Path("test.py").resolve())],
    }

    with patch("psutil.process_iter", return_value=[mock_proc]):
        duplicates = cleaner.detect_duplicates()

    # Even though script path matches, process is not Python — must be skipped
    assert duplicates == []


@pytest.mark.unit
def test_detect_duplicates_finds_duplicate():
    """Test detect_duplicates finds matching Python process."""
    script_path = "/path/to/test.py"
    cleaner = DuplicateProcessCleaner(script_name=script_path)

    mock_proc = MagicMock()
    mock_proc.info = {
        "pid": 9999,
        "name": "python3",
        "cmdline": ["python3", script_path],
    }

    with patch("psutil.process_iter", return_value=[mock_proc]):
        duplicates = cleaner.detect_duplicates()

    assert len(duplicates) == 1
    assert duplicates[0] == mock_proc


@pytest.mark.unit
def test_detect_duplicates_handles_exceptions():
    """Test detect_duplicates handles psutil exceptions gracefully."""
    cleaner = DuplicateProcessCleaner()

    mock_proc1 = MagicMock()
    mock_proc1.info = {"pid": 1000, "name": "python", "cmdline": ["python"]}

    mock_proc2 = MagicMock()
    mock_proc2.info = MagicMock(side_effect=psutil.NoSuchProcess(9999))

    with patch("psutil.process_iter", return_value=[mock_proc1, mock_proc2]):
        duplicates = cleaner.detect_duplicates()

    assert isinstance(duplicates, list)


@pytest.mark.unit
def test_detect_duplicates_empty_cmdline():
    """Test detect_duplicates skips processes with empty cmdline."""
    cleaner = DuplicateProcessCleaner()

    mock_proc = MagicMock()
    mock_proc.info = {
        "pid": 9999,
        "name": "python",
        "cmdline": [],
    }

    with patch("psutil.process_iter", return_value=[mock_proc]):
        duplicates = cleaner.detect_duplicates()

    assert duplicates == []


# DUPLICATE PROCESS CLEANER: TERMINATION
@pytest.mark.unit
def test_terminate_duplicates_no_duplicates():
    """Test terminate_duplicates returns 0 when no duplicates found."""
    cleaner = DuplicateProcessCleaner()

    with patch.object(cleaner, "detect_duplicates", return_value=[]):
        count = cleaner.terminate_duplicates()

    assert count == 0


@pytest.mark.unit
def test_terminate_duplicates_success():
    """Test terminate_duplicates successfully terminates processes."""
    cleaner = DuplicateProcessCleaner()
    logger = logging.getLogger("test")

    # Mock duplicate process
    mock_proc = MagicMock()
    mock_proc.terminate = MagicMock()
    mock_proc.wait = MagicMock()

    with patch.object(cleaner, "detect_duplicates", return_value=[mock_proc]):
        count = cleaner.terminate_duplicates(logger=logger)

    assert count == 1
    mock_proc.terminate.assert_called_once()
    mock_proc.wait.assert_called_once_with(timeout=1)


@pytest.mark.unit
def test_terminate_duplicates_multiple():
    """Test terminate_duplicates handles multiple processes."""
    cleaner = DuplicateProcessCleaner()

    mock_procs = [MagicMock() for _ in range(3)]
    for proc in mock_procs:
        proc.terminate = MagicMock()
        proc.wait = MagicMock()

    with patch.object(cleaner, "detect_duplicates", return_value=mock_procs):
        with patch("time.sleep"):
            count = cleaner.terminate_duplicates()

    assert count == 3


@pytest.mark.unit
def test_terminate_duplicates_handles_no_such_process():
    """Test terminate_duplicates handles NoSuchProcess exception."""
    cleaner = DuplicateProcessCleaner()

    mock_proc = MagicMock()
    mock_proc.terminate.side_effect = psutil.NoSuchProcess(9999)

    with patch.object(cleaner, "detect_duplicates", return_value=[mock_proc]):
        count = cleaner.terminate_duplicates()

    assert count == 0


@pytest.mark.unit
def test_terminate_duplicates_handles_access_denied():
    """Test terminate_duplicates handles AccessDenied exception."""
    cleaner = DuplicateProcessCleaner()

    mock_proc = MagicMock()
    mock_proc.terminate.side_effect = psutil.AccessDenied()

    with patch.object(cleaner, "detect_duplicates", return_value=[mock_proc]):
        count = cleaner.terminate_duplicates()

    assert count == 0


@pytest.mark.unit
def test_detect_duplicates_handles_no_such_process():
    """Test detect_duplicates handles NoSuchProcess exception."""
    cleaner = DuplicateProcessCleaner()

    mock_proc = MagicMock()
    type(mock_proc).info = PropertyMock(side_effect=psutil.NoSuchProcess(9999))

    with patch("psutil.process_iter", return_value=[mock_proc]):
        duplicates = cleaner.detect_duplicates()

    assert duplicates == []


@pytest.mark.unit
def test_detect_duplicates_handles_access_denied():
    """Test detect_duplicates handles AccessDenied exception."""
    cleaner = DuplicateProcessCleaner()

    mock_proc = MagicMock()
    type(mock_proc).info = PropertyMock(side_effect=psutil.AccessDenied())

    with patch("psutil.process_iter", return_value=[mock_proc]):
        duplicates = cleaner.detect_duplicates()

    assert duplicates == []


@pytest.mark.unit
def test_detect_duplicates_handles_zombie_process():
    """Test detect_duplicates handles ZombieProcess exception."""
    cleaner = DuplicateProcessCleaner()

    mock_proc = MagicMock()
    type(mock_proc).info = PropertyMock(side_effect=psutil.ZombieProcess(9999))

    with patch("psutil.process_iter", return_value=[mock_proc]):
        duplicates = cleaner.detect_duplicates()

    assert duplicates == []


@pytest.mark.unit
def test_terminate_duplicates_with_logger():
    """Test terminate_duplicates logs termination and applies cooldown."""
    cleaner = DuplicateProcessCleaner()
    logger = MagicMock()

    mock_proc = MagicMock()

    with patch.object(cleaner, "detect_duplicates", return_value=[mock_proc]):
        with patch("time.sleep") as mock_sleep:
            count = cleaner.terminate_duplicates(logger=logger)

    assert count == 1
    logger.info.assert_called_once()
    mock_sleep.assert_called_once_with(1.5)


@pytest.mark.unit
@patch("orchard.core.environment.guards.HAS_FCNTL", True)
def test_release_single_instance_unlock_ioerror_real(tmp_path):
    """Test release_single_instance handles IOError during unlock."""
    lock_file = tmp_path / "test.lock"
    lock_file.touch()

    with open(lock_file, "a") as mock_fd:
        with patch("orchard.core.environment.guards._lock_fd", mock_fd):
            with patch("fcntl.flock", side_effect=IOError("Unlock IO failed")):
                release_single_instance(lock_file)

    assert not lock_file.exists()


@pytest.mark.unit
@patch("orchard.core.environment.guards.HAS_FCNTL", True)
def test_release_single_instance_close_ioerror_real(tmp_path):
    """Test release_single_instance handles IOError during close."""
    lock_file = tmp_path / "test.lock"
    lock_file.touch()

    real_fd = open(lock_file, "a")
    original_close = real_fd.close

    def mock_close_ioerror():
        raise IOError("Close IO failed")

    real_fd.close = mock_close_ioerror

    with patch("orchard.core.environment.guards._lock_fd", real_fd):
        with patch("fcntl.flock"):
            release_single_instance(lock_file)

    try:
        original_close()
    except OSError:
        pass

    assert not lock_file.exists()


@pytest.mark.unit
@patch("platform.system", return_value="Linux")
@patch("orchard.core.environment.guards.HAS_FCNTL", True)
def test_ensure_single_instance_creates_nested_parent_dirs(mock_platform, tmp_path):
    """Test ensure_single_instance creates deeply nested parent directories."""
    import uuid

    from orchard.core.environment import guards

    unique_id = str(uuid.uuid4())
    lock_file = tmp_path / unique_id / "level1" / "level2" / "test.lock"
    logger = logging.getLogger("test")

    assert not lock_file.parent.exists()

    try:
        with patch("fcntl.flock"):
            ensure_single_instance(lock_file, logger)

        assert lock_file.parent.exists()
        assert lock_file.exists()
    finally:
        if guards._lock_fd is not None:
            guards._lock_fd.close()
            guards._lock_fd = None


@pytest.mark.unit
def test_terminate_duplicates_timeout_then_kill_success():
    """Test terminate_duplicates falls back to kill() after TimeoutExpired."""
    cleaner = DuplicateProcessCleaner()

    mock_proc = MagicMock()
    mock_proc.terminate = MagicMock()
    # First wait() raises TimeoutExpired, second wait() (after kill) succeeds
    mock_proc.wait = MagicMock(side_effect=[psutil.TimeoutExpired(1), None])
    mock_proc.kill = MagicMock()

    with patch.object(cleaner, "detect_duplicates", return_value=[mock_proc]):
        with patch("time.sleep"):
            count = cleaner.terminate_duplicates()

    assert count == 1
    mock_proc.terminate.assert_called_once()
    mock_proc.kill.assert_called_once()
    assert mock_proc.wait.call_count == 2


@pytest.mark.unit
def test_terminate_duplicates_timeout_then_kill_fails():
    """Test terminate_duplicates handles exception after kill()."""
    cleaner = DuplicateProcessCleaner()

    mock_proc = MagicMock()
    mock_proc.terminate = MagicMock()
    # First wait() raises TimeoutExpired, second wait() also raises TimeoutExpired
    mock_proc.wait = MagicMock(side_effect=[psutil.TimeoutExpired(1), psutil.TimeoutExpired(1)])
    mock_proc.kill = MagicMock()

    with patch.object(cleaner, "detect_duplicates", return_value=[mock_proc]):
        count = cleaner.terminate_duplicates()

    # Process couldn't be terminated, count stays 0
    assert count == 0
    mock_proc.terminate.assert_called_once()
    mock_proc.kill.assert_called_once()


@pytest.mark.unit
def test_terminate_duplicates_timeout_kill_no_such_process():
    """Test terminate_duplicates handles NoSuchProcess after kill()."""
    cleaner = DuplicateProcessCleaner()

    mock_proc = MagicMock()
    mock_proc.terminate = MagicMock()
    mock_proc.wait = MagicMock(side_effect=psutil.TimeoutExpired(1))
    mock_proc.kill = MagicMock(side_effect=psutil.NoSuchProcess(9999))

    with patch.object(cleaner, "detect_duplicates", return_value=[mock_proc]):
        count = cleaner.terminate_duplicates()

    assert count == 0
    mock_proc.kill.assert_called_once()


# RANK-AWARE LOCKING
@pytest.mark.unit
@patch("platform.system", return_value="Linux")
@patch("orchard.core.environment.guards.HAS_FCNTL", True)
def test_ensure_single_instance_skips_on_non_main_rank(mock_platform, tmp_path, monkeypatch):
    """Test ensure_single_instance skips lock acquisition for non-main rank."""
    monkeypatch.setenv("RANK", "1")
    lock_file = tmp_path / "test.lock"
    logger = logging.getLogger("test")

    with patch("fcntl.flock") as mock_flock:
        ensure_single_instance(lock_file, logger)

        mock_flock.assert_not_called()


@pytest.mark.unit
@patch("platform.system", return_value="Linux")
@patch("orchard.core.environment.guards.HAS_FCNTL", True)
def test_ensure_single_instance_acquires_on_rank_zero(mock_platform, tmp_path, monkeypatch):
    """Test ensure_single_instance acquires lock for rank 0."""
    from orchard.core.environment import guards

    monkeypatch.setenv("RANK", "0")
    lock_file = tmp_path / "test.lock"
    logger = logging.getLogger("test")

    try:
        with patch("fcntl.flock") as mock_flock:
            ensure_single_instance(lock_file, logger)
            mock_flock.assert_called_once()
    finally:
        if guards._lock_fd is not None:
            guards._lock_fd.close()
            guards._lock_fd = None


@pytest.mark.unit
@patch("platform.system", return_value="Linux")
@patch("orchard.core.environment.guards.HAS_FCNTL", True)
def test_ensure_single_instance_acquires_when_no_rank_env(mock_platform, tmp_path, monkeypatch):
    """Test ensure_single_instance acquires lock when RANK is not set (single-process)."""
    from orchard.core.environment import guards

    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    lock_file = tmp_path / "test.lock"
    logger = logging.getLogger("test")

    try:
        with patch("fcntl.flock") as mock_flock:
            ensure_single_instance(lock_file, logger)
            mock_flock.assert_called_once()
    finally:
        if guards._lock_fd is not None:
            guards._lock_fd.close()
            guards._lock_fd = None


# RANK-AWARE DUPLICATE PROCESS CLEANUP
@pytest.mark.unit
def test_terminate_duplicates_skips_in_distributed_mode(monkeypatch):
    """Test terminate_duplicates returns 0 in distributed mode without scanning."""
    monkeypatch.setenv("RANK", "0")
    cleaner = DuplicateProcessCleaner()
    logger = logging.getLogger("test")

    with patch.object(cleaner, "detect_duplicates") as mock_detect:
        count = cleaner.terminate_duplicates(logger=logger)

    assert count == 0
    mock_detect.assert_not_called()


@pytest.mark.unit
def test_terminate_duplicates_runs_outside_distributed(monkeypatch):
    """Test terminate_duplicates runs normally when not in distributed mode."""
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    cleaner = DuplicateProcessCleaner()

    with patch.object(cleaner, "detect_duplicates", return_value=[]):
        count = cleaner.terminate_duplicates()

    assert count == 0


# MUTATION: ensure_single_instance — `and` vs `or` for platform/HAS_FCNTL condition
@pytest.mark.unit
@patch("platform.system", return_value="Windows")
@patch("orchard.core.environment.guards.HAS_FCNTL", True)
def test_ensure_single_instance_windows_no_lock_even_with_fcntl(mock_platform, tmp_path):
    """Kills mutmut_2: `and` → `or`. On Windows fcntl must NOT be used even if available."""
    lock_file = tmp_path / "test.lock"
    logger = logging.getLogger("test")

    with patch("fcntl.flock") as mock_flock:
        ensure_single_instance(lock_file, logger)
        mock_flock.assert_not_called()


# MUTATION: ensure_single_instance — `f = None` vs `f = ""`
@pytest.mark.unit
@patch("platform.system", return_value="Linux")
@patch("orchard.core.environment.guards.HAS_FCNTL", True)
def test_ensure_single_instance_open_raises_ioerror(mock_platform, tmp_path):
    """Kills mutmut_10: `f = None` → `f = ""`. If open() raises IOError, f stays None and close must NOT be called."""
    lock_file = tmp_path / "test.lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("test")

    with patch("builtins.open", side_effect=IOError("Cannot open")):
        with pytest.raises(SystemExit) as exc_info:
            ensure_single_instance(lock_file, logger)
        assert exc_info.value.code == 1


# MUTATION: ensure_single_instance — logger.error args (LogStyle.WARNING)
@pytest.mark.unit
@patch("platform.system", return_value="Linux")
@patch("orchard.core.environment.guards.HAS_FCNTL", True)
def test_ensure_single_instance_error_log_includes_warning_style(mock_platform, tmp_path):
    """Kills mutmut_31 and mutmut_32: logger.error must include LogStyle.WARNING as second arg."""
    from orchard.core.paths.constants import LogStyle

    lock_file = tmp_path / "test.lock"
    logger = MagicMock()

    with patch("fcntl.flock", side_effect=BlockingIOError):
        with pytest.raises(SystemExit):
            ensure_single_instance(lock_file, logger)

    logger.error.assert_called_once()
    call_args = logger.error.call_args[0]
    assert len(call_args) == 2, "logger.error must be called with format string + LogStyle.WARNING"
    assert call_args[1] is LogStyle.WARNING


# MUTATION: detect_duplicates — `continue` vs `break` when skipping self
@pytest.mark.unit
def test_detect_duplicates_continues_after_skipping_self():
    """Kills mutmut_17: `continue` → `break`. Must find duplicate even after skipping self."""
    script_path = "/path/to/test.py"
    cleaner = DuplicateProcessCleaner(script_name=script_path)

    # First process is self (should be skipped, not break)
    mock_self = MagicMock()
    mock_self.info = {
        "pid": cleaner.current_pid,
        "name": "python",
        "cmdline": ["python", script_path],
    }
    # Second process is a valid duplicate
    mock_dup = MagicMock()
    mock_dup.info = {
        "pid": 9999,
        "name": "python3",
        "cmdline": ["python3", script_path],
    }

    with patch("psutil.process_iter", return_value=[mock_self, mock_dup]):
        duplicates = cleaner.detect_duplicates()

    assert len(duplicates) == 1
    assert duplicates[0] is mock_dup


# MUTATION: detect_duplicates — `continue` vs `break` when skipping non-Python
@pytest.mark.unit
def test_detect_duplicates_continues_after_non_python():
    """Kills mutmut_27: `continue` → `break`. Must find duplicate even after skipping non-Python."""
    script_path = "/path/to/test.py"
    cleaner = DuplicateProcessCleaner(script_name=script_path)

    # First: non-Python process
    mock_bash = MagicMock()
    mock_bash.info = {
        "pid": 8888,
        "name": "bash",
        "cmdline": ["bash", script_path],
    }
    # Second: valid Python duplicate
    mock_dup = MagicMock()
    mock_dup.info = {
        "pid": 9999,
        "name": "python3",
        "cmdline": ["python3", script_path],
    }

    with patch("psutil.process_iter", return_value=[mock_bash, mock_dup]):
        duplicates = cleaner.detect_duplicates()

    assert len(duplicates) == 1


# MUTATION: detect_duplicates — `continue` vs `break` in except handler
@pytest.mark.unit
def test_detect_duplicates_continues_after_exception():
    """Kills mutmut_35: `continue` → `break`. Must find duplicate even after exception in prior proc."""
    script_path = "/path/to/test.py"
    cleaner = DuplicateProcessCleaner(script_name=script_path)

    # First: raises NoSuchProcess
    mock_error = MagicMock()
    type(mock_error).info = PropertyMock(side_effect=psutil.NoSuchProcess(1111))

    # Second: valid duplicate
    mock_dup = MagicMock()
    mock_dup.info = {
        "pid": 9999,
        "name": "python3",
        "cmdline": ["python3", script_path],
    }

    with patch("psutil.process_iter", return_value=[mock_error, mock_dup]):
        duplicates = cleaner.detect_duplicates()

    assert len(duplicates) == 1


# MUTATION: terminate_duplicates — `continue` vs `break` for NoSuchProcess/AccessDenied
@pytest.mark.unit
def test_terminate_duplicates_continues_after_nosuchprocess():
    """Kills mutmut_11: `continue` → `break`. Must try second proc after first vanishes."""
    cleaner = DuplicateProcessCleaner()

    mock_gone = MagicMock()
    mock_gone.terminate.side_effect = psutil.NoSuchProcess(1111)

    mock_ok = MagicMock()
    mock_ok.terminate = MagicMock()
    mock_ok.wait = MagicMock()

    with patch.object(cleaner, "detect_duplicates", return_value=[mock_gone, mock_ok]):
        with patch("time.sleep"):
            count = cleaner.terminate_duplicates()

    assert count == 1
    mock_ok.terminate.assert_called_once()


# MUTATION: terminate_duplicates — kill wait timeout value
@pytest.mark.unit
def test_terminate_duplicates_kill_wait_timeout_is_one():
    """Kills mutmut_12 and mutmut_13: timeout must be exactly 1 in kill().wait()."""
    cleaner = DuplicateProcessCleaner()

    mock_proc = MagicMock()
    mock_proc.terminate = MagicMock()
    mock_proc.wait = MagicMock(side_effect=[psutil.TimeoutExpired(1), None])
    mock_proc.kill = MagicMock()

    with patch.object(cleaner, "detect_duplicates", return_value=[mock_proc]):
        with patch("time.sleep"):
            cleaner.terminate_duplicates()

    # Second wait call (after kill) must use timeout=1
    assert mock_proc.wait.call_count == 2
    second_wait_kwargs = mock_proc.wait.call_args_list[1]
    assert second_wait_kwargs == ((), {"timeout": 1})


# MUTATION: terminate_duplicates — `count += 1` vs `count = 1` in kill branch
@pytest.mark.unit
def test_terminate_duplicates_kill_branch_increments_count():
    """Kills mutmut_14: `count += 1` → `count = 1`. With 2 procs in kill path, count must be 2."""
    cleaner = DuplicateProcessCleaner()

    def make_kill_proc() -> MagicMock:
        p = MagicMock()
        p.terminate = MagicMock()
        p.wait = MagicMock(side_effect=[psutil.TimeoutExpired(1), None])
        p.kill = MagicMock()
        return p

    procs = [make_kill_proc(), make_kill_proc()]

    with patch.object(cleaner, "detect_duplicates", return_value=procs):
        with patch("time.sleep"):
            count = cleaner.terminate_duplicates()

    assert count == 2


# MUTATION: terminate_duplicates — `continue` vs `break` in kill except handler
@pytest.mark.unit
def test_terminate_duplicates_continues_after_kill_failure():
    """Kills mutmut_17: `continue` → `break`. Must try second proc after first's kill fails."""
    cleaner = DuplicateProcessCleaner()

    # First proc: terminate times out, kill also fails
    mock_fail = MagicMock()
    mock_fail.terminate = MagicMock()
    mock_fail.wait = MagicMock(side_effect=psutil.TimeoutExpired(1))
    mock_fail.kill = MagicMock(side_effect=psutil.AccessDenied())

    # Second proc: terminates normally
    mock_ok = MagicMock()
    mock_ok.terminate = MagicMock()
    mock_ok.wait = MagicMock()

    with patch.object(cleaner, "detect_duplicates", return_value=[mock_fail, mock_ok]):
        with patch("time.sleep"):
            count = cleaner.terminate_duplicates()

    assert count == 1
    mock_ok.terminate.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
