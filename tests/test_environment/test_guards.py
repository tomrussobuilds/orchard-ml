"""
Test Suite for Process & Resource Guarding Utilities.

Tests filesystem locking, duplicate process detection,
and process termination utilities.
"""

# =========================================================================== #
#                         Standard Imports                                    #
# =========================================================================== #
import logging
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# =========================================================================== #
#                         Third-Party Imports                                 #
# =========================================================================== #
import psutil
import pytest

# =========================================================================== #
#                         Internal Imports                                    #
# =========================================================================== #
from orchard.core.environment import (
    DuplicateProcessCleaner,
    ensure_single_instance,
    release_single_instance,
)

# =========================================================================== #
#                    SINGLE INSTANCE LOCKING                                  #
# =========================================================================== #


@pytest.mark.unit
@patch("platform.system", return_value="Linux")
@patch("orchard.core.environment.guards.HAS_FCNTL", True)
def test_ensure_single_instance_success(mock_platform, tmp_path):
    """Test ensure_single_instance acquires lock successfully."""
    lock_file = tmp_path / "test.lock"
    logger = logging.getLogger("test")

    with patch("fcntl.flock") as mock_flock:
        ensure_single_instance(lock_file, logger)

        # Verify flock was called with correct flags
        mock_flock.assert_called_once()
        # Lock file parent should be created
        assert lock_file.parent.exists()


@pytest.mark.unit
@patch("platform.system", return_value="Linux")
@patch("orchard.core.environment.guards.HAS_FCNTL", True)
def test_ensure_single_instance_already_locked(mock_platform, tmp_path):
    """Test ensure_single_instance exits when lock already held."""
    lock_file = tmp_path / "test.lock"
    logger = logging.getLogger("test")

    with patch("fcntl.flock", side_effect=BlockingIOError):
        with pytest.raises(SystemExit) as exc_info:
            ensure_single_instance(lock_file, logger)

        assert exc_info.value.code == 1


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

    # Should not raise any errors on Windows
    ensure_single_instance(lock_file, logger)


@pytest.mark.unit
@patch("platform.system", return_value="Darwin")
@patch("orchard.core.environment.guards.HAS_FCNTL", True)
def test_ensure_single_instance_macos(mock_platform, tmp_path):
    """Test ensure_single_instance works on macOS."""
    lock_file = tmp_path / "test.lock"
    logger = logging.getLogger("test")

    with patch("fcntl.flock") as mock_flock:
        ensure_single_instance(lock_file, logger)

        # Should work on Darwin (macOS)
        mock_flock.assert_called_once()


@pytest.mark.unit
@patch("platform.system", return_value="Linux")
@patch("orchard.core.environment.guards.HAS_FCNTL", False)
def test_ensure_single_instance_no_fcntl(mock_platform, tmp_path):
    """Test ensure_single_instance when fcntl not available."""
    lock_file = tmp_path / "test.lock"
    logger = logging.getLogger("test")

    # Should not crash when fcntl unavailable
    ensure_single_instance(lock_file, logger)


# =========================================================================== #
#                    LOCK RELEASE                                             #
# =========================================================================== #


@pytest.mark.unit
@patch("orchard.core.environment.guards.HAS_FCNTL", True)
def test_release_single_instance_with_lock(tmp_path):
    """Test release_single_instance releases lock and removes file."""
    lock_file = tmp_path / "test.lock"
    lock_file.touch()

    # Mock the global lock file descriptor
    mock_fd = MagicMock()
    with patch("orchard.core.environment.guards._lock_fd", mock_fd):
        with patch("fcntl.flock") as mock_flock:
            release_single_instance(lock_file)

            # Verify unlock and close were called
            mock_flock.assert_called_once()
            mock_fd.close.assert_called_once()

    # File should be removed
    assert not lock_file.exists()


@pytest.mark.unit
def test_release_single_instance_no_lock(tmp_path):
    """Test release_single_instance when no lock is held."""
    lock_file = tmp_path / "test.lock"
    lock_file.touch()

    with patch("orchard.core.environment.guards._lock_fd", None):
        release_single_instance(lock_file)

    # File should still be removed
    assert not lock_file.exists()


@pytest.mark.unit
def test_release_single_instance_file_not_exists(tmp_path):
    """Test release_single_instance when lock file doesn't exist."""
    lock_file = tmp_path / "nonexistent.lock"

    with patch("orchard.core.environment.guards._lock_fd", None):
        # Should not raise errors
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
                # Should not raise errors
                release_single_instance(lock_file)


# =========================================================================== #
#                    DUPLICATE PROCESS CLEANER: INITIALIZATION                #
# =========================================================================== #


@pytest.mark.unit
def test_duplicate_process_cleaner_init_default():
    """Test DuplicateProcessCleaner initializes with default script name."""
    cleaner = DuplicateProcessCleaner()

    assert cleaner.script_path == os.path.realpath(sys.argv[0])
    assert cleaner.current_pid == os.getpid()


@pytest.mark.unit
def test_duplicate_process_cleaner_init_custom_script():
    """Test DuplicateProcessCleaner initializes with custom script name."""
    custom_script = "/path/to/custom_script.py"

    with patch("os.path.realpath", return_value=custom_script):
        cleaner = DuplicateProcessCleaner(script_name=custom_script)

    assert cleaner.script_path == custom_script


# =========================================================================== #
#                    DUPLICATE PROCESS CLEANER: DETECTION                     #
# =========================================================================== #


@pytest.mark.unit
def test_detect_duplicates_no_duplicates():
    """Test detect_duplicates returns empty list when no duplicates."""
    cleaner = DuplicateProcessCleaner()

    # Mock process iteration to return no Python processes
    mock_procs = []
    with patch("psutil.process_iter", return_value=mock_procs):
        duplicates = cleaner.detect_duplicates()

    assert duplicates == []


@pytest.mark.unit
def test_detect_duplicates_skips_self():
    """Test detect_duplicates skips current process."""
    cleaner = DuplicateProcessCleaner(script_name="test.py")

    # Mock current process
    mock_proc = MagicMock()
    mock_proc.info = {
        "pid": cleaner.current_pid,
        "name": "python",
        "cmdline": ["python", os.path.realpath("test.py")],
    }

    with patch("psutil.process_iter", return_value=[mock_proc]):
        duplicates = cleaner.detect_duplicates()

    # Should skip own PID
    assert duplicates == []


@pytest.mark.unit
def test_detect_duplicates_skips_non_python():
    """Test detect_duplicates skips non-Python processes."""
    cleaner = DuplicateProcessCleaner(script_name="test.py")

    # Mock non-Python process
    mock_proc = MagicMock()
    mock_proc.info = {
        "pid": 9999,
        "name": "bash",
        "cmdline": ["bash", "script.sh"],
    }

    with patch("psutil.process_iter", return_value=[mock_proc]):
        duplicates = cleaner.detect_duplicates()

    # Should skip non-Python processes
    assert duplicates == []


@pytest.mark.unit
def test_detect_duplicates_finds_duplicate():
    """Test detect_duplicates finds matching Python process."""
    script_path = "/path/to/test.py"
    cleaner = DuplicateProcessCleaner(script_name=script_path)

    # Mock duplicate process
    mock_proc = MagicMock()
    mock_proc.info = {
        "pid": 9999,
        "name": "python3",
        "cmdline": ["python3", script_path],
    }

    with patch("psutil.process_iter", return_value=[mock_proc]):
        with patch("os.path.realpath", side_effect=lambda x: x):
            duplicates = cleaner.detect_duplicates()

    assert len(duplicates) == 1
    assert duplicates[0] == mock_proc


@pytest.mark.unit
def test_detect_duplicates_handles_exceptions():
    """Test detect_duplicates handles psutil exceptions gracefully."""
    cleaner = DuplicateProcessCleaner()

    # Mock a process that works
    mock_proc1 = MagicMock()
    mock_proc1.info = {"pid": 1000, "name": "python", "cmdline": ["python"]}

    # Mock a process that raises exception when accessing info
    mock_proc2 = MagicMock()
    mock_proc2.info = MagicMock(side_effect=psutil.NoSuchProcess(9999))

    with patch("psutil.process_iter", return_value=[mock_proc1, mock_proc2]):
        # Should not crash and skip the problematic process
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

    # Should skip processes with no cmdline
    assert duplicates == []


# =========================================================================== #
#                    DUPLICATE PROCESS CLEANER: TERMINATION                   #
# =========================================================================== #


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
    mock_proc.wait.assert_called_once()


@pytest.mark.unit
def test_terminate_duplicates_multiple():
    """Test terminate_duplicates handles multiple processes."""
    cleaner = DuplicateProcessCleaner()

    mock_procs = [MagicMock() for _ in range(3)]
    for proc in mock_procs:
        proc.terminate = MagicMock()
        proc.wait = MagicMock()

    with patch.object(cleaner, "detect_duplicates", return_value=mock_procs):
        with patch("time.sleep"):  # Skip the cooldown sleep
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

    # Should handle exception and not count as terminated
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
def test_terminate_duplicates_with_logger():
    """Test terminate_duplicates logs termination when logger provided."""
    cleaner = DuplicateProcessCleaner()
    logger = MagicMock()

    mock_proc = MagicMock()

    with patch.object(cleaner, "detect_duplicates", return_value=[mock_proc]):
        with patch("time.sleep"):
            count = cleaner.terminate_duplicates(logger=logger)

    assert count == 1
    # Verify logger was called
    logger.info.assert_called_once()


# =========================================================================== #
#                    INTEGRATION TESTS                                        #
# =========================================================================== #


@pytest.mark.integration
@patch("platform.system", return_value="Linux")
@patch("orchard.core.environment.guards.HAS_FCNTL", True)
def test_full_guard_workflow(mock_platform, tmp_path):
    """Test complete guard workflow: lock, detect, release."""
    lock_file = tmp_path / "test.lock"
    logger = logging.getLogger("test")

    # Acquire lock
    with patch("fcntl.flock"):
        ensure_single_instance(lock_file, logger)

    # Check for duplicates
    cleaner = DuplicateProcessCleaner()
    duplicates = cleaner.detect_duplicates()

    assert isinstance(duplicates, list)

    # Release lock
    with patch("orchard.core.environment.guards._lock_fd", MagicMock()):
        with patch("fcntl.flock"):
            release_single_instance(lock_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
