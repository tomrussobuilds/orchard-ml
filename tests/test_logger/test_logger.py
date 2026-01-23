"""
Test Suite for Logging Management Module.

Tests logger configuration, file rotation, reconfiguration,
and singleton-like behavior.
"""

# Standard Imports
import logging
import os
from pathlib import Path
from unittest.mock import patch

# Third-Party Imports
import pytest

# Internal Imports
from orchard.core.logger import Logger
from orchard.core.paths import LOGGER_NAME


# LOGGER: INITIALIZATION
@pytest.mark.unit
def test_logger_init_console_only():
    """Test Logger initializes with console handler only when no log_dir."""
    logger = Logger(name="test_console", log_dir=None, log_to_file=False)

    assert logger.logger is not None
    assert logger.name == "test_console"
    assert logger.log_to_file is False
    assert len(logger.logger.handlers) == 1  # Console handler only


@pytest.mark.unit
def test_logger_init_with_file(tmp_path):
    """Test Logger initializes with console and file handlers when log_dir provided."""
    log_dir = tmp_path / "logs"

    logger = Logger(name="test_file", log_dir=log_dir, log_to_file=True)

    assert logger.log_to_file is True
    assert log_dir.exists()
    assert len(logger.logger.handlers) == 2  # Console + File


@pytest.mark.unit
def test_logger_default_name():
    """Test Logger uses LOGGER_NAME as default."""
    logger = Logger()

    assert logger.name == LOGGER_NAME


@pytest.mark.unit
def test_logger_default_level():
    """Test Logger defaults to INFO level."""
    logger = Logger(name="test_level")

    assert logger.logger.level == logging.INFO


# LOGGER: CONFIGURATION
@pytest.mark.unit
def test_logger_custom_level():
    """Test Logger accepts custom log level."""
    logger = Logger(name="test_debug", level=logging.DEBUG)

    assert logger.logger.level == logging.DEBUG


@pytest.mark.unit
def test_logger_formatter():
    """Test Logger applies correct formatter to handlers."""
    logger = Logger(name="test_format")

    handler = logger.logger.handlers[0]
    formatter = handler.formatter

    assert formatter is not None
    assert "%(asctime)s" in formatter._fmt
    assert "%(levelname)s" in formatter._fmt
    assert "%(message)s" in formatter._fmt


@pytest.mark.unit
def test_logger_propagate_false():
    """Test Logger sets propagate to False to prevent duplicate logs."""
    logger = Logger(name="test_propagate")

    assert logger.logger.propagate is False


# LOGGER: FILE HANDLING
@pytest.mark.unit
def test_logger_creates_log_directory(tmp_path):
    """Test Logger creates log directory if it doesn't exist."""
    log_dir = tmp_path / "new_logs"
    assert not log_dir.exists()

    Logger(name="test_mkdir", log_dir=log_dir, log_to_file=True)

    assert log_dir.exists()
    assert log_dir.is_dir()


@pytest.mark.unit
def test_logger_log_file_naming(tmp_path):
    """Test Logger creates log file with correct naming pattern."""
    log_dir = tmp_path / "logs"

    logger = Logger(name="test_naming", log_dir=log_dir).get_logger()
    logger.info("test message")

    log_files = list(log_dir.glob("test_naming_*.log"))
    assert len(log_files) == 1, f"Expected 1 log file, found {len(log_files)}"
    assert log_files[0].name.startswith("test_naming_")
    assert log_files[0].suffix == ".log"


@pytest.mark.unit
def test_logger_rotating_file_handler(tmp_path):
    """Test Logger uses RotatingFileHandler with correct settings."""
    log_dir = tmp_path / "logs"
    max_bytes = 1024
    backup_count = 3

    logger = Logger(
        name="test_rotate",
        log_dir=log_dir,
        log_to_file=True,
        max_bytes=max_bytes,
        backup_count=backup_count,
    )

    # Find file handler
    file_handler = None
    for handler in logger.logger.handlers:
        if hasattr(handler, "maxBytes"):
            file_handler = handler
            break

    assert file_handler is not None
    assert file_handler.maxBytes == max_bytes
    assert file_handler.backupCount == backup_count


# LOGGER: RECONFIGURATION
@pytest.mark.unit
def test_logger_reconfiguration_removes_old_handlers(tmp_path):
    """Test Logger removes old handlers when reconfigured."""
    log_dir1 = tmp_path / "logs1"
    log_dir2 = tmp_path / "logs2"

    # First configuration
    logger1 = Logger(name="test_reconfig", log_dir=log_dir1, log_to_file=True)
    initial_handler_count = len(logger1.logger.handlers)

    # Reconfigure with new log_dir
    logger2 = Logger(name="test_reconfig", log_dir=log_dir2, log_to_file=True)

    # Should have same number of handlers (old ones removed)
    assert len(logger2.logger.handlers) == initial_handler_count


@pytest.mark.unit
def test_logger_singleton_behavior():
    """Test Logger maintains singleton-like behavior per name."""
    logger1 = Logger(name="test_singleton")
    logger2 = Logger(name="test_singleton")

    # Should return the same logging.Logger instance
    assert logger1.logger is logger2.logger


# LOGGER: CLASS METHODS
@pytest.mark.unit
def test_get_logger_returns_logger_instance():
    """Test get_logger() returns logging.Logger instance."""
    logger = Logger(name="test_get")

    log_instance = logger.get_logger()

    assert isinstance(log_instance, logging.Logger)
    assert log_instance.name == "test_get"


@pytest.mark.unit
def test_get_log_file_returns_path(tmp_path):
    """Test get_log_file() returns active log file path."""
    log_dir = tmp_path / "logs"

    Logger(name="test_get_file", log_dir=log_dir, log_to_file=True)

    log_file = Logger.get_log_file()

    assert log_file is not None
    assert isinstance(log_file, Path)
    assert log_file.exists()


@pytest.mark.unit
def test_get_log_file_none_when_no_file():
    """Test get_log_file() returns None when no file logging."""
    # Reset class state
    Logger._active_log_file = None

    Logger(name="test_no_file", log_dir=None, log_to_file=False)

    log_file = Logger.get_log_file()

    assert log_file is None


@pytest.mark.unit
def test_setup_class_method(tmp_path):
    """Test setup() class method configures logger correctly."""
    log_dir = tmp_path / "logs"

    logger = Logger.setup(name="test_setup", log_dir=log_dir, level="DEBUG")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_setup"
    assert logger.level == logging.DEBUG


@pytest.mark.unit
def test_setup_level_string_mapping():
    """Test setup() correctly maps level strings to logging constants."""
    test_cases = [
        ("INFO", logging.INFO),
        ("DEBUG", logging.DEBUG),
        ("WARNING", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("CRITICAL", logging.CRITICAL),
    ]

    for level_str, expected_level in test_cases:
        logger = Logger.setup(name=f"test_{level_str.lower()}", level=level_str)
        assert logger.level == expected_level


@pytest.mark.unit
def test_setup_invalid_level_defaults_to_info():
    """Test setup() defaults to INFO for invalid level strings."""
    logger = Logger.setup(name="test_invalid", level="INVALID")

    assert logger.level == logging.INFO


@pytest.mark.unit
@patch.dict(os.environ, {"DEBUG": "1"})
def test_setup_debug_env_var():
    """Test setup() uses DEBUG level when DEBUG=1 environment variable set."""
    logger = Logger.setup(name="test_debug_env", level="INFO")

    assert logger.level == logging.DEBUG


# LOGGER: LOGGING FUNCTIONALITY
@pytest.mark.unit
def test_logger_can_log_messages(tmp_path):
    """Test Logger can successfully log messages."""
    log_dir = tmp_path / "logs"

    logger = Logger(name="test_logging", log_dir=log_dir, log_to_file=True)

    # Log a test message
    logger.logger.info("Test message")

    # Verify file was created and contains message
    log_files = list(log_dir.glob("test_logging_*.log"))
    assert len(log_files) == 1

    log_content = log_files[0].read_text()
    assert "Test message" in log_content


@pytest.mark.unit
def test_logger_handles_unicode(tmp_path):
    """Test Logger handles unicode characters correctly."""
    log_dir = tmp_path / "logs"

    logger = Logger(name="test_unicode", log_dir=log_dir, log_to_file=True)

    # Log unicode message
    logger.logger.info("Test emoji: 🚀 ✓ ⚠")

    # Verify unicode is preserved
    log_files = list(log_dir.glob("test_unicode_*.log"))
    log_content = log_files[0].read_text(encoding="utf-8")
    assert "🚀" in log_content


# LOGGER: EDGE CASES
@pytest.mark.unit
def test_logger_handles_permission_error(tmp_path):
    """Test Logger handles permission errors gracefully."""
    log_dir = tmp_path / "readonly"
    log_dir.mkdir()
    log_dir.chmod(0o444)  # Read-only

    try:
        # Should not crash
        logger = Logger(name="test_perm", log_dir=log_dir, log_to_file=True)
        assert logger is not None
    except PermissionError:
        # Expected on some systems
        pass
    finally:
        log_dir.chmod(0o755)  # Restore permissions


@pytest.mark.unit
def test_logger_multiple_names_independent():
    """Test loggers with different names are independent."""
    logger1 = Logger(name="logger_a")
    logger2 = Logger(name="logger_b")

    assert logger1.logger.name != logger2.logger.name
    assert logger1.logger is not logger2.logger


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
