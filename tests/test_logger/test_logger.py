"""
Test Suite for Logging Management Module.

Tests logger configuration, file rotation, reconfiguration,
and singleton-like behavior.
"""

import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from orchard.core.logger import Logger
from orchard.core.paths import LOGGER_NAME


# LOGGER: INITIALIZATION
@pytest.mark.unit
def test_logger_init_console_only():
    """Test Logger initializes with console handler only when no log_dir."""
    logger = Logger(name="test_console", log_dir=None, log_to_file=False)

    assert logger._log is not None
    assert logger.name == "test_console"
    assert logger.log_to_file is False
    assert len(logger._log.handlers) == 1


@pytest.mark.unit
def test_logger_init_with_file(tmp_path):
    """Test Logger initializes with console and file handlers when log_dir provided."""
    log_dir = tmp_path / "logs"

    logger = Logger(name="test_file", log_dir=log_dir, log_to_file=True)

    assert logger.log_to_file is True
    assert log_dir.exists()
    assert len(logger._log.handlers) == 2


@pytest.mark.unit
def test_logger_default_name():
    """Test Logger uses LOGGER_NAME as default."""
    logger = Logger()

    assert logger.name == LOGGER_NAME


@pytest.mark.unit
def test_logger_default_level():
    """Test Logger defaults to INFO level."""
    logger = Logger(name="test_level")

    assert logger._log.level == logging.INFO


# LOGGER: CONFIGURATION
@pytest.mark.unit
def test_logger_custom_level():
    """Test Logger accepts custom log level."""
    logger = Logger(name="test_debug", level=logging.DEBUG)

    assert logger._log.level == logging.DEBUG


@pytest.mark.unit
def test_logger_formatter():
    """Test Logger applies correct formatter to handlers."""
    logger = Logger(name="test_format")

    handler = logger._log.handlers[0]
    formatter = handler.formatter

    assert formatter is not None
    assert "%(asctime)s" in formatter._fmt
    assert "%(levelname)s" in formatter._fmt
    assert "%(message)s" in formatter._fmt


@pytest.mark.unit
def test_logger_propagate_false():
    """Test Logger sets propagate to False to prevent duplicate logs."""
    logger = Logger(name="test_propagate")

    assert logger._log.propagate is False


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

    file_handler = None
    for handler in logger._log.handlers:
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

    logger1 = Logger(name="test_reconfig", log_dir=log_dir1, log_to_file=True)
    initial_handler_count = len(logger1._log.handlers)

    logger2 = Logger(name="test_reconfig", log_dir=log_dir2, log_to_file=True)

    assert len(logger2._log.handlers) == initial_handler_count


@pytest.mark.unit
def test_logger_singleton_behavior():
    """Test Logger maintains singleton-like behavior per name."""
    logger1 = Logger(name="test_singleton")
    logger2 = Logger(name="test_singleton")

    assert logger1._log is logger2._log


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

    logger._log.info("Test message")

    log_files = list(log_dir.glob("test_logging_*.log"))
    assert len(log_files) == 1

    log_content = log_files[0].read_text()
    assert "Test message" in log_content


@pytest.mark.unit
def test_logger_handles_unicode(tmp_path):
    """Test Logger handles unicode characters correctly."""
    log_dir = tmp_path / "logs"

    logger = Logger(name="test_unicode", log_dir=log_dir, log_to_file=True)

    logger._log.info("Test emoji: 🚀 ✓ ⚠")

    log_files = list(log_dir.glob("test_unicode_*.log"))
    log_content = log_files[0].read_text(encoding="utf-8")
    assert "🚀" in log_content


# LOGGER: EDGE CASES
@pytest.mark.unit
def test_logger_handles_permission_error(tmp_path):
    """Test Logger handles permission errors gracefully."""
    log_dir = tmp_path / "readonly"
    log_dir.mkdir()
    log_dir.chmod(0o444)

    try:
        logger = Logger(name="test_perm", log_dir=log_dir, log_to_file=True)
        assert logger is not None
    except PermissionError:
        pass
    finally:
        log_dir.chmod(0o755)


@pytest.mark.unit
def test_logger_multiple_names_independent():
    """Test loggers with different names are independent."""
    logger1 = Logger(name="logger_a")
    logger2 = Logger(name="logger_b")

    assert logger1._log.name != logger2._log.name
    assert logger1._log is not logger2._log


# COLOR FORMATTER
@pytest.mark.unit
def test_color_formatter_success_symbol():
    """Test ColorFormatter applies green only to the message, not the prefix."""
    from orchard.core.logger.logger import ColorFormatter
    from orchard.core.logger.styles import LogStyle

    formatter = ColorFormatter("%(levelname)s - %(message)s")
    msg = "  ✓ Test Accuracy  : 92.34%"
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg=msg,
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    assert LogStyle.GREEN in output
    assert LogStyle.RESET in output
    # Color must start right at the message, prefix before it stays clean
    idx = output.find(LogStyle.GREEN)
    prefix = output[:idx]
    assert "\033[" not in prefix
    assert output.endswith(f"{LogStyle.GREEN}{msg}{LogStyle.RESET}")


@pytest.mark.unit
def test_color_formatter_new_best_model():
    """Test ColorFormatter applies green to 'New best model' lines."""
    from orchard.core.logger.logger import ColorFormatter
    from orchard.core.logger.styles import LogStyle

    formatter = ColorFormatter("%(levelname)s - %(message)s")
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="New best model! Val AUC: 0.9998 ↑ Checkpoint saved.",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    assert LogStyle.GREEN in output


@pytest.mark.unit
def test_color_formatter_separator_dim():
    """Test ColorFormatter dims separator lines."""
    from orchard.core.logger.logger import ColorFormatter
    from orchard.core.logger.styles import LogStyle

    formatter = ColorFormatter("%(levelname)s - %(message)s")
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="━" * 80,
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    assert LogStyle.DIM in output


@pytest.mark.unit
def test_color_formatter_header_bold_magenta():
    """Test ColorFormatter applies bold magenta only to the message, not the prefix."""
    from orchard.core.logger.logger import ColorFormatter
    from orchard.core.logger.styles import LogStyle

    formatter = ColorFormatter("%(levelname)s - %(message)s")
    msg = "ENVIRONMENT INITIALIZATION"
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg=msg,
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    assert LogStyle.BOLD in output
    assert LogStyle.MAGENTA in output
    assert LogStyle.RESET in output
    # Color must start right at the message, prefix stays clean
    idx = output.find(LogStyle.BOLD)
    prefix = output[:idx]
    assert "\033[" not in prefix
    assert output.endswith(f"{LogStyle.BOLD}{LogStyle.MAGENTA}{msg}{LogStyle.RESET}")


@pytest.mark.unit
def test_color_formatter_warning_yellow():
    """Test ColorFormatter applies yellow only to the message, not the timestamp."""
    from orchard.core.logger.logger import ColorFormatter
    from orchard.core.logger.styles import LogStyle

    formatter = ColorFormatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    msg = "Early stopping triggered at epoch 42."
    record = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname="",
        lineno=0,
        msg=msg,
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    assert LogStyle.YELLOW in output
    # Yellow must wrap only the message, not the timestamp
    idx = output.find(msg)
    assert idx > 0
    # The color code appears right before the message
    assert output[idx - len(LogStyle.YELLOW) : idx] == LogStyle.YELLOW
    assert output.endswith(f"{msg}{LogStyle.RESET}")


@pytest.mark.unit
def test_color_formatter_early_stopping_green():
    """Test ColorFormatter applies green to EARLY STOPPING banner."""
    from orchard.core.logger.logger import ColorFormatter
    from orchard.core.logger.styles import LogStyle

    formatter = ColorFormatter("%(levelname)s - %(message)s")
    msg = "EARLY STOPPING: Target performance achieved!"
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg=msg,
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    assert LogStyle.GREEN in output
    assert output.endswith(f"{msg}{LogStyle.RESET}")


@pytest.mark.unit
def test_color_formatter_plain_info():
    """Test ColorFormatter does not add color to plain INFO messages."""
    from orchard.core.logger.logger import ColorFormatter
    from orchard.core.logger.styles import LogStyle

    formatter = ColorFormatter("%(levelname)s - %(message)s")
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Loss: [T: 0.2131 | V: 0.1196]",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    assert LogStyle.GREEN not in output
    assert LogStyle.DIM not in output
    assert LogStyle.CYAN not in output


@pytest.mark.unit
def test_color_formatter_error_red():
    """Test ColorFormatter applies red to ERROR level prefix."""
    from orchard.core.logger.logger import ColorFormatter
    from orchard.core.logger.styles import LogStyle

    formatter = ColorFormatter("%(levelname)s - %(message)s")
    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname="",
        lineno=0,
        msg="Pipeline failed",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    assert LogStyle.RED in output


@pytest.mark.unit
def test_console_uses_color_formatter_on_tty():
    """Test console handler uses ColorFormatter when stdout is a TTY."""
    from orchard.core.logger.logger import ColorFormatter

    with patch("sys.stdout") as mock_stdout:
        mock_stdout.isatty.return_value = True
        logger_obj = Logger(name="test_tty_color", log_dir=None, log_to_file=False)
        console_handler = logger_obj._log.handlers[0]
        assert isinstance(console_handler.formatter, ColorFormatter)


@pytest.mark.unit
def test_console_uses_plain_formatter_on_pipe():
    """Test console handler uses plain Formatter when stdout is not a TTY."""
    from orchard.core.logger.logger import ColorFormatter

    with patch("sys.stdout") as mock_stdout:
        mock_stdout.isatty.return_value = False
        logger_obj = Logger(name="test_pipe_plain", log_dir=None, log_to_file=False)
        console_handler = logger_obj._log.handlers[0]
        assert not isinstance(console_handler.formatter, ColorFormatter)


@pytest.mark.unit
def test_file_handler_uses_plain_formatter(tmp_path):
    """Test file handler always uses plain Formatter, never ColorFormatter."""
    from orchard.core.logger.logger import ColorFormatter

    log_dir = tmp_path / "logs"
    logger_obj = Logger(name="test_file_plain", log_dir=log_dir, log_to_file=True)
    file_handler = [h for h in logger_obj._log.handlers if hasattr(h, "maxBytes")][0]
    assert not isinstance(file_handler.formatter, ColorFormatter)


@pytest.mark.unit
def test_file_output_has_no_ansi_codes(tmp_path):
    """Test log file output contains no ANSI escape codes."""
    log_dir = tmp_path / "logs"
    logger_obj = Logger(name="test_no_ansi", log_dir=log_dir, log_to_file=True)
    log = logger_obj.get_logger()

    log.info("✓ Success line")
    log.info("━" * 80)
    log.warning("Early stopping triggered")

    log_files = list(log_dir.glob("test_no_ansi_*.log"))
    content = log_files[0].read_text()
    assert "\033[" not in content


@pytest.mark.unit
def test_color_formatter_subtitle_bold_magenta():
    """Test ColorFormatter applies bold magenta to [Subtitle] tags."""
    from orchard.core.logger.logger import ColorFormatter
    from orchard.core.logger.styles import LogStyle

    formatter = ColorFormatter("%(levelname)s - %(message)s")
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="[HARDWARE]",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    assert f"{LogStyle.BOLD}{LogStyle.MAGENTA}[HARDWARE]{LogStyle.RESET}" in output
    # Prefix must stay clean
    idx = output.find(LogStyle.BOLD)
    assert "\033[" not in output[:idx]


@pytest.mark.unit
def test_color_formatter_subtitle_mixed_case():
    """Test ColorFormatter colors [Export Settings] style subtitles."""
    from orchard.core.logger.logger import ColorFormatter
    from orchard.core.logger.styles import LogStyle

    formatter = ColorFormatter("%(levelname)s - %(message)s")
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="  [Export Settings]",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    assert f"{LogStyle.BOLD}{LogStyle.MAGENTA}[Export Settings]{LogStyle.RESET}" in output


@pytest.mark.unit
def test_color_formatter_subtitle_ignores_data_brackets():
    """Test ColorFormatter does NOT color data brackets like [T: 0.2131]."""
    from orchard.core.logger.logger import ColorFormatter
    from orchard.core.logger.styles import LogStyle

    formatter = ColorFormatter("%(levelname)s - %(message)s")
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Loss: [T: 0.2131 | V: 0.1196]",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    assert LogStyle.MAGENTA not in output


@pytest.mark.unit
def test_color_message_only_fallback_when_msg_not_found():
    """Test _color_message_only returns formatted unchanged when msg is not found."""
    from orchard.core.logger.logger import ColorFormatter
    from orchard.core.logger.styles import LogStyle

    formatter = ColorFormatter("%(levelname)s - %(message)s")
    result = formatter._color_message_only("INFO - hello", "nonexistent", LogStyle.GREEN)
    assert result == "INFO - hello"


@pytest.mark.unit
def test_color_subtitles_fallback_when_msg_not_found():
    """Test _color_subtitles returns formatted unchanged when msg is not found."""
    from orchard.core.logger.logger import ColorFormatter

    formatter = ColorFormatter("%(levelname)s - %(message)s")
    result = formatter._color_subtitles("INFO - hello", "nonexistent")
    assert result == "INFO - hello"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
