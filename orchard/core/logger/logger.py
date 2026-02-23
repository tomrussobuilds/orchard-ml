"""
Logging Management Module

Handles centralized logging configuration with dynamic reconfiguration support.
Enables transition from console-only logging (bootstrap phase) to dual console+file
logging once experiment directories are provisioned by RootOrchestrator.

Key Features:
    - Singleton-like Behavior: Prevents duplicate logger configurations
    - Dynamic Reconfiguration: Switches from console-only to file-based logging
    - Rotating File Handler: Automatic log rotation with size limits
    - Thread-safe: Safe for concurrent access across modules
    - Timestamp-based Files: Unique log files per experiment session
"""

from __future__ import annotations

import logging
import os
import re
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Final

from ..paths import LOGGER_NAME
from .styles import LogStyle

# Separator characters used to detect decorative lines
_SEPARATOR_CHARS = {"━", "═", "─"}

# Matches subtitle tags like [Hardware], [OPTIMIZATION], [Export Settings]
# but NOT data brackets like [T: 0.2131 | V: 0.1196] or [!]
_SUBTITLE_RE = re.compile(r"\[([A-Za-z][A-Za-z ]*)\]")


class ColorFormatter(logging.Formatter):
    """Formatter that applies ANSI colors to console output.

    Colors are applied based on log level and message content:
        - WARNING/ERROR/CRITICAL: yellow/red level prefix
        - Lines with ✓ or 'New best model': green
        - Lines with 'EARLY STOPPING': green
        - Separator lines (━, ═, ─): dim
        - Centered UPPER CASE headers: bold magenta
        - Subtitle tags like [Hardware], [OPTIMIZATION]: bold magenta
    """

    _LEVEL_COLORS = {
        logging.WARNING: LogStyle.YELLOW,
        logging.ERROR: LogStyle.RED,
        logging.CRITICAL: LogStyle.RED + LogStyle.BOLD,
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with ANSI color codes.

        Colors are applied **only to the message text**; the timestamp and
        level prefix on the left remain uncolored.
        """
        formatted = super().format(record)
        msg = record.getMessage()

        # Color the level prefix for WARNING+
        level_color = self._LEVEL_COLORS.get(record.levelno)
        if level_color:
            formatted = formatted.replace(
                record.levelname,
                f"{level_color}{record.levelname}{LogStyle.RESET}",
                1,
            )

        # Content-based coloring for INFO messages
        if record.levelno == logging.INFO:
            stripped = msg.strip()

            # Separator lines → dim
            if stripped and all(c in _SEPARATOR_CHARS for c in stripped):
                return self._color_message_only(formatted, msg, LogStyle.DIM)

            # Centered headers (e.g. "ENVIRONMENT INITIALIZATION",
            # "TRAINING PIPELINE - RESNET_18") → bold magenta
            if (
                stripped == stripped.upper()
                and len(stripped) > 5
                and any(c.isalpha() for c in stripped)
            ):
                return self._color_message_only(formatted, msg, LogStyle.BOLD + LogStyle.MAGENTA)

            # Early stopping banner → green
            if "EARLY STOPPING" in stripped:
                return self._color_message_only(formatted, msg, LogStyle.GREEN)

            # Success lines (✓, New best model) → green
            if "✓" in msg or "New best model" in msg:
                return self._color_message_only(formatted, msg, LogStyle.GREEN)

        # Warning-level content coloring
        if record.levelno == logging.WARNING:
            return self._color_message_only(formatted, msg, LogStyle.YELLOW)

        # Subtitle tags [Text] → bold magenta (within message portion only)
        if _SUBTITLE_RE.search(msg):
            formatted = self._color_subtitles(formatted, msg)

        return formatted

    def _color_message_only(self, formatted: str, msg: str, color: str) -> str:
        """Apply *color* only to the message portion of *formatted*, leaving the prefix plain."""
        idx = formatted.find(msg)
        if idx == -1:
            return formatted
        prefix = formatted[:idx]
        return f"{prefix}{color}{formatted[idx:]}{LogStyle.RESET}"

    def _color_subtitles(self, formatted: str, msg: str) -> str:
        """Apply bold magenta to ``[Subtitle]`` tags in the message portion only."""
        idx = formatted.find(msg)
        if idx == -1:
            return formatted
        prefix = formatted[:idx]
        msg_part = formatted[idx:]
        colored_msg = _SUBTITLE_RE.sub(
            rf"{LogStyle.BOLD}{LogStyle.MAGENTA}\g<0>{LogStyle.RESET}",
            msg_part,
        )
        return prefix + colored_msg


# LOGGER CLASS
class Logger:
    """
    Manages centralized logging configuration with singleton-like behavior.

    Provides a unified logging interface for the entire framework with support for
    dynamic reconfiguration. Initially bootstraps with console-only output, then
    transitions to dual console+file logging when experiment directories become available.

    The logger implements pseudo-singleton semantics via class-level tracking (_configured_names)
    to prevent duplicate handler registration while allowing intentional reconfiguration
    when log directories are provided.

    Lifecycle:
        1. Bootstrap Phase: Console-only logging (no log_dir specified)
        2. Orchestration Phase: RootOrchestrator calls setup() with log_dir
        3. Reconfiguration: Existing handlers removed, file handler added
        4. Audit Trail: Log file path stored in _active_log_file for reference

    Class Attributes:
        _configured_names (dict[str, bool]): Tracks which logger names have been configured
        _active_log_file (Path | None): Current active log file path for auditing

    Attributes:
        name (str): Logger identifier (typically LOGGER_NAME constant)
        log_dir (Path | None): Directory for log file storage
        log_to_file (bool): Enable file logging (requires log_dir)
        level (int): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        max_bytes (int): Maximum log file size before rotation (default: 5MB)
        backup_count (int): Number of rotated log files to retain (default: 5)
        _log (logging.Logger): Underlying Python logger instance

    Example:
        >>> # Bootstrap phase (console-only)
        >>> logger = Logger().get_logger()
        >>> logger.info("Framework initializing...")

        >>> # Orchestration phase (add file logging)
        >>> logger = Logger.setup(
        ...     name=LOGGER_NAME,
        ...     log_dir=Path("./outputs/run_123/logs"),
        ...     level="INFO"
        ... )
        >>> logger.info("Logging to file now")

        >>> # Retrieve log file path
        >>> log_path = Logger.get_log_file()
        >>> print(f"Logs saved to: {log_path}")

    Notes:
        - Reconfiguration is idempotent: calling setup() multiple times is safe
        - All handlers are properly closed before reconfiguration
        - Log files use UTC timestamps for consistency across time zones
        - RotatingFileHandler prevents disk space exhaustion
    """

    _configured_names: Final[dict[str, bool]] = {}
    _active_log_file: Path | None = None

    def __init__(
        self,
        name: str = LOGGER_NAME,
        log_dir: Path | None = None,
        log_to_file: bool = True,
        level: int = logging.INFO,
        max_bytes: int = 5 * 1024 * 1024,
        backup_count: int = 5,
    ) -> None:
        """
        Initializes the Logger with specified configuration.

        Args:
            name: Logger identifier (default: LOGGER_NAME constant)
            log_dir: Directory for log file storage (None = console-only)
            log_to_file: Enable file logging if log_dir provided (default: True)
            level: Logging level as integer constant (default: logging.INFO)
            max_bytes: Maximum log file size before rotation in bytes (default: 5MB)
            backup_count: Number of rotated backup files to retain (default: 5)
        """
        self.name = name
        self.log_dir = log_dir
        self.log_to_file = log_to_file and (log_dir is not None)
        self.level = level
        self.max_bytes = max_bytes
        self.backup_count = backup_count

        self._log = logging.getLogger(name)

        if name not in Logger._configured_names or log_dir is not None:
            self._setup_logger()
            Logger._configured_names[name] = True

    def _setup_logger(self) -> None:
        """
        Configures log handlers: Console always, File only if log_dir is provided.

        Removes existing handlers before configuration to prevent duplicate output
        during reconfiguration. Creates a console handler for stdout and optionally
        adds a rotating file handler when log_dir is specified.

        Handler Configuration:
            - Console: Always enabled, outputs to sys.stdout
            - File: Enabled only when log_dir is provided, uses RotatingFileHandler
            - Format: "YYYY-MM-DD HH:MM:SS - LEVEL - message"
            - Rotation: Automatic when max_bytes threshold exceeded
        """
        fmt_str = "%(asctime)s - %(levelname)s - %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        plain_formatter = logging.Formatter(fmt_str, datefmt)

        self._log.setLevel(self.level)
        self._log.propagate = False

        # Clean up existing handlers to prevent duplicates during reconfiguration
        if self._log.hasHandlers():
            for handler in self._log.handlers[:]:
                handler.close()
                self._log.removeHandler(handler)

        # 1. Console Handler (Standard Output)
        console_h = logging.StreamHandler(sys.stdout)
        if sys.stdout.isatty():
            console_h.setFormatter(ColorFormatter(fmt_str, datefmt))
        else:
            console_h.setFormatter(plain_formatter)
        self._log.addHandler(console_h)

        # 2. Rotating File Handler (Activated when log_dir is known)
        if self.log_to_file and self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = self.log_dir / f"{self.name}_{timestamp}.log"

            file_h = RotatingFileHandler(
                filename, maxBytes=self.max_bytes, backupCount=self.backup_count, encoding="utf-8"
            )
            file_h.setFormatter(plain_formatter)
            self._log.addHandler(file_h)

            Logger._active_log_file = filename

    def get_logger(self) -> logging.Logger:
        """
        Returns the configured logging.Logger instance.

        Returns:
            The underlying Python logging.Logger instance with configured handlers
        """
        return self._log

    @classmethod
    def get_log_file(cls) -> Path | None:
        """
        Returns the current active log file path for auditing.

        Returns:
            Path to the active log file, or None if file logging is not enabled
        """
        return cls._active_log_file

    @classmethod
    def setup(
        cls, name: str, log_dir: Path | None = None, level: str = "INFO", **kwargs
    ) -> logging.Logger:
        """
        Main entry point for configuring the logger, called by RootOrchestrator.

        Bridges semantic LogLevel strings (INFO, DEBUG, WARNING) to Python logging
        constants. Provides convenient string-based level specification while internally
        using numeric logging constants.

        Args:
            name: Logger identifier (typically LOGGER_NAME constant)
            log_dir: Directory for log file storage (None = console-only mode)
            level: Logging level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            **kwargs (Any): Additional arguments passed to Logger constructor

        Returns:
            Configured logging.Logger instance ready for use

        Environment Variables:
            DEBUG: If set to "1", overrides level to DEBUG regardless of level parameter

        Example:
            >>> logger = Logger.setup(
            ...     name="OrchardML",
            ...     log_dir=Path("./outputs/run_123/logs"),
            ...     level="INFO"
            ... )
            >>> logger.info("Training started")
        """
        if os.getenv("DEBUG") == "1":
            numeric_level = logging.DEBUG
        else:
            numeric_level = getattr(logging, level.upper(), logging.INFO)

        return cls(name=name, log_dir=log_dir, level=numeric_level, **kwargs).get_logger()


# GLOBAL INSTANCE
# Initial bootstrap instance (Console-only).
# Level is set to INFO by default, overridden by setup() during orchestration.
logger: Final[logging.Logger] = Logger().get_logger()
