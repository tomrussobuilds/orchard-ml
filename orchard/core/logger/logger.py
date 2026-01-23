"""
Logging Management Module

Handles centralized logging configuration. Supports dynamic reconfiguration
to switch from console-only logging to file-based logging once experiment
directories are initialized.
"""

# Standard Imports
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Final, Optional

# Internal Imports
from ..paths import LOGGER_NAME


# LOGGER CLASS
class Logger:
    """
    Manages centralized logging configuration with singleton-like behavior.
    """

    _configured_names: Final[Dict[str, bool]] = {}
    _active_log_file: Optional[Path] = None

    def __init__(
        self,
        name: str = LOGGER_NAME,
        log_dir: Optional[Path] = None,
        log_to_file: bool = True,
        level: int = logging.INFO,
        max_bytes: int = 5 * 1024 * 1024,
        backup_count: int = 5,
    ):
        self.name = name
        self.log_dir = log_dir
        self.log_to_file = log_to_file and (log_dir is not None)
        self.level = level
        self.max_bytes = max_bytes
        self.backup_count = backup_count

        self.logger = logging.getLogger(name)

        if name not in Logger._configured_names or log_dir is not None:
            self._setup_logger()
            Logger._configured_names[name] = True

    def _setup_logger(self) -> None:
        """
        Configures log handlers: Console always, File only if log_dir is provided.
        """
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
        self.logger.setLevel(self.level)
        self.logger.propagate = False

        # Clean up existing handlers to prevent duplicates during reconfiguration
        if self.logger.hasHandlers():
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)

        # 1. Console Handler (Standard Output)
        console_h = logging.StreamHandler(sys.stdout)
        console_h.setFormatter(formatter)
        self.logger.addHandler(console_h)

        # 2. Rotating File Handler (Activated when log_dir is known)
        if self.log_to_file and self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = self.log_dir / f"{self.name}_{timestamp}.log"

            file_h = RotatingFileHandler(
                filename, maxBytes=self.max_bytes, backupCount=self.backup_count, encoding="utf-8"
            )
            file_h.setFormatter(formatter)
            self.logger.addHandler(file_h)

            Logger._active_log_file = filename

    def get_logger(self) -> logging.Logger:
        """Returns the configured logging.Logger instance."""
        return self.logger

    @classmethod
    def get_log_file(cls) -> Optional[Path]:
        """Returns the current active log file path for auditing."""
        return cls._active_log_file

    @classmethod
    def setup(
        cls, name: str, log_dir: Optional[Path] = None, level: str = "INFO", **kwargs
    ) -> logging.Logger:
        """
        Main entry point for configuring the logger, called by RootOrchestrator.
        Bridges semantic LogLevel strings to Python logging constants.
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
