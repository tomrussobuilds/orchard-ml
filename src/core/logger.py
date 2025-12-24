"""
Logging Management Module

Handles centralized logging configuration. Supports dynamic reconfiguration 
to switch from console-only logging to file-based logging once experiment 
directories are initialized.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import logging
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional, Final
from logging.handlers import RotatingFileHandler

# =========================================================================== #
#                                LOGGER CLASS                                  #
# =========================================================================== #

class Logger:
    """
    Manages centralized logging configuration with singleton-like behavior.
    """
    _configured_names: Final[Dict[str, bool]] = {}
    _active_log_file: Optional[Path] = None

    def __init__(
        self,
        name: str = "medmnist_pipeline",
        log_dir: Optional[Path] = None,
        log_to_file: bool = True,
        level: int = logging.INFO,
        max_bytes: int = 5 * 1024 * 1024,
        backup_count: int = 5,
    ):
        self.name = name
        self.log_dir = log_dir
        self.log_to_file = log_to_file and (log_dir is not None)
        self.level = logging.DEBUG if os.getenv("DEBUG") == "1" else level
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        
        # Retrieve the system logger instance
        self.logger = logging.getLogger(name)
        
        # Configure if it's the first time or if a new log directory is provided
        # This allows the "re-configuration" once RootOrchestrator creates the run folder
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

        # Clean up existing handlers to prevent duplicates (Crucial for reconfiguration)
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
            
            # File name with timestamp for uniqueness
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = self.log_dir / f"{self.name}_{timestamp}.log"

            file_h = RotatingFileHandler(
                filename, 
                maxBytes=self.max_bytes, 
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            file_h.setFormatter(formatter)
            self.logger.addHandler(file_h)
            
            # Update class-level reference for global tracking
            Logger._active_log_file = filename

    def get_logger(self) -> logging.Logger:
        """Returns the configured logging.Logger instance."""
        return self.logger
    
    @classmethod
    def get_log_file(cls) -> Optional[Path]:
        """Returns the current active log file path."""
        return cls._active_log_file
    
    @classmethod
    def setup(cls, name: str, **kwargs) -> logging.Logger:
        """
        Main entry point for configuring the logger, typically called from RootOrchestrator.
        """
        return cls(name=name, **kwargs).get_logger()

# =========================================================================== #
#                                GLOBAL INSTANCE                              #
# =========================================================================== #

# Initial safe instance for immediate imports (writes only to console initially).
# Later, the Orchestrator will call Logger.setup() with a real path to enable file logging.
logger: Final[logging.Logger] = Logger().get_logger()