"""
Logging Management Module

This module provides a centralized logger configuration with support for
console output and rotating file storage. It ensures a consistent logging
format across the entire pipeline.
"""
# =========================================================================== #
#                                Standard Imports
# =========================================================================== #
import logging
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional
from logging.handlers import RotatingFileHandler

# =========================================================================== #
#                                Internal Imports
# =========================================================================== #
from .constants import LOG_DIR

# =========================================================================== #
#                                GLOBAL STATE
# =========================================================================== #

# This will store the path of the current log file for external reference
log_file: Optional[Path] = None

# =========================================================================== #
#                                LOGGER CLASS
# =========================================================================== #

class Logger:
    """
    Configurable logger with rotating file handler and stdout output.

    Ensures a single, well-configured logger instance is used across the
    application.

    Attributes:
        _loggers (Dict): Dictionary to cache logger instances (Singleton pattern).
    """

    _loggers: Dict[str, logging.Logger] = {}

    def __init__(
        self,
        name: str = "bloodmnist_pipeline",
        log_dir: Path = LOG_DIR,
        log_to_file: bool = True,
        level: int = logging.INFO,
        max_bytes: int = 5 * 1024 * 1024,
        backup_count: int = 5,
    ):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_to_file = log_to_file
        self.level = logging.DEBUG if os.getenv("DEBUG") == "1" else level 
        self.max_bytes = max_bytes
        self.backup_count = backup_count

        if name in Logger._loggers:
            self.logger = Logger._loggers[name]
        else:
            self.logger = logging.getLogger(name)
            self._setup_logger()
            Logger._loggers[name] = self.logger

    def _setup_logger(self):
        """Internal method to configure logging handlers and formatter."""
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
        self.logger.setLevel(self.level)
        self.logger.propagate = False

        # Console handler (StreamHandler)
        if self.logger.hasHandlers():
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Rotating File handler
        if self.log_to_file:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = self.log_dir / f"{self.name}_{timestamp}.log"

            file_handler = RotatingFileHandler(
                filename,
                maxBytes=self.max_bytes,
                backupCount=self.backup_count,
            )
            
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            # Update global log_file path
            import scripts.core.logger as logger_module
            logger_module.log_file = filename

    def get_logger(self) -> logging.Logger:
        """
        Returns the configured logger instance.

        Returns:
            logging.Logger: The configured logger object.
        """
        return self.logger
    
    @classmethod
    def setup(cls, name: str, **kwargs) -> logging.Logger:
        """
        Class method to get or create a logger instance.

        Args:
            name (str): Name of the logger.
        Returns:
            logging.Logger: The configured logger object.
        """
        instance = cls(name=name, **kwargs)
        return instance.get_logger()
        

# =========================================================================== #
#                                INITIALIZATION
# =========================================================================== #

# Create the default global logger instance
_logger_manager = Logger(log_dir=LOG_DIR)
logger = _logger_manager.get_logger()