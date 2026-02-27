"""
Telemetry and Reporting Package.

This package centralizes experiment logging, environment reporting, and
visual telemetry. It provides high-level utilities to initialize system-wide
loggers and format experiment metadata for reproducibility.

Available Components:

- Logger: Static utility for stream and file logging initialization.
- Reporter: Metadata reporting engine for environment baseline status.
- LogStyle: Unified logging style constants.
- Progress functions: Optimization and training progress logging.
"""

from ..paths.constants import LogStyle
from .logger import Logger
from .progress import (
    log_optimization_header,
    log_optimization_summary,
    log_pipeline_summary,
    log_trial_start,
)
from .reporter import Reporter, ReporterProtocol

__all__ = [
    "Logger",
    "Reporter",
    "ReporterProtocol",
    "LogStyle",
    "log_optimization_header",
    "log_optimization_summary",
    "log_pipeline_summary",
    "log_trial_start",
]
