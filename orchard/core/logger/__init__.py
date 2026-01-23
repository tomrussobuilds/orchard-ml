"""
Telemetry and Reporting Package.

This package centralizes experiment logging, environment reporting, and
visual telemetry. It provides high-level utilities to initialize system-wide
loggers and format experiment metadata for reproducibility.

Available Components:
    - Logger: Static utility for stream and file logging initialization.
    - Reporter: Metadata reporting engine for environment baseline status.
"""

#                                Logger & Reporter Exports                    #
from .logger import Logger
from .reporter import (
    LogStyle,
    Reporter,
    log_best_config_export,
    log_optimization_header,
    log_optimization_summary,
    log_study_summary,
    log_training_summary,
    log_trial_params_compact,
    log_trial_start,
)

#                                __all__ Definition                           #
__all__ = [
    "Logger",
    "Reporter",
    "LogStyle",
    "log_best_config_export",
    "log_optimization_header",
    "log_optimization_summary",
    "log_study_summary",
    "log_training_summary",
    "log_trial_params_compact",
    "log_trial_start",
]
