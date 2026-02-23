"""
Orchard ML: type-Safe Deep Learning for Reproducible Research.

Top-level convenience API re-exporting the most commonly used components
from subpackages, so users and the ``orchard`` CLI can write:

    from orchard import Config, RootOrchestrator, get_model
"""

from importlib.metadata import version as _pkg_version

__version__ = _pkg_version("orchard-ml")

from .architectures import get_model
from .core import (
    Config,
    LogStyle,
    RootOrchestrator,
    log_pipeline_summary,
)
from .core.paths import MLRUNS_DB
from .pipeline import run_export_phase, run_optimization_phase, run_training_phase
from .tracking import create_tracker

__all__ = [
    "__version__",
    # Core
    "Config",
    "LogStyle",
    "RootOrchestrator",
    "log_pipeline_summary",
    # Paths
    "MLRUNS_DB",
    # Architectures
    "get_model",
    # Pipeline
    "run_export_phase",
    "run_optimization_phase",
    "run_training_phase",
    # Tracking
    "create_tracker",
]
