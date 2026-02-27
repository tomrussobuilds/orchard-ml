"""
Filesystem Authority and Path Orchestration Package.

Centralizes all path-related logic for Orchard ML using a three-layer approach:

1. **Constants Layer** (constants module):

- SUPPORTED_RESOLUTIONS, METRIC_*, LOGGER_NAME: Pure project constants
- LogStyle: Unified logging style constants (symbols, separators, ANSI)

2. **Root Discovery Layer** (root module):

- PROJECT_ROOT: Dynamically resolved project root
- DATASET_DIR, OUTPUTS_ROOT: Global directory constants

3. **Dynamic Layer** (RunPaths class):

- Experiment-specific directory management
- Atomic run isolation via deterministic hashing
- Automatic subdirectory creation (figures, models, reports, logs, etc.)

Example:
    >>> from orchard.core.paths import PROJECT_ROOT, RunPaths, LogStyle
    >>> print(PROJECT_ROOT)
    PosixPath('/home/user/orchard-ml')
"""

from .constants import (
    HEALTHCHECK_LOGGER_NAME,
    LOGGER_NAME,
    METRIC_ACCURACY,
    METRIC_AUC,
    METRIC_F1,
    METRIC_LOSS,
    SUPPORTED_RESOLUTIONS,
    LogStyle,
)
from .root import (
    DATASET_DIR,
    MLRUNS_DB,
    OUTPUTS_ROOT,
    PROJECT_ROOT,
    STATIC_DIRS,
    get_project_root,
    setup_static_directories,
)
from .run_paths import RunPaths

__all__ = [
    # Constants
    "SUPPORTED_RESOLUTIONS",
    "METRIC_ACCURACY",
    "METRIC_AUC",
    "METRIC_LOSS",
    "METRIC_F1",
    "LOGGER_NAME",
    "HEALTHCHECK_LOGGER_NAME",
    "LogStyle",
    # Root & Paths
    "PROJECT_ROOT",
    "DATASET_DIR",
    "OUTPUTS_ROOT",
    "MLRUNS_DB",
    "STATIC_DIRS",
    "get_project_root",
    "setup_static_directories",
    # Run Paths
    "RunPaths",
]
