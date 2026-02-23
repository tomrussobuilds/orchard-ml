"""
Filesystem Authority and Path Orchestration Package.

Centralizes all path-related logic for Orchard ML using a dual-layer approach:

1. **Static Layer** (constants module):
   - PROJECT_ROOT: Dynamically resolved project root
   - DATASET_DIR, OUTPUTS_ROOT: Global directory constants
   - LOGGER_NAME: Unified logging identity

2. **Dynamic Layer** (RunPaths class):
   - Experiment-specific directory management
   - Atomic run isolation via deterministic hashing
   - Automatic subdirectory creation (figures, models, reports, logs, etc.)

Example:
    >>> from orchard.core.paths import PROJECT_ROOT, RunPaths
    >>> print(PROJECT_ROOT)
    PosixPath('/home/user/orchard-ml')
    >>> paths = RunPaths.create(
    ...     dataset_slug="organcmnist",
    ...     model_name="EfficientNet-B0",
    ...     training_cfg={"lr": 0.001}
    ... )
"""

from .constants import (
    DATASET_DIR,
    HEALTHCHECK_LOGGER_NAME,
    LOGGER_NAME,
    METRIC_ACCURACY,
    METRIC_AUC,
    METRIC_F1,
    METRIC_LOSS,
    MLRUNS_DB,
    OUTPUTS_ROOT,
    PROJECT_ROOT,
    STATIC_DIRS,
    SUPPORTED_RESOLUTIONS,
    get_project_root,
    setup_static_directories,
)
from .run_paths import RunPaths

__all__ = [
    "PROJECT_ROOT",
    "DATASET_DIR",
    "OUTPUTS_ROOT",
    "MLRUNS_DB",
    "LOGGER_NAME",
    "HEALTHCHECK_LOGGER_NAME",
    "STATIC_DIRS",
    "SUPPORTED_RESOLUTIONS",
    "METRIC_ACCURACY",
    "METRIC_AUC",
    "METRIC_LOSS",
    "METRIC_F1",
    "get_project_root",
    "setup_static_directories",
    "RunPaths",
]
