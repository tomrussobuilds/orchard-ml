"""
Filesystem Authority and Path Orchestration Package.

This package centralizes all path-related logic for the MedMNIST pipeline.
It provides a dual-layer approach:
1. Static: Project root and global directory constants via 'constants'.
2. Dynamic: Experiment-specific directory management via 'RunPaths'.
"""

# =========================================================================== #
#                                Public Interface                             #
# =========================================================================== #
from .constants import (
    DATASET_DIR,
    LOGGER_NAME,
    OUTPUTS_ROOT,
    PROJECT_ROOT,
    setup_static_directories
)
from .run_paths import RunPaths

# =========================================================================== #
#                                Export Schema                                #
# =========================================================================== #
__all__ = [
    "PROJECT_ROOT",
    "DATASET_DIR",
    "OUTPUTS_ROOT",
    "LOGGER_NAME",
    "STATIC_DIR",
    "setup_static_directories",
    "RunPaths",
]
