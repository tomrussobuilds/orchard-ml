"""
Project-wide Path Constants and Static Directory Management.

Single source of truth for the physical filesystem layout. Handles dynamic
project root discovery and defines static infrastructure (dataset and output
directories) required for pipeline initialization.

Module Attributes:
    LOGGER_NAME: Global logger identity used by all modules for log synchronization.
    HEALTHCHECK_LOGGER_NAME: Logger identity for dataset validation utilities.
    PROJECT_ROOT: Dynamically resolved absolute path to the project root.
    DATASET_DIR: Absolute path to the raw datasets directory.
    OUTPUTS_ROOT: Default root directory for all experiment results.
    STATIC_DIRS: List of directories that must exist at startup.
"""

import os
from pathlib import Path
from typing import Final, FrozenSet, List

# GLOBAL CONSTANTS
# Supported image resolutions across all model architectures
SUPPORTED_RESOLUTIONS: Final[FrozenSet[int]] = frozenset({28, 32, 64, 224})

# Global logger identity used by all modules to ensure log synchronization
LOGGER_NAME: Final[str] = "OrchardML"

# Health check logger identity for dataset validation utilities
HEALTHCHECK_LOGGER_NAME: Final[str] = "healthcheck"


# PATH CALCULATIONS
def get_project_root() -> Path:
    """
    Dynamically locate the project root by searching for anchor files.

    Traverses upward from current file's directory until finding a marker
    file (.git or requirements.txt). Supports Docker environments via
    IN_DOCKER environment variable override.

    Returns:
        Resolved absolute Path to the project root directory.

    Note:
        - IN_DOCKER=1 or IN_DOCKER=TRUE returns /app
        - Falls back to fixed parent traversal if no markers found
    """
    # Environment override for Docker setups
    if str(os.getenv("IN_DOCKER")).upper() in ("1", "TRUE"):
        return Path("/app").resolve()

    # Start from the directory of this file
    current_path = Path(__file__).resolve().parent

    # Look for markers that define the project root
    # Note: .git is most reliable; README.md alone can exist in subdirectories
    root_markers = {".git", "requirements.txt"}

    for parent in [current_path] + list(current_path.parents):
        if any((parent / marker).exists() for marker in root_markers):
            return parent

    # Fallback if no markers are found
    try:
        if len(current_path.parents) >= 3:
            return current_path.parents[2]
    except IndexError:  # pragma: no cover
        pass

    # Final fallback
    return current_path.parent.parent  # pragma: no cover


# Central Filesystem Authority
PROJECT_ROOT: Final[Path] = get_project_root().resolve()

# STATIC DIRECTORIES
# Input: Where raw datasets are stored
DATASET_DIR: Final[Path] = (PROJECT_ROOT / "dataset").resolve()

# Output: Default root directory for all experiment results
OUTPUTS_ROOT: Final[Path] = (PROJECT_ROOT / "outputs").resolve()

# Tracking: SQLite database for MLflow experiment tracking
MLRUNS_DB: Final[Path] = (PROJECT_ROOT / "mlruns.db").resolve()

# Directories that must exist at startup
STATIC_DIRS: Final[List[Path]] = [DATASET_DIR, OUTPUTS_ROOT]


# INITIAL SETUP
def setup_static_directories() -> None:
    """
    Ensure core project directories exist at startup.

    Creates DATASET_DIR and OUTPUTS_ROOT if they do not exist, preventing
    runtime errors during data fetching or artifact creation. Uses
    mkdir(parents=True, exist_ok=True) for idempotent operation.
    """
    for directory in STATIC_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
