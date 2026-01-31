"""
Project-wide Path Constants and Static Directory Management.

This module serves as the single source of truth for the physical filesystem
layout. It handles dynamic project root discovery and defines the static
infrastructure (dataset and output folders) required for the pipeline to boot.
"""

import os
from pathlib import Path
from typing import Final, List

# GLOBAL CONSTANTS
# Global logger identity used by all modules to ensure log synchronization
LOGGER_NAME: Final[str] = "VisionForge"


# PATH CALCULATIONS
def get_project_root() -> Path:
    """
    Dynamically locates the project root by searching for anchor files.

    Starts from the current file's directory and traverses upwards until
    it finds a marker (e.g., '.git', 'requirements.txt'). Fallback to
    fixed parents if no markers are found.
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
        return current_path.parent.parent


# Central Filesystem Authority
PROJECT_ROOT: Final[Path] = get_project_root().resolve()

# STATIC DIRECTORIES
# Input: Where raw datasets are stored
DATASET_DIR: Final[Path] = (PROJECT_ROOT / "dataset").resolve()

# Output: Default root directory for all experiment results
OUTPUTS_ROOT: Final[Path] = (PROJECT_ROOT / "outputs").resolve()

# Directories that must exist at startup
STATIC_DIRS: Final[List[Path]] = [DATASET_DIR, OUTPUTS_ROOT]


# INITIAL SETUP
def setup_static_directories() -> None:
    """
    Ensures the core project structure is present at startup.

    Creates the necessary dataset and output folders if they do not exist,
    preventing runtime errors during data fetching or log creation.
    """
    for directory in STATIC_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
