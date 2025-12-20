"""
Constants and Path Configuration Module

This module defines all global constants, including directory structures,
dataset metadata (URLs, MD5 checksums), and class taxonomies for BloodMNIST.
"""
# =========================================================================== #
#                                Standard Imports
# =========================================================================== #
from pathlib import Path
from typing import Final, List

# =========================================================================== #
#                                PATH CALCULATIONS
# =========================================================================== #

def get_project_root() -> Path:
    """
    Returns the absolute path to the project root directory.
    
    The root is assumed to be one level above the 'scripts/core' directory.
    """
    try:
        # Resolves to the 'scripts' folder, then parent to reach project root
        return Path(__file__).resolve().parent.parent.parent
    except NameError:
        # Fallback for interactive shells
        return Path.cwd()

PROJECT_ROOT: Final[Path] = get_project_root()

# =========================================================================== #
#                                DIRECTORIES
# =========================================================================== #

DATASET_DIR: Final[Path] = PROJECT_ROOT / "dataset"
FIGURES_DIR: Final[Path] = PROJECT_ROOT / "figures"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
LOG_DIR: Final[Path] = PROJECT_ROOT / "logs"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"

ALL_DIRS: Final[List[Path]] = [
    DATASET_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    LOG_DIR,
    REPORTS_DIR
]

# =========================================================================== #
#                                DATASET METADATA
# =========================================================================== #

# File path to the local .npz file
DATASET_NAME: Final[str] = "BloodMNIST"
NPZ_PATH: Final[Path] = DATASET_DIR / f"{DATASET_NAME}.npz"

# Integrity and Source
EXPECTED_MD5: Final[str] = "7053d0359d879ad8a5505303e11de1dc"
URL: Final[str] = "https://zenodo.org/record/5208230/files/bloodmnist.npz?download=1"

# Official BloodMNIST taxonomy
BLOODMNIST_CLASSES: Final[List[str]] = [
    "basophil", 
    "eosinophil", 
    "erythroblast", 
    "immature granulocyte",
    "lymphocyte", 
    "monocyte", 
    "neutrophil", 
    "platelet"
]

# =========================================================================== #
#                                DIRECTORY SETUP
# =========================================================================== #

def setup_directories(directories: List[Path]) -> None:
    """
    Ensures that all required project directories exist.
    
    Args:
        directories (List[Path]): List of directory paths to create.
    """
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)