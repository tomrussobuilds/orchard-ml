"""
Constants and Path Configuration Module

This module defines the global directory structure and provides utilities for
managing experiment-specific paths (Runs). It supports a clean separation 
between raw datasets and timestamped experiment outputs.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import time
from pathlib import Path
from typing import Final, List

# =========================================================================== #
#                                PATH CALCULATIONS                            #
# =========================================================================== #

def get_project_root() -> Path:
    """
    Returns the absolute path to the project root directory.
    
    The root is assumed to be due livelli sopra la cartella dove risiede questo file.
    """
    try:
        # Resolves: scripts/core/constants.py -> scripts/core/ -> scripts/ -> root/
        return Path(__file__).resolve().parent.parent.parent
    except NameError:
        # Fallback per ambienti interattivi senza __file__
        return Path.cwd().resolve()

PROJECT_ROOT: Final[Path] = get_project_root()

# =========================================================================== #
#                                STATIC DIRECTORIES                           #
# =========================================================================== #

# Input: Where raw datasets are stored
DATASET_DIR: Final[Path] = (PROJECT_ROOT / "dataset").resolve()

# Output: Root directory for all experiment results
OUTPUTS_ROOT: Final[Path] = (PROJECT_ROOT / "outputs").resolve()

# Directories that must exist at startup
STATIC_DIRS: Final[List[Path]] = [DATASET_DIR, OUTPUTS_ROOT]

# =========================================================================== #
#                                RUN MANAGEMENT                               #
# =========================================================================== #

class RunPaths:
    """
    Manages experiment-specific directories to prevent overwriting results.
    
    Each training session gets a unique directory under 'outputs/' based on 
    the timestamp, dataset, and model name.
    """
    def __init__(self, dataset_slug: str, model_name: str):
        """
        Args:
            dataset_slug (str): Unique identifier for the dataset (from DATASET_REGISTRY).
            model_name (str): Human-readable model name (from Config).
        """
        # Format: 20251221_143005_bloodmnist_resnet18 (include secondi per unicità)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        self.model_slug : Final[str] = model_name.lower().replace(" ", "_").replace("-", "_")
        self.ds_slug: Final[str] = dataset_slug
        self.project_id: Final[str] = f"{self.ds_slug}_{self.model_slug}"
        self.run_id: Final[str] = f"{timestamp}_{self.project_id}"
        
        self.root: Final[Path] = OUTPUTS_ROOT / self.run_id
        self.figures: Final[Path] = self.root / "figures"
        self.models: Final[Path] = self.root / "models"
        self.reports: Final[Path] = self.root / "reports"
        self.logs: Final[Path] = self.root / "logs"
        
        self._setup_run_directories()

    def _setup_run_directories(self) -> None:
        """Ensures all sub-directories for the current run exist."""
        run_dirs = [self.figures, self.models, self.reports, self.logs]
        for path in run_dirs:
            path.mkdir(parents=True, exist_ok=True)

    def get_config_path(self) -> Path:
        """Returns the path for the configuration YAML file."""
        return self.root / "config.yaml"
    
    def __repr__(self) -> str:
        return f"RunPaths(run_id={self.run_id}, root={self.root})"


# =========================================================================== #
#                                INITIAL SETUP                                #
# =========================================================================== #

def setup_static_directories() -> None:
    """Ensures the core project structure is present."""
    for directory in STATIC_DIRS:
        directory.mkdir(parents=True, exist_ok=True)