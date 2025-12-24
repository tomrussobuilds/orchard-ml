"""
Path Management and Project Constants

Centralizes the filesystem authority for the project. It defines the root 
directory structure and provides the `RunPaths` orchestrator to manage 
unique, timestamped experiment directories, ensuring that logs, models, 
and reports are never overwritten.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import time
import re
from pathlib import Path
from typing import Final, List, Optional

# =========================================================================== #
#                                PATH CALCULATIONS                            #
# =========================================================================== #

def get_project_root() -> Path:
    """
    Dynamically locates the project root by searching for anchor files.
    
    Starts from the current file's directory and traverses upwards until 
    it finds a marker (e.g., '.git', 'requirements.txt'). Fallback to 
    fixed parents if no markers are found.
    """
    # Start from the directory of this file
    current_path = Path(__file__).resolve().parent
    
    # Look for markers that define the project root
    root_markers = {".git",
                    "requirements.txt",
                    "README.md"
                    }
    
    for parent in [current_path] + list(current_path.parents):
        if any((parent / marker).exists() for marker in root_markers):
            return parent
            
    return current_path.parent.parent

PROJECT_ROOT: Final[Path] = get_project_root().resolve()

# =========================================================================== #
#                                STATIC DIRECTORIES                           #
# =========================================================================== #

# Input: Where raw datasets are stored
DATASET_DIR: Final[Path] = (PROJECT_ROOT / "dataset").resolve()

# Output: Default root directory for all experiment results
OUTPUTS_ROOT: Final[Path] = (PROJECT_ROOT / "outputs").resolve()

# Directories that must exist at startup
STATIC_DIRS: Final[List[Path]] = [DATASET_DIR, OUTPUTS_ROOT]

# =========================================================================== #
#                                RUN MANAGEMENT                               #
# =========================================================================== #

class RunPaths:
    """
    Manages experiment-specific directories to prevent overwriting results.
    
    Each training session gets a unique directory based on the timestamp, 
    dataset slug, and model name.
    """
    def __init__(self, dataset_slug: str, model_name: str, base_dir: Optional[Path] = None):
        """
        Args:
            dataset_slug (str): Unique identifier for the dataset.
            model_name (str): Human-readable model name.
            base_dir (Path, optional): Base directory for outputs. Defaults to OUTPUTS_ROOT.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Consistent slugification
        clean_model_name = re.sub(r'[^a-zA-Z0-9]', '_', model_name.lower())
        self.model_slug: Final[str] = clean_model_name.strip('_')
        self.ds_slug: Final[str] = dataset_slug.lower()
        self.project_id: Final[str] = f"{self.ds_slug}_{self.model_slug}"
        self.run_id: Final[str] = f"{timestamp}_{self.project_id}"
        
        # Use provided base_dir (from Config/CLI) or fallback to global constant
        base = base_dir if base_dir is not None else OUTPUTS_ROOT
        self.root: Final[Path] = base / self.run_id
        
        # Sub-directories
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
    """Ensures the core project structure is present at startup."""
    for directory in STATIC_DIRS:
        directory.mkdir(parents=True, exist_ok=True)