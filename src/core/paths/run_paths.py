"""
Dynamic Run Orchestration and Experiment Directory Management.

This module provides the RunPaths class, which automates the creation of 
unique, timestamped directory structures for each experiment. It ensures 
that model weights, logs, and diagnostic figures are organized and 
protected from accidental overwrites.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import re
import time
from pathlib import Path
from typing import Final, Optional

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from .constants import OUTPUTS_ROOT

# =========================================================================== #
#                                RUN MANAGEMENT                               #
# =========================================================================== #

class RunPaths:
    """
    Manages experiment-specific directories to prevent overwriting results.
    
    Each training session gets a unique directory based on the timestamp, 
    dataset slug, and model name. 
    
    Example structure:
        outputs/20260103_131843_pathmnist_resnet_18/
        ├── figures/
        ├── models/
        ├── reports/
        └── logs/
    """
    
    def __init__(
        self, 
        dataset_slug: str, 
        model_name: str, 
        base_dir: Optional[Path] = None
    ):
        """
        Initializes the unique run environment and creates physical directories.

        Args:
            dataset_slug (str): Unique identifier for the dataset (e.g., 'pathmnist').
            model_name (str): Human-readable model name (e.g., 'ResNet-18').
            base_dir (Path, optional): Custom base directory for outputs. 
                                      Defaults to constants.OUTPUTS_ROOT.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Consistent slugification for filesystem safety
        clean_model_name = re.sub(r'[^a-zA-Z0-9]', '_', model_name.lower())
        self.model_slug: Final[str] = clean_model_name.strip('_')
        self.ds_slug: Final[str] = dataset_slug.lower()
        
        # Identifiers for logging and reporting
        self.project_id: Final[str] = f"{self.ds_slug}_{self.model_slug}"
        self.run_id: Final[str] = f"{timestamp}_{self.project_id}"
        
        # Root directory for this specific run
        base = base_dir if base_dir is not None else OUTPUTS_ROOT
        self.root: Final[Path] = base / self.run_id
        
        # Standardized sub-directories
        self.figures: Final[Path] = self.root / "figures"
        self.models: Final[Path] = self.root / "models"
        self.reports: Final[Path] = self.root / "reports"
        self.logs: Final[Path] = self.root / "logs"
        
        # Immediate physical setup
        self._setup_run_directories()
    
    @property
    def best_model_path(self) -> Path:
        """Standardized path for the top-performing model checkpoint."""
        return self.models / f"best_model_{self.model_slug}.pth"
    
    @property
    def final_report_path(self) -> Path:
        """Destination path for the comprehensive experiment summary (Excel/CSV)."""
        return self.reports / "training_summary.xlsx"
    
    def get_fig_path(self, filename: str) -> Path:
        """Generates an absolute path for a visualization artifact."""
        return self.figures / filename

    def _setup_run_directories(self) -> None:
        """Creates the physical directory tree for the current run."""
        run_dirs = [self.figures, self.models, self.reports, self.logs]
        for path in run_dirs:
            path.mkdir(parents=True, exist_ok=True)

    def get_config_path(self) -> Path:
        """Returns the path where the run configuration is archived."""
        return self.reports / "config.yaml"
    
    def __repr__(self) -> str:
        return f"RunPaths(run_id={self.run_id}, root={self.root})"