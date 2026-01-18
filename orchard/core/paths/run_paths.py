"""
Dynamic Run Orchestration and Experiment Directory Management.

This module provides the RunPaths class, which implements an 'Atomic Run Isolation' 
strategy. It automates the creation of immutable, hashed directory structures, 
ensuring that hyperparameters, model weights, and logs are uniquely identified 
and shielded from accidental resource overlap or overwrites.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import json
import re
import time
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, ClassVar

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
from pydantic import BaseModel, ConfigDict

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from .constants import OUTPUTS_ROOT

# =========================================================================== #
#                                RUN MANAGEMENT                               #
# =========================================================================== #

class RunPaths(BaseModel):
    """
    Manages experiment-specific directories using an atomic hashing strategy.
    
    Instead of long timestamps, it uses a combination of DATE + SLUGS + HASH 
    to create clean, unique, and immutable directory structures.
    
    Example structure:
        outputs/20260116_organcmnist_efficientnetb0_a3f7c2/
        ├── figures/    <- Plots, confusion matrices, ROC curves
        ├── models/     <- Saved checkpoints (.pth)
        ├── reports/    <- Config mirrors, CSV summaries
        ├── logs/       <- Standard output and training logs
        └── database/   <- SQLite optimization studies
    """
    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True
    )

    # Immutable blueprint for the directory tree
    SUB_DIRS: ClassVar[tuple[str, ...]] = (
        "figures",
        "models",
        "reports",
        "logs",
        "database",
    )

    # Core Identifiers
    run_id: str
    dataset_slug: str
    model_slug: str
    
    # Physical Paths
    root: Path
    figures: Path
    models: Path
    reports: Path
    logs: Path
    database: Path

    @classmethod
    def create(
        cls, 
        dataset_slug: str, 
        model_name: str, 
        training_cfg: Dict[str, Any], 
        base_dir: Optional[Path] = None
    ) -> "RunPaths":
        """
        Factory method to initialize a unique run environment.
        
        Args:
            dataset_slug: Identifier for the dataset (e.g., 'organcmnist').
            model_name: Human-readable model name (e.g., 'EfficientNet-B0').
            training_cfg: Dictionary of hyperparameters used for unique hashing.
            base_dir: Optional custom base directory for outputs.
        """
        if not isinstance(dataset_slug, str):
            raise ValueError(
                f"Expected string for dataset_slug but got {type(dataset_slug)}"
            )
        ds_slug = dataset_slug.lower()

        if not isinstance(model_name, str):
            raise ValueError(
                f"Expected string for model_name but got {type(model_name)}"
            )
        m_slug = re.sub(r'[^a-zA-Z0-9]', '', model_name.lower())

        # Determine the unique run ID
        run_id = cls._generate_unique_id(ds_slug, m_slug, training_cfg)
        
        base = Path(base_dir or OUTPUTS_ROOT)
        root_path = base / run_id

        # Safety collision check (rare with blake2b but handled)
        if root_path.exists():
            run_id = f"{run_id}_{time.strftime('%H%M%S')}"
            root_path = base / run_id
        
        instance = cls(
            run_id=run_id,
            dataset_slug=ds_slug,
            model_slug=m_slug,
            root=root_path,
            figures=root_path / "figures",
            models=root_path / "models",
            reports=root_path / "reports",
            logs=root_path / "logs",
            database=root_path / "database",
        )
        
        instance._setup_run_directories()
        return instance

    # ======================================================================= #
    #                             Internal Helpers                            #
    # ======================================================================= #

    @staticmethod
    def _generate_unique_id(ds_slug: str, m_slug: str, cfg: Dict[str, Any]) -> str:
        """
        Calculates a deterministic 6-character hash from the training configuration.
        Ensures that even slight hyperparameter changes result in new directories.
        """
        # Filter for hashable primitives to avoid serialization errors
        hashable = {
            k: v for k, v in cfg.items()
            if isinstance(v, (int, float, str, bool, list))
        }
        params_json = json.dumps(hashable, sort_keys=True)
        run_hash = hashlib.blake2b(params_json.encode(), digest_size=3).hexdigest()
        
        date_str = time.strftime("%Y%m%d")
        return f"{date_str}_{ds_slug}_{m_slug}_{run_hash}"

    def _setup_run_directories(self) -> None:
        """Physically creates the directory structure on the filesystem."""
        for folder_name in self.SUB_DIRS:
            (self.root / folder_name).mkdir(parents=True, exist_ok=True)

    # ======================================================================= #
    #                            Dynamic Properties                           #
    # ======================================================================= #

    @property
    def best_model_path(self) -> Path:
        """Standardized path for the top-performing model checkpoint."""
        return self.models / f"best_{self.model_slug}.pth"
    
    @property
    def final_report_path(self) -> Path:
        """Destination path for the comprehensive experiment summary."""
        return self.reports / "training_summary.xlsx"

    def get_fig_path(self, filename: str) -> Path:
        """Generates an absolute path for a visualization artifact."""
        return self.figures / filename

    def get_config_path(self) -> Path:
        """Returns the path where the run configuration is archived."""
        return self.reports / "config.yaml"
    
    def get_db_path(self) -> Path:
        """
        Returns path for Optuna SQLite database.
        
        Database directory is created during RunPaths initialization.
        
        Returns:
            Path to study database file
        """
        return self.database / "study.db"
    
    def __repr__(self) -> str:
        return f"RunPaths(run_id='{self.run_id}', root={self.root})"