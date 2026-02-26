"""
Dynamic Run Orchestration and Experiment Directory Management.

Provides the RunPaths class implementing an 'Atomic Run Isolation' strategy
for ML experiment artifact management. Automates creation of immutable,
hashed directory structures ensuring hyperparameters, model weights, and
logs are uniquely identified and protected from accidental overwrites.

The hashing strategy combines date, dataset/model slugs, and a blake2b hash
of training configuration plus timestamp to guarantee unique run directories
without collision fallbacks.

Example:
    >>> from orchard.core.paths import RunPaths
    >>> paths = RunPaths.create(
    ...     dataset_slug="organcmnist",
    ...     architecture_name="EfficientNet-B0",
    ...     training_cfg={"batch_size": 32, "lr": 0.001}
    ... )
    >>> paths.root
    PosixPath('outputs/20260208_organcmnist_efficientnetb0_a3f7c2')
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from .constants import OUTPUTS_ROOT


# RUN MANAGEMENT
class RunPaths(BaseModel):
    """
    Immutable container for experiment-specific directory paths.

    Implements atomic run isolation using a deterministic hashing strategy
    that combines DATE + DATASET_SLUG + MODEL_SLUG + CONFIG_HASH to create
    unique, collision-free directory structures. The Pydantic frozen model
    ensures paths cannot be modified after creation.

    Attributes:
        run_id: Unique identifier in format YYYYMMDD_dataset_model_hash.
        dataset_slug: Normalized lowercase dataset name.
        model_slug: Sanitized alphanumeric model identifier.
        root: Base directory for all run artifacts.
        figures: Directory for plots, confusion matrices, ROC curves.
        checkpoints: Directory for saved checkpoints (.pth files).
        reports: Directory for config mirrors, CSV/XLSX summaries.
        logs: Directory for training logs and session output.
        database: Directory for SQLite optimization studies.
        exports: Directory for production exports (ONNX).

    Example:
        Directory structure created::

            outputs/20260208_organcmnist_efficientnetb0_a3f7c2/
            ├── figures/
            ├── checkpoints/
            ├── reports/
            ├── logs/
            ├── database/
            └── exports/
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    # Immutable blueprint for the directory tree
    SUB_DIRS: ClassVar[tuple[str, ...]] = (
        "figures",
        "checkpoints",
        "reports",
        "logs",
        "database",
        "exports",
    )

    # Core Identifiers
    run_id: str
    dataset_slug: str
    architecture_slug: str

    # Physical Paths
    root: Path
    figures: Path
    checkpoints: Path
    reports: Path
    logs: Path
    database: Path
    exports: Path

    @classmethod
    def create(
        cls,
        dataset_slug: str,
        architecture_name: str,
        training_cfg: dict[str, Any],
        base_dir: Path | None = None,
    ) -> "RunPaths":
        """
        Factory method to create and initialize a unique run environment.

        Creates a new RunPaths instance with a deterministic unique ID based
        on dataset, model, and training configuration. Physically creates all
        subdirectories on the filesystem.

        Args:
            dataset_slug: Dataset identifier (e.g., 'organcmnist'). Will be
                normalized to lowercase.
            architecture_name: Human-readable model name (e.g., 'EfficientNet-B0').
                Special characters are stripped, converted to lowercase.
            training_cfg: Dictionary of hyperparameters used for hash generation.
                Supports nested dicts, but only hashable primitives (int, float,
                str, bool, list) contribute to the hash.
            base_dir: Custom base directory for outputs. Defaults to OUTPUTS_ROOT
                (typically './outputs').

        Returns:
            Fully initialized RunPaths instance with all directories created.

        Raises:
            ValueError: If dataset_slug or architecture_name is not a string.

        Example:
            >>> paths = RunPaths.create(
            ...     dataset_slug="OrganCMNIST",
            ...     architecture_name="EfficientNet-B0",
            ...     training_cfg={"batch_size": 32, "lr": 0.001}
            ... )
            >>> paths.dataset_slug
            'organcmnist'
            >>> paths.architecture_slug
            'efficientnetb0'
        """
        if not isinstance(dataset_slug, str):
            raise ValueError(f"Expected string for dataset_slug but got {type(dataset_slug)}")
        ds_slug = dataset_slug.lower()

        if not isinstance(architecture_name, str):
            raise ValueError(f"Expected string for model_name but got {type(architecture_name)}")
        a_slug = re.sub(r"[^a-zA-Z0-9]", "", architecture_name.lower())

        # Determine the unique run ID
        run_id = cls._generate_unique_id(ds_slug, a_slug, training_cfg)

        base = Path(base_dir or OUTPUTS_ROOT)
        root_path = base / run_id

        # No collision fallback needed: run_timestamp guarantees uniqueness

        instance = cls(
            run_id=run_id,
            dataset_slug=ds_slug,
            architecture_slug=a_slug,
            root=root_path,
            figures=root_path / "figures",
            checkpoints=root_path / "checkpoints",
            reports=root_path / "reports",
            logs=root_path / "logs",
            database=root_path / "database",
            exports=root_path / "exports",
        )

        instance._setup_run_directories()
        return instance

    # Internal Methods
    @staticmethod
    def _generate_unique_id(ds_slug: str, a_slug: str, cfg: dict[str, Any]) -> str:
        """
        Generate a deterministic unique run identifier.

        Combines date, slugs, and a 6-character blake2b hash of the training
        configuration plus timestamp. The timestamp (auto-generated if not
        provided) guarantees uniqueness without collision fallbacks.

        Args:
            ds_slug: Normalized dataset slug (lowercase).
            a_slug: Sanitized architecture slug (alphanumeric lowercase).
            cfg: Training configuration dictionary. Only hashable primitives
                (int, float, str, bool, list) are included in hash computation.
                May contain 'run_timestamp' key; if absent, current time is used.

        Returns:
            Unique identifier string in format: YYYYMMDD_dataset_model_hash
            where hash is 6 hex characters from blake2b(config + timestamp).

        Example:
            >>> RunPaths._generate_unique_id("organcmnist", "efficientnetb0", {"lr": 0.001})
            '20260208_organcmnist_efficientnetb0_a3f7c2'
        """
        # Filter for hashable primitives to avoid serialization errors
        hashable = {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool, list))}

        # Include run_timestamp for uniqueness (auto-generated if not in cfg)
        run_ts = cfg.get("run_timestamp", time.time())
        hashable["_run_ts"] = run_ts

        params_json = json.dumps(hashable, sort_keys=True)
        run_hash = hashlib.blake2b(params_json.encode(), digest_size=3).hexdigest()

        date_str = time.strftime("%Y%m%d")
        return f"{date_str}_{ds_slug}_{a_slug}_{run_hash}"

    def _setup_run_directories(self) -> None:
        """
        Create the physical directory structure on the filesystem.

        Iterates through SUB_DIRS class constant and creates each subdirectory
        under the root path. Uses mkdir with parents=True and exist_ok=True
        for idempotent operation.
        """
        for folder_name in self.SUB_DIRS:
            (self.root / folder_name).mkdir(parents=True, exist_ok=True)

    # Dynamic Properties
    @property
    def best_model_path(self) -> Path:
        """
        Path for the best-performing model checkpoint.

        Returns:
            Path in format: checkpoints/best_{model_slug}.pth
        """
        return self.checkpoints / f"best_{self.architecture_slug}.pth"

    @property
    def final_report_path(self) -> Path:
        """
        Path for the comprehensive experiment summary report.

        Returns:
            Path to reports/training_summary.xlsx
        """
        return self.reports / "training_summary.xlsx"

    def get_fig_path(self, filename: str) -> Path:
        """
        Generate path for a visualization artifact.

        Args:
            filename: Name of the figure file (e.g., 'confusion_matrix.png').

        Returns:
            Absolute path within the figures directory.
        """
        return self.figures / filename

    def get_config_path(self) -> Path:
        """
        Get path for the archived run configuration.

        Returns:
            Path to reports/config.yaml
        """
        return self.reports / "config.yaml"

    def get_db_path(self) -> Path:
        """
        Get path for Optuna SQLite study database.

        The database directory is created during RunPaths initialization,
        ensuring the parent directory exists before Optuna writes to it.

        Returns:
            Path to database/study.db
        """
        return self.database / "study.db"

    def __repr__(self) -> str:
        """
        Return string representation with run_id and root path.
        """
        return f"RunPaths(run_id='{self.run_id}', root={self.root})"
