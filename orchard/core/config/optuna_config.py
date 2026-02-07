"""
Optuna Optimization Configuration Schema.

Pydantic v2 schema defining Optuna study parameters, search strategies,
pruning policies, and storage backend configuration.
"""

import argparse
import warnings
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .types import NonNegativeInt, PositiveInt, ValidatedPath


# OPTUNA CONFIGURATION
class OptunaConfig(BaseModel):
    """
    Optuna hyperparameter optimization study configuration.

    Defines search strategy, pruning policy, budget, and storage backend.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # ==================== Study Basics ====================
    study_name: str = Field(default="vision_optimization", description="Name of the Optuna study")

    n_trials: PositiveInt = Field(default=50, description="Number of optimization trials to run")

    epochs: PositiveInt = Field(
        default=15, description="Epochs per trial (shorter than final training)"
    )

    timeout: Optional[NonNegativeInt] = Field(
        default=None, description="Max seconds for optimization (None = unlimited)"
    )

    # ==================== Optimization Target ====================
    metric_name: str = Field(
        default="auc", description="Metric to optimize ['loss', 'accuracy', 'auc']"
    )

    direction: Literal["maximize", "minimize"] = Field(
        default="maximize", description="Optimization direction"
    )

    # ==================== Search Strategy ====================
    sampler_type: Literal["tpe", "cmaes", "random", "grid"] = Field(
        default="tpe", description="Hyperparameter sampling algorithm"
    )

    search_space_preset: Literal["quick", "full", "optimization_only", "regularization_only"] = (
        Field(default="full", description="Predefined search space configuration")
    )

    enable_model_search: bool = Field(
        default=False,
        description=(
            "Include model architecture in search space (resolution-aware: "
            "28px → mini_cnn/resnet_18, 224px → efficientnet_b0/vit_tiny + weight variants)"
        ),
    )

    # ==================== Early Stopping ====================
    enable_early_stopping: bool = Field(
        default=True, description="Stop study when target performance is reached"
    )

    early_stopping_threshold: Optional[float] = Field(
        default=None, description="Metric threshold for early stopping (None=auto from metric_name)"
    )

    early_stopping_patience: PositiveInt = Field(
        default=2, description="Consecutive trials meeting threshold before stopping"
    )

    # ==================== Pruning Strategy ====================
    enable_pruning: bool = Field(
        default=True, description="Enable early stopping of unpromising trials"
    )

    pruner_type: Literal["median", "percentile", "hyperband", "none"] = Field(
        default="median", description="Pruning algorithm for early stopping"
    )

    pruning_warmup_epochs: NonNegativeInt = Field(
        default=5, description="Min epochs before pruning can trigger"
    )

    # ==================== Storage Backend ====================
    storage_type: Literal["sqlite", "memory", "postgresql"] = Field(
        default="sqlite", description="Backend for study persistence"
    )

    storage_path: Optional[ValidatedPath] = Field(
        default=None, description="Path to SQLite database (auto-generated if None)"
    )

    # ==================== Execution Policy ====================
    n_jobs: int = Field(default=1, description="Parallel trials (1=sequential, -1=all cores)")

    load_if_exists: bool = Field(default=True, description="Resume existing study or create new")

    show_progress_bar: bool = Field(default=False, description="Display optimization progress")

    # ==================== Reporting ====================
    save_plots: bool = Field(
        default=True, description="Generate and save optimization visualizations"
    )

    save_best_config: bool = Field(default=True, description="Export best trial as YAML config")

    @model_validator(mode="after")
    def validate_storage(self) -> "OptunaConfig":
        """Validates storage backend configuration."""
        if self.storage_type == "postgresql" and self.storage_path is None:
            raise ValueError("PostgreSQL storage requires storage_path with connection string")
        return self

    @model_validator(mode="after")
    def check_metric_name(self) -> "OptunaConfig":
        """Validates metric_name is supported."""
        allowed_metrics = ["auc", "accuracy", "loss"]
        if self.metric_name not in allowed_metrics:
            raise ValueError(f"metric_name '{self.metric_name}' invalid. Choose: {allowed_metrics}")
        return self

    @model_validator(mode="after")
    def check_pruning(self) -> "OptunaConfig":
        """Validates pruning configuration."""
        if self.enable_pruning and self.pruning_warmup_epochs >= self.epochs:
            raise ValueError(
                f"pruning_warmup_epochs ({self.pruning_warmup_epochs}) "
                f"must be < epochs ({self.epochs})"
            )
        return self

    @model_validator(mode="after")
    def check_tqdm_flag(self) -> "OptunaConfig":
        """Warns about tqdm with parallel jobs."""
        if self.show_progress_bar and self.n_jobs != 1:
            warnings.warn("show_progress_bar=True with n_jobs>1 may corrupt tqdm output.")
        return self

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "OptunaConfig":
        """Factory from CLI arguments."""
        args_dict = vars(args)
        valid_fields = cls.model_fields.keys()
        params = {k: v for k, v in args_dict.items() if k in valid_fields and v is not None}
        return cls(**params)

    def get_storage_url(self, paths) -> Optional[str]:
        """
        Constructs storage URL for Optuna study.

        Args:
            paths: RunPaths instance

        Returns:
            Storage URL string (sqlite:// or postgresql://)
        """
        if self.storage_type == "memory":
            return None

        if self.storage_type == "sqlite":
            if self.storage_path:
                db_path = self.storage_path
            else:
                # Use RunPaths database path
                db_path = paths.get_db_path()
            return f"sqlite:///{db_path}"

        if self.storage_type == "postgresql":
            return str(self.storage_path)

        raise ValueError(f"Unknown storage type: {self.storage_type}")
