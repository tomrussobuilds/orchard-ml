"""
Optuna Optimization Configuration Schema.

Pydantic v2 schema defining Optuna study parameters, search strategies,
pruning policies, storage backend configuration, and configurable
search space bounds.

Search Space Overrides:

- ``FloatRange`` / ``IntRange``: Typed bounds for continuous/discrete parameters.
- ``SearchSpaceOverrides``: Aggregates all search ranges with domain defaults.
  Overrides are applied by ``SearchSpaceRegistry`` at trial sampling time.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover
    from ..paths import RunPaths

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...exceptions import OrchardConfigError
from .types import NonNegativeInt, PositiveInt, ValidatedPath


# SEARCH SPACE RANGE MODELS
class FloatRange(BaseModel):
    """
    Typed bounds for a continuous hyperparameter search range.

    Attributes:
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).
        log: If True, sample in log-uniform distribution.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    low: float
    high: float
    log: bool = False

    @model_validator(mode="after")
    def check_bounds(self) -> "FloatRange":
        """
        Validate low < high.
        """
        if self.low >= self.high:
            raise OrchardConfigError(
                f"FloatRange low ({self.low}) must be strictly less than high ({self.high})"
            )
        return self


class IntRange(BaseModel):
    """
    Typed bounds for a discrete hyperparameter search range.

    Attributes:
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    low: int
    high: int

    @model_validator(mode="after")
    def check_bounds(self) -> "IntRange":
        """
        Validate low < high.
        """
        if self.low >= self.high:
            raise OrchardConfigError(
                f"IntRange low ({self.low}) must be strictly less than high ({self.high})"
            )
        return self


class SearchSpaceOverrides(BaseModel):
    """
    Configurable bounds for Optuna hyperparameter search ranges.

    Provides sensible defaults for image classification while allowing
    full customization via YAML. Each field maps 1:1 to a parameter
    sampled by SearchSpaceRegistry.

    Example YAML::

        optuna:
          search_space_overrides:
            learning_rate:
              low: 1e-4
              high: 1e-1
              log: true
            batch_size_low_res: [32, 64, 128]
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # ---- Optimization ----
    learning_rate: FloatRange = Field(
        default_factory=lambda: FloatRange(low=1e-5, high=1e-2, log=True)
    )
    weight_decay: FloatRange = Field(
        default_factory=lambda: FloatRange(low=1e-6, high=1e-3, log=True)
    )
    momentum: FloatRange = Field(default_factory=lambda: FloatRange(low=0.85, high=0.95))
    min_lr: FloatRange = Field(default_factory=lambda: FloatRange(low=1e-7, high=1e-5, log=True))

    # ---- Loss ----
    criterion_type: list[str] = Field(
        default=["cross_entropy", "focal"],
        description="Loss function types to explore",
    )
    focal_gamma: FloatRange = Field(default_factory=lambda: FloatRange(low=0.5, high=5.0))

    # ---- Regularization ----
    mixup_alpha: FloatRange = Field(default_factory=lambda: FloatRange(low=0.0, high=0.4))
    label_smoothing: FloatRange = Field(default_factory=lambda: FloatRange(low=0.0, high=0.2))
    dropout: FloatRange = Field(default_factory=lambda: FloatRange(low=0.1, high=0.5))

    # ---- Scheduler ----
    scheduler_type: list[str] = Field(
        default=["cosine", "plateau", "step"],
        description="Scheduler types to explore",
    )
    scheduler_patience: IntRange = Field(default_factory=lambda: IntRange(low=3, high=10))

    # ---- Augmentation ----
    rotation_angle: IntRange = Field(default_factory=lambda: IntRange(low=0, high=15))
    jitter_val: FloatRange = Field(default_factory=lambda: FloatRange(low=0.0, high=0.15))
    min_scale: FloatRange = Field(default_factory=lambda: FloatRange(low=0.9, high=1.0))

    # ---- Batch size (categorical, resolution-aware) ----
    batch_size_low_res: list[int] = Field(default=[16, 32, 48, 64])
    batch_size_high_res: list[int] = Field(default=[8, 12, 16])


# OPTUNA CONFIGURATION
class OptunaConfig(BaseModel):
    """
    Optuna hyperparameter optimization study configuration.

    Defines search strategy, pruning policy, budget, and storage backend
    for automated hyperparameter tuning.

    Attributes:
        study_name: Identifier for the Optuna study.
        n_trials: Total number of optimization trials to run.
        epochs: Training epochs per trial (typically shorter than final training).
        timeout: Maximum optimization time in seconds (None=unlimited).
        direction: Whether to 'maximize' or 'minimize' the metric.
        sampler_type: Sampling algorithm ('tpe', 'cmaes', 'random').
        search_space_preset: Predefined search space ('quick', 'full', etc.).
        enable_model_search: Include architecture in search space.
        model_pool: Restrict model search to these architectures (None=all).
        enable_early_stopping: Stop study when target performance reached.
        early_stopping_threshold: Metric threshold for early stopping.
        early_stopping_patience: Consecutive trials meeting threshold before stop.
        enable_pruning: Enable early termination of unpromising trials.
        pruner_type: Pruning algorithm ('median', 'percentile', 'hyperband').
        pruning_warmup_epochs: Minimum epochs before pruning can trigger.
        storage_type: Backend for study persistence ('sqlite', 'memory', 'postgresql').
        storage_path: Path to SQLite database file (auto-generated if None).
        postgresql_url: PostgreSQL connection string (required when storage_type='postgresql').
        n_jobs: Parallel trial execution (1=sequential, -1=all cores).
        load_if_exists: Resume existing study or create new.
        show_progress_bar: Display tqdm progress during optimization.
        save_plots: Generate optimization visualization plots.
        save_best_config: Export best trial hyperparameters as YAML.
        search_space_overrides: Configurable bounds for search ranges.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # ==================== Study Basics ====================
    study_name: str = Field(default="vision_optimization", description="Name of the Optuna study")

    n_trials: PositiveInt = Field(default=50, description="Number of optimization trials to run")

    epochs: PositiveInt = Field(
        default=15, description="Epochs per trial (shorter than final training)"
    )

    timeout: NonNegativeInt | None = Field(
        default=None, description="Max seconds for optimization (None = unlimited)"
    )

    # ==================== Optimization Target ====================
    direction: Literal["maximize", "minimize"] = Field(
        default="maximize", description="Optimization direction"
    )

    # ==================== Search Strategy ====================
    sampler_type: Literal["tpe", "cmaes", "random"] = Field(
        default="tpe", description="Hyperparameter sampling algorithm"
    )

    search_space_preset: Literal["quick", "full"] = Field(
        default="full", description="Predefined search space configuration"
    )

    enable_model_search: bool = Field(
        default=False,
        description=(
            "Include model architecture in search space (resolution-aware: "
            "28px → mini_cnn/resnet_18, 224px → efficientnet_b0/vit_tiny + weight variants)"
        ),
    )

    model_pool: list[str] | None = Field(
        default=None,
        description=(
            "Restrict model search to a subset of architectures. "
            "When None, all built-in models for the target resolution are searched. "
            "Requires enable_model_search=True. Minimum 2 entries. "
            "Each name must be a valid model factory key or use 'timm/' prefix."
        ),
    )

    # ==================== Early Stopping ====================
    enable_early_stopping: bool = Field(
        default=True, description="Stop study when target performance is reached"
    )

    early_stopping_threshold: float | None = Field(
        default=None,
        description="Metric threshold for early stopping (None=auto from monitor_metric)",
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

    storage_path: ValidatedPath | None = Field(
        default=None, description="Path to SQLite database (auto-generated if None)"
    )

    postgresql_url: str | None = Field(
        default=None,
        description="PostgreSQL connection string (e.g. postgresql://user:pass@host/db)",
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

    # ==================== Search Space Overrides ====================
    search_space_overrides: SearchSpaceOverrides = Field(
        default_factory=SearchSpaceOverrides,
        description="Configurable bounds for hyperparameter search ranges",
    )

    @model_validator(mode="after")
    def check_model_pool(self) -> "OptunaConfig":
        """
        Validate model_pool constraints.

        Raises:
            ValueError: If model_pool set without enable_model_search,
                        or contains fewer than 2 entries.

        Returns:
            Validated OptunaConfig instance.
        """
        if self.model_pool is not None:
            if not self.enable_model_search:
                raise OrchardConfigError("model_pool requires enable_model_search=True")
            if len(self.model_pool) < 2:
                raise OrchardConfigError(
                    "model_pool must contain at least 2 architectures for meaningful search"
                )
        return self

    @model_validator(mode="after")
    def validate_storage(self) -> "OptunaConfig":
        """
        Validate storage backend configuration.

        Raises:
            OrchardConfigError: If PostgreSQL selected without postgresql_url,
                or if postgresql_url set with non-postgresql storage_type,
                or if postgresql_url has an invalid scheme.

        Returns:
            Validated OptunaConfig instance.
        """
        if self.storage_type == "postgresql":
            if self.postgresql_url is None:
                raise OrchardConfigError(
                    "PostgreSQL storage requires postgresql_url "
                    "(e.g. postgresql://user:pass@host/db)"
                )
            if not self.postgresql_url.startswith(("postgresql://", "postgresql+")):
                raise OrchardConfigError(
                    f"postgresql_url must start with 'postgresql://' or 'postgresql+', "
                    f"got: '{self.postgresql_url[:30]}...'"
                )
        elif self.postgresql_url is not None:
            raise OrchardConfigError(
                f"postgresql_url is set but storage_type is '{self.storage_type}', "
                f"not 'postgresql'"
            )
        return self

    @model_validator(mode="after")
    def check_pruning(self) -> "OptunaConfig":
        """
        Validate pruning warmup is less than total epochs.

        Raises:
            ValueError: If pruning_warmup_epochs >= epochs.

        Returns:
            Validated OptunaConfig instance.
        """
        if self.enable_pruning and self.pruning_warmup_epochs >= self.epochs:
            raise OrchardConfigError(
                f"pruning_warmup_epochs ({self.pruning_warmup_epochs}) "
                f"must be < epochs ({self.epochs})"
            )
        return self

    @model_validator(mode="after")
    def check_tqdm_flag(self) -> "OptunaConfig":
        """
        Warn about potential tqdm corruption with parallel execution.

        Returns:
            Validated OptunaConfig instance (with warning if applicable).
        """
        if self.show_progress_bar and self.n_jobs != 1:
            warnings.warn("show_progress_bar=True with n_jobs!=1 may corrupt tqdm output.")
        return self

    def get_storage_url(self, paths: RunPaths) -> str | None:
        """
        Constructs storage URL for Optuna study.

        Args:
            paths (RunPaths): RunPaths instance providing database directory

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
            return self.postgresql_url

        raise OrchardConfigError(f"Unknown storage type: {self.storage_type}")
