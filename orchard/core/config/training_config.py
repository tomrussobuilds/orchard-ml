"""
Optimization & Regularization Configuration Schema.

Declarative schema for training lifecycle, orchestrating optimization
landscape and regularization boundaries. Synchronizes learning rate
dynamics (Cosine Annealing) with data augmentation (Mixup) and
precision policies (AMP).

Key Features:
    * Optimization dynamics: LR schedules and momentum for loss landscape navigation
    * Regularization suite: Label smoothing and Mixup for medical imaging generalization
    * Precision & stability: AMP and gradient clipping for hardware throughput and stability
    * Reproducibility guard: Global seeding and strict determinism for bit-perfect replication

Strict boundary validation (probability ranges, LR bounds) prevents unstable
training states before first batch processing.
"""

# Standard Imports
import argparse
from typing import Optional

# Third-Party Imports
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Internal Imports
from .types import (
    GradNorm,
    LearningRate,
    Momentum,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    Probability,
    SmoothingValue,
    WeightDecay,
)


# TRAINING CONFIGURATION
class TrainingConfig(BaseModel):
    """
    Optimization landscape and regularization strategies.

    Validates training hyperparameters and provides structure for
    reproducibility, optimization, regularization, and scheduling.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # ==================== Reproducibility ====================
    seed: int = Field(default=42, description="Random seed")
    reproducible: bool = Field(default=False, description="Strict reproducibility mode")

    # ==================== Training Loop ====================
    batch_size: PositiveInt = Field(default=16, description="Samples per batch")
    epochs: PositiveInt = Field(default=60, description="Maximum epochs")
    patience: NonNegativeInt = Field(default=15, description="Early stopping patience")
    use_tqdm: bool = Field(default=True, description="Progress Bar activation / deactivaion")

    # ==================== Optimization ====================
    learning_rate: LearningRate = Field(default=0.008, description="Initial learning rate")
    min_lr: LearningRate = Field(default=1e-6, description="Minimum learning rate")
    momentum: Momentum = Field(default=0.9, description="SGD momentum")
    weight_decay: WeightDecay = Field(default=5e-4, description="L2 regularization")
    grad_clip: Optional[GradNorm] = Field(default=1.0, description="Max gradient norm")

    # ==================== Regularization ====================
    label_smoothing: SmoothingValue = Field(default=0.0, description="Label smoothing factor")
    mixup_alpha: NonNegativeFloat = Field(default=0.2, description="Mixup coefficient")
    mixup_epochs: NonNegativeInt = Field(default=20, description="Mixup duration (epochs)")
    use_tta: bool = Field(default=True, description="Test-time augmentation")

    # ==================== Scheduler ====================
    scheduler_type: str = Field(
        default="cosine", description="LR scheduler: 'cosine', 'plateau', 'step', 'none'"
    )
    cosine_fraction: Probability = Field(default=0.5, description="Cosine annealing fraction")
    scheduler_patience: NonNegativeInt = Field(default=5, description="Plateau patience")
    scheduler_factor: Probability = Field(default=0.1, description="LR reduction factor")
    step_size: PositiveInt = Field(default=20, description="StepLR decay period")
    use_amp: bool = Field(default=True, description="Automatic Mixed Precision")

    # ==================== Loss ====================
    criterion_type: str = Field(
        default="cross_entropy", description="Loss: 'cross_entropy', 'focal', 'bce_logit'"
    )
    weighted_loss: bool = Field(default=False, description="Class-frequency weighting")
    focal_gamma: NonNegativeFloat = Field(default=2.0, description="Focal Loss gamma")

    @model_validator(mode="after")
    def validate_batch_size(self) -> "TrainingConfig":
        if self.batch_size > 128:
            raise ValueError(
                f"Batch size too large ({self.batch_size}). Reduce to <=128 for AMP stability."
            )
        return self

    @model_validator(mode="after")
    def validate_amp(self) -> "TrainingConfig":
        if self.use_amp and self.batch_size < 4:
            raise ValueError("AMP enabled with very small batch size (<4) can cause NaN gradients.")
        return self

    @model_validator(mode="after")
    def validate_lr(self) -> "TrainingConfig":
        if not (0 < self.learning_rate <= 1):
            raise ValueError(f"learning_rate={self.learning_rate} is out of bounds (0,1).")
        return self

    # ==================== Factory Method ====================
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainingConfig":
        """
        Factory from CLI arguments.

        Only overrides schema fields present in args and not None.
        Automatically adapts to new schema fields.

        Args:
            args: Parsed command-line arguments

        Returns:
            TrainingConfig with CLI-overridden values
        """
        args_dict = vars(args)
        valid_fields = cls.model_fields.keys()
        params = {k: v for k, v in args_dict.items() if k in valid_fields and v is not None}
        return cls(**params)
