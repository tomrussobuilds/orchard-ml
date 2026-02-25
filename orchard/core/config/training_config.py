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
    * Reproducibility: Global seeding for experiment replication (determinism in HardwareConfig)

Strict boundary validation (probability ranges, LR bounds) prevents unstable
training states before first batch processing.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .types import (
    BatchSize,
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

    Attributes:
        seed: Random seed for reproducibility.
        batch_size: Training samples per batch (1-128).
        epochs: Maximum training epochs.
        patience: Early stopping patience in epochs.
        use_tqdm: Enable progress bar display.
        optimizer_type: Optimizer algorithm ('sgd' or 'adamw').
        learning_rate: Initial learning rate (1e-8 to 1.0).
        min_lr: Minimum learning rate for scheduler.
        momentum: SGD momentum coefficient.
        weight_decay: L2 regularization strength.
        grad_clip: Maximum gradient norm for clipping.
        label_smoothing: Label smoothing factor (0.0-0.3).
        mixup_alpha: Mixup interpolation coefficient.
        mixup_epochs: Number of epochs to apply mixup.
        use_tta: Enable test-time augmentation.
        scheduler_type: LR scheduler type ('cosine', 'plateau', 'step', 'none').
        monitor_metric: Metric driving checkpointing and early stopping
            ('auc', 'accuracy', or 'f1').
        scheduler_patience: ReduceLROnPlateau patience epochs.
        scheduler_factor: LR reduction factor for plateau scheduler.
        step_size: StepLR decay period in epochs.
        use_amp: Enable Automatic Mixed Precision training.
        criterion_type: Loss function ('cross_entropy' or 'focal').
        weighted_loss: Enable class-frequency loss weighting.
        focal_gamma: Focal loss focusing parameter.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # ==================== Reproducibility ====================
    seed: int = Field(default=42, description="Random seed")

    # ==================== Training Loop ====================
    batch_size: BatchSize = Field(default=16, description="Samples per batch")
    epochs: PositiveInt = Field(default=60, description="Maximum epochs")
    patience: NonNegativeInt = Field(default=15, description="Early stopping patience")
    use_tqdm: bool = Field(default=True, description="Progress Bar activation / deactivation")

    # ==================== Optimization ====================
    optimizer_type: Literal["sgd", "adamw"] = Field(
        default="sgd", description="Optimizer algorithm"
    )
    learning_rate: LearningRate = Field(default=0.008, description="Initial learning rate")
    min_lr: LearningRate = Field(default=1e-6, description="Minimum learning rate")
    momentum: Momentum = Field(default=0.9, description="SGD momentum")
    weight_decay: WeightDecay = Field(default=5e-4, description="L2 regularization")
    grad_clip: GradNorm | None = Field(default=1.0, description="Max gradient norm")

    # ==================== Regularization ====================
    label_smoothing: SmoothingValue = Field(default=0.0, description="Label smoothing factor")
    mixup_alpha: NonNegativeFloat = Field(default=0.2, description="Mixup coefficient")
    mixup_epochs: NonNegativeInt = Field(default=20, description="Mixup duration (epochs)")
    use_tta: bool = Field(default=True, description="Test-time augmentation")

    # ==================== Scheduler ====================
    scheduler_type: Literal["cosine", "plateau", "step", "none"] = Field(
        default="cosine", description="LR scheduler type"
    )
    monitor_metric: Literal["auc", "accuracy", "f1"] = Field(
        default="auc", description="Metric for checkpointing and early stopping"
    )
    scheduler_patience: NonNegativeInt = Field(default=5, description="Plateau patience")
    scheduler_factor: Probability = Field(default=0.1, description="LR reduction factor")
    step_size: PositiveInt = Field(default=20, description="StepLR decay period")
    use_amp: bool = Field(default=True, description="Automatic Mixed Precision")

    # ==================== Loss ====================
    criterion_type: Literal["cross_entropy", "focal"] = Field(
        default="cross_entropy", description="Loss function type"
    )
    weighted_loss: bool = Field(default=False, description="Class-frequency weighting")
    focal_gamma: NonNegativeFloat = Field(default=2.0, description="Focal Loss gamma")

    @model_validator(mode="after")
    def validate_amp(self) -> "TrainingConfig":
        """
        Validate AMP compatibility with batch size.

        Raises:
            ValueError: If AMP enabled with batch_size < 4.

        Returns:
            Validated TrainingConfig instance.
        """
        if self.use_amp and self.batch_size < 4:
            raise ValueError("AMP enabled with very small batch size (<4) can cause NaN gradients.")
        return self
