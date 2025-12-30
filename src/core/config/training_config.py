"""
Optimization & Regularization Configuration Schema.

This module defines the hyperparameters and strategy settings for model 
training. It encapsulates the optimization landscape, learning rate 
scheduling (Cosine Annealing), and advanced regularization techniques 
like Mixup and Label Smoothing.

The schema ensures that all training-related parameters are validated 
against domain-specific constraints (e.g., probability ranges and 
positive definite learning rates) before the training loop starts.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import argparse
from typing import Optional

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
from pydantic import BaseModel, Field, ConfigDict

# =========================================================================== #
#                               Internal Imports                              #
# =========================================================================== #
from .types import (
    PositiveInt, NonNegativeInt, NonNegativeFloat,
    Probability, SmoothingValue, LearningRate, GradNorm
)

# =========================================================================== #
#                             TRAINING CONFIGURATION                          #
# =========================================================================== #

class TrainingConfig(BaseModel):
    """
    Defines the optimization landscape and regularization strategies.
    """
    model_config = ConfigDict(
        frozen=True,
        extra="forbid"
    )
    
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    batch_size: PositiveInt = Field(default=128)
    epochs: PositiveInt = Field(default=60)
    patience: NonNegativeInt = Field(default=15)
    learning_rate: LearningRate = Field(default=0.008)
    min_lr: LearningRate = Field(default=1e-6)
    momentum: Probability = Field(default=0.9)
    weight_decay: NonNegativeFloat = Field(default=5e-4)
    label_smoothing: SmoothingValue = 0.0
    
    mixup_alpha: NonNegativeFloat = Field(
        default=0.2,
        description="Mixup interpolation coefficient"
    )
    mixup_epochs: NonNegativeInt = Field(
        default=20,
        description="Number of epochs to apply mixup"
    )
    
    use_tta: bool = True
    cosine_fraction: Probability = Field(default=0.5)
    use_amp: bool = False
    
    grad_clip: Optional[GradNorm] = Field(
        default=1.0,
        description="Max norm for gradient clipping; None to disable"
    )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainingConfig":
        """Map training parameters ensuring defaults are present."""
        return cls(
            seed=getattr(args, 'seed', 42),
            batch_size=getattr(args, 'batch_size', 128),
            learning_rate=getattr(args, 'lr', 0.008),
            momentum=getattr(args, 'momentum', 0.9),
            weight_decay=getattr(args, 'weight_decay', 5e-4),
            epochs=getattr(args, 'epochs', 60),
            patience=getattr(args, 'patience', 15),
            mixup_alpha=getattr(args, 'mixup_alpha', 0.2),
            mixup_epochs=getattr(args, 'mixup_epochs', 20),
            use_tta=getattr(args, 'use_tta', True),
            cosine_fraction=getattr(args, 'cosine_fraction', 0.5),
            use_amp=getattr(args, 'use_amp', False),
            grad_clip=getattr(args, 'grad_clip', 1.0),
            label_smoothing=getattr(args, 'label_smoothing', 0.0),
            min_lr=getattr(args, 'min_lr', 1e-6)
        )