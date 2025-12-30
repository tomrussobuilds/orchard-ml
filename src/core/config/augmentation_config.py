"""
Data Augmentation & Test-Time Augmentation (TTA) Schema.

This module defines the parameters for the stochastic transformation pipeline. 
It synchronizes geometric and photometric noise levels used during training 
with the intensity of perturbations applied during Test-Time Augmentation (TTA), 
ensuring a calibrated approach to model robustness.

Key Architectural Components:
    * Training Augmentations: Parameters for horizontal flips, rotations, 
      color jitter, and scaling.
    * TTA Strategy: Specific offsets for translation, scaling factors, 
      and Gaussian blur sigma used for ensemble-based inference.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import argparse

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
from pydantic import BaseModel, Field, ConfigDict

# =========================================================================== #
#                               Internal Imports                              #
# =========================================================================== #
from .types import (
    Probability, RotationDegrees, NonNegativeFloat
)

# =========================================================================== #
#                          AUGMENTATION CONFIGURATION                         #
# =========================================================================== #

class AugmentationConfig(BaseModel):
    """
    Configures the stochastic transformation pipeline for training and TTA.
    
    Standardizes parameters for geometric and photometric augmentations, 
    ensuring consistency between training-time noise and Test-Time 
    Augmentation (TTA) intensity.
    """
    model_config = ConfigDict(
        frozen=True,
        extra="forbid"
    )
    
    hflip: Probability = Field(default=0.5)
    rotation_angle: RotationDegrees = Field(default=10)
    jitter_val: NonNegativeFloat = Field(default=0.2)
    min_scale: Probability = Field(default=0.9)

    tta_translate: float = Field(
        default=2.0,
        description="Pixel shift for TTA"
    )
    tta_scale: float = Field(
        default=1.1,
        description="Scale factor for TTA"
    )
    tta_blur_sigma: float = Field(
        default=0.4,
        description="Gaussian blur sigma for TTA"
    )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "AugmentationConfig":
        """Map augmentation parameters ensuring defaults are present."""
        return cls(
            hflip=getattr(args, 'hflip', 0.5),
            rotation_angle=getattr(args, 'rotation_angle', 10),
            jitter_val=getattr(args, 'jitter_val', 0.2),
            min_scale=getattr(args, 'min_scale', 0.9),
            tta_translate=getattr(args, 'tta_translate', 2.0),
            tta_scale=getattr(args, 'tta_scale', 1.1),
            tta_blur_sigma=getattr(args, 'tta_blur_sigma', 0.4)
        )