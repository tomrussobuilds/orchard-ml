"""
Data Augmentation & Test-Time Augmentation (TTA) Schema.

Declarative schema for stochastic transformation pipeline. Synchronizes
geometric and photometric noise used during training with TTA perturbations
for calibrated model robustness.

Key Features:
    * Geometric invariance: Horizontal flips and rotation for medical scan
      orientation generalization
    * Photometric consistency: Color jitter and scaling for acquisition variations
    * TTA ensemble strategy: Pixel shifts, scaling, and Gaussian blur for
      stable predictions through stochastic averaging
    * Validation guards: Domain-specific types ensuring physically plausible
      ranges for medical imaging
"""

import argparse

from pydantic import BaseModel, ConfigDict, Field

from .types import BlurSigma, NonNegativeFloat, PixelShift, Probability, RotationDegrees, ZoomScale


# AUGMENTATION CONFIGURATION
class AugmentationConfig(BaseModel):
    """
    Stochastic transformations for training and test-time augmentation (TTA).

    Centralizes hyperparameters for geometric and photometric perturbations
    applied during training and inference phases.

    Attributes:
        hflip: Probability of horizontal flip during training (0.0-1.0).
        rotation_angle: Maximum rotation angle in degrees (0-360).
        jitter_val: Color jitter intensity for brightness, contrast, saturation.
        min_scale: Minimum scale factor for random resized crop (0.0-1.0).
        tta_translate: Pixel translation range for TTA ensemble.
        tta_scale: Scale factor for TTA zoom augmentation.
        tta_blur_sigma: Gaussian blur sigma for TTA smoothing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Training augmentations
    hflip: Probability = Field(default=0.5, description="Horizontal flip probability")
    rotation_angle: RotationDegrees = Field(
        default=10, description="Maximum rotation angle (degrees)"
    )
    jitter_val: NonNegativeFloat = Field(default=0.2, description="Color jitter intensity")
    min_scale: Probability = Field(default=0.9, description="Minimum random resize scale")

    # TTA parameters
    tta_mode: str = Field(
        default="full",
        description="TTA ensemble complexity: 'full' (with rotations) or 'light' (no rotations)",
    )
    tta_translate: PixelShift = Field(default=2.0, description="TTA pixel shift")
    tta_scale: ZoomScale = Field(default=1.1, description="TTA scaling factor")
    tta_blur_sigma: BlurSigma = Field(default=0.4, description="TTA Gaussian blur sigma")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "AugmentationConfig":
        """
        Factory from CLI arguments.

        Args:
            args: Parsed argparse namespace

        Returns:
            Configured AugmentationConfig instance
        """
        # Extract with defaults matching Field definitions
        params = {
            field: getattr(args, field, cls.model_fields[field].default)
            for field in cls.model_fields
            if hasattr(args, field)
        }
        return cls(**params)
