"""
Model Architecture Configuration Module.

This module defines the declarative schema for deep learning architectures.
It has been refactored to delegate geometric resolution (channels, classes)
to the DatasetConfig, ensuring a Single Source of Truth (SSOT) and preventing
architectural mismatches during model instantiation.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .types import DropoutRate


# ARCHITECTURE CONFIGURATION
class ArchitectureConfig(BaseModel):
    """
    Configuration for model architecture and weight initialization.

    Manages structural identity and regularization policies. Geometric
    constraints (input channels and output classes) are resolved dynamically
    via DatasetConfig at runtime to ensure consistency.

    Attributes:
        name: Model architecture identifier (e.g., 'efficientnet_b0', 'vit_tiny').
        pretrained: Whether to initialize with pretrained ImageNet weights.
        dropout: Dropout probability for the classification head (0.0-0.9).
        weight_variant: Specific pretrained weight variant for architectures
            with multiple options (e.g., ViT variants with different pretraining).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(
        default="mini_cnn",
        description="The unique identifier for the model architecture. E.g., 'efficientnet_b0'.",
    )

    pretrained: bool = Field(
        default=False, description="Whether to initialize the model with pre-trained weights."
    )

    dropout: DropoutRate = Field(
        default=0.2,
        description=(
            "Dropout probability for the classification head. "
            "Only wired for mini_cnn and timm_backbone; ignored by "
            "convnext_tiny, efficientnet_b0, vit_tiny, resnet_18."
        ),
    )

    weight_variant: str | None = Field(
        default=None,
        description=(
            "Pretrained weight variant for architectures with multiple options. "
            "Examples for ViT-Tiny: "
            "'vit_tiny_patch16_224.augreg_in21k_ft_in1k' (ImageNet-21k → 1k), "
            "'vit_tiny_patch16_224.augreg_in21k' (ImageNet-21k only), "
            "'vit_tiny_patch16_224' (ImageNet-1k baseline). "
            "If None, uses default variant for the selected architecture."
        ),
    )
