"""
Model Architecture Configuration Module.

This module defines the declarative schema for deep learning architectures.
It has been refactored to delegate geometric resolution (channels, classes)
to the DatasetConfig, ensuring a Single Source of Truth (SSOT) and preventing
architectural mismatches during model instantiation.
"""

# Standard Imports
import argparse
from typing import Optional

# Third-Party Imports
from pydantic import BaseModel, ConfigDict, Field

# Internal Imports
from .types import DropoutRate


# MODELS CONFIGURATION
class ModelConfig(BaseModel):
    """
    Configuration for Model Architecture and Weight Initialization.

    This sub-config manages the structural identity and regularization policies.
    Geometric constraints (input depth and output logits) are intentionally
    omitted here to be resolved dynamically via DatasetConfig at runtime.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    name: str = Field(
        default="resnet_18_adapted",
        description="The unique identifier for the model architecture. E.g., 'resnet_18_adapted', 'efficientnet_b0'.",
    )

    pretrained: bool = Field(
        default=True, description="Whether to initialize the model with pre-trained weights."
    )

    dropout: DropoutRate = Field(
        default=0.2, description="Dropout probability for the classification head."
    )

    weight_variant: Optional[str] = Field(
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

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ModelConfig":
        """
        Factory method to create a ModelConfig from CLI arguments.

        This method no longer requires Metadata injection as geometric
        resolution has been moved to the Dataset orchestration layer.
        """
        return cls(
            name=getattr(args, "model_name", "resnet18"),
            pretrained=getattr(args, "pretrained", True),
            dropout=getattr(args, "dropout", 0.2),
        )
