"""
Model Architecture Configuration Module.

This module defines the schema and validation logic for the deep learning 
architectures used in the pipeline. It handles parameters related to 
model identification, weight initialization (pre-training), and 
structural adaptations like dropout for regularization.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import argparse

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
from pydantic import (
    BaseModel, ConfigDict, Field
)

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from .types import DropoutRate

# =========================================================================== #
#                                MODEL CONFIGURATION                         #
# =========================================================================== #

class ModelConfig(BaseModel):
    """
    Configuration for Model Architecture and Weight Initialization.
    
    This sub-config manages the structural identity of the model and its
    initial state. By isolating these parameters, the pipeline can easily 
    scale to support diverse architectures while maintaining a consistent 
    interface for the Model Factory.
    """
    model_config = ConfigDict(
        extra="forbid",
        frozen=True
    )
    
    name: str = Field(
        default="resnet_18_adapted",
        description="The unique identifier for the model architecture."
    )
    
    pretrained: bool = Field(
        default=True,
        description="Whether to initialize the model with pre-trained weights."
    )
    
    dropout: DropoutRate = Field(
        default=0.2,
        description="Dropout probability for the classification head."
    )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ModelConfig":
        """
        Factory method to create a ModelConfig from CLI arguments.
        """
        return cls(
            name=getattr(args, 'model_name', "resnet18"),
            pretrained=getattr(args, 'pretrained', True),
            dropout=getattr(args, 'dropout', 0.2)
        )