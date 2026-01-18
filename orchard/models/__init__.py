"""
Models Factory Package

This package implements the Factory Pattern to decouple model instantiation 
from the main execution logic. It routes requests to specific architecture 
definitions and ensures models are correctly adapted to the dataset geometry 
(channels and classes) resolved at runtime.
"""

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from .factory import get_model
from .resnet_18_adapted import build_resnet18_adapted
from .efficientnet_b0 import build_efficientnet_b0
from .vit_tiny import build_vit_tiny
from .mini_cnn import build_mini_cnn

# =========================================================================== #
#                                PACKAGE INTERFACE                            #
# =========================================================================== #

__all__ = [
    "get_model",
    "build_resnet18_adapted",
    "build_efficientnet_b0",
    "build_vit_tiny",
    "build_mini_cnn",
]