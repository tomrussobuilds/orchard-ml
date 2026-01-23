"""
Models Factory Package

This package implements the Factory Pattern to decouple model instantiation
from the main execution logic. It routes requests to specific architecture
definitions and ensures models are correctly adapted to the dataset geometry
(channels and classes) resolved at runtime.
"""

from .efficientnet_b0 import build_efficientnet_b0

# Internal Imports
from .factory import get_model
from .mini_cnn import build_mini_cnn
from .resnet_18_adapted import build_resnet18_adapted
from .vit_tiny import build_vit_tiny

# PACKAGE INTERFACE
__all__ = [
    "get_model",
    "build_resnet18_adapted",
    "build_efficientnet_b0",
    "build_vit_tiny",
    "build_mini_cnn",
]
