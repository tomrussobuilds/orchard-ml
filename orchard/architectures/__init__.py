"""
Architectures Factory Package.

Implements the Factory Pattern to decouple model instantiation from the
main execution logic. Routes requests to specific architecture definitions
and ensures models are correctly adapted to the dataset geometry (channels
and classes) resolved at runtime.
"""

from .convnext_tiny import build_convnext_tiny
from .efficientnet_b0 import build_efficientnet_b0
from .factory import get_model
from .mini_cnn import build_mini_cnn
from .resnet_18 import build_resnet18
from .timm_backbone import build_timm_model
from .vit_tiny import build_vit_tiny

__all__ = [
    "get_model",
    "build_resnet18",
    "build_efficientnet_b0",
    "build_convnext_tiny",
    "build_vit_tiny",
    "build_mini_cnn",
    "build_timm_model",
]
