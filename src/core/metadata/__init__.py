"""
Dataset Metadata Package

This package centralizes the specifications for all supported datasets. 
It serves as the single source of truth for the Orchard, ensuring that 
data dimensions, labels, and normalization constants are consistent 
across the entire pipeline.
"""

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from .base import DatasetMetadata
from .medmnist_v2_28x28 import DATASET_REGISTRY

# =========================================================================== #
#                                PUBLIC REGISTRY                              #
# =========================================================================== #

# Expose at package level for direct access via scripts.core.metadata
__all__ = [
    "DatasetMetadata",
    "DATASET_REGISTRY"
]