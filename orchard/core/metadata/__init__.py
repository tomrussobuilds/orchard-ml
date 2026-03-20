"""
Dataset Metadata Package.

This package centralizes the specifications for all supported datasets.
It serves as the single source of truth for the Orchard, ensuring that
data dimensions, labels, and normalization constants are consistent
across the entire pipeline.
"""

from .base import DatasetMetadata
from .wrapper import (
    ClassificationRegistryWrapper,
    DatasetRegistryWrapper,
    DetectionRegistryWrapper,
    get_registry,
)

__all__ = [
    "ClassificationRegistryWrapper",
    "DatasetMetadata",
    "DatasetRegistryWrapper",
    "DetectionRegistryWrapper",
    "get_registry",
]
