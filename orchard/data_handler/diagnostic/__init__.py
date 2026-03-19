"""
Diagnostic Utilities for Health Checks and Smoke Tests.

This private submodule provides lightweight data utilities used exclusively
for pipeline validation (health checks, smoke tests, CI). These are not
part of the production training pipeline.
"""

from .synthetic import create_synthetic_dataset, create_synthetic_grayscale_dataset
from .synthetic_detection import SyntheticDetectionData, create_synthetic_detection_dataset
from .temp_loader import create_temp_loader

__all__ = [
    "create_synthetic_dataset",
    "create_synthetic_grayscale_dataset",
    "create_synthetic_detection_dataset",
    "SyntheticDetectionData",
    "create_temp_loader",
]
