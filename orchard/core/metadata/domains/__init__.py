"""
Dataset Domain Modules.

Organizes dataset metadata by task type and domain to support
multi-domain framework usage while maintaining single source of truth pattern.

- ``classification/`` — medical, space, benchmark domains
- ``detection/`` — object detection domains
"""

from .classification import (
    BENCHMARK_32,
    MEDICAL_28,
    MEDICAL_64,
    MEDICAL_128,
    MEDICAL_224,
    SPACE_224,
)

__all__ = [
    "BENCHMARK_32",
    "MEDICAL_28",
    "MEDICAL_64",
    "MEDICAL_128",
    "MEDICAL_224",
    "SPACE_224",
]
