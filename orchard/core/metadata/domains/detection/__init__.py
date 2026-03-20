"""
Detection Dataset Domain Modules.

Organizes detection dataset metadata by domain with per-resolution
registries. Detection datasets provide bounding-box annotations
alongside images.
"""

from .pennfudan import REGISTRY_224 as PENNFUDAN_224

__all__ = [
    "PENNFUDAN_224",
]
