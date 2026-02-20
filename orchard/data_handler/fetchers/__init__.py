"""
Domain-Specific Dataset Fetchers

Each module in this sub-package handles the download and conversion logic
for a single dataset domain (MedMNIST, Galaxy10, etc.), keeping the main
``fetcher`` dispatcher clean as new sources are added.
"""

from .cifar_converter import ensure_cifar_npz
from .galaxy10_converter import ensure_galaxy10_npz
from .medmnist_fetcher import ensure_medmnist_npz

__all__ = [
    "ensure_cifar_npz",
    "ensure_galaxy10_npz",
    "ensure_medmnist_npz",
]
