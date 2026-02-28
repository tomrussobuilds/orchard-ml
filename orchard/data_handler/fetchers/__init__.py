"""
Domain-Specific Dataset Fetchers.

Each module in this sub-package handles the download and conversion logic
for a single dataset domain (MedMNIST, Galaxy10, etc.), keeping the main
``fetcher`` dispatcher clean as new sources are added.

Design note: fetcher modules intentionally duplicate some logic (e.g.
stratified splitting) rather than sharing a base class.  Each fetcher is
a self-contained adapter to an external resource whose URL, format, or
availability may change without notice.  Isolation ensures that breaking
changes in one source never cascade to others, and that any single
fetcher can be removed cleanly.
"""

from .cifar_converter import ensure_cifar_npz
from .galaxy10_converter import ensure_galaxy10_npz
from .medmnist_fetcher import ensure_medmnist_npz

__all__ = [
    "ensure_cifar_npz",
    "ensure_galaxy10_npz",
    "ensure_medmnist_npz",
]
