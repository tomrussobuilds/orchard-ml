"""
Data Handler Package

This package manages the end-to-end data pipeline, from downloading raw NPZ 
files using the Dataset Registry to providing fully configured PyTorch DataLoaders.
"""

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from .fetcher import (
    load_medmnist, 
    MedMNISTData, 
    ensure_dataset_npz
)

from .factory import (
    create_temp_loader,
    get_dataloaders
)

from .data_explorer import (
    show_sample_images
)

from .transforms import (
    get_augmentations_description,
    get_pipeline_transforms,
    worker_init_fn
)

from .dataset import (
    MedMNISTDataset
)

# =========================================================================== #
#                                PUBLIC API                                   #
# =========================================================================== #

__all__ = [
    "load_medmnist",
    "MedMNISTData",
    "ensure_dataset_npz",
    "get_dataloaders",
    "show_sample_images",
    "get_augmentations_description",
    "get_pipeline_transforms",
    "worker_init_fn",
    "MedMNISTDataset"
]