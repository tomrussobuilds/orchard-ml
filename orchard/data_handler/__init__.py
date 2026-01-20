"""
Data Handler Package
This package manages the end-to-end data pipeline, from downloading raw NPZ files
using the Dataset Registry to providing fully configured PyTorch DataLoaders.
"""

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from .data_explorer import show_sample_images, show_samples_for_dataset
from .dataset import MedMNISTDataset
from .factory import DataLoaderFactory, get_dataloaders
from .fetcher import MedMNISTData, ensure_dataset_npz, load_medmnist
from .synthetic import create_synthetic_dataset
from .transforms import get_augmentations_description, get_pipeline_transforms

# =========================================================================== #
#                                PUBLIC API                                   #
# =========================================================================== #
__all__ = [
    "load_medmnist",
    "MedMNISTData",
    "ensure_dataset_npz",
    "get_dataloaders",
    "DataLoaderFactory",
    "show_sample_images",
    "show_samples_for_dataset",
    "get_augmentations_description",
    "get_pipeline_transforms",
    "MedMNISTDataset",
    "create_synthetic_dataset",
]
