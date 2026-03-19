"""
Data Handler Package.

This package manages the end-to-end data pipeline, from downloading raw NPZ files
using the Dataset Registry to providing fully configured PyTorch DataLoaders.
"""

from .collate import detection_collate_fn
from .data_explorer import show_sample_images, show_samples_for_dataset
from .dataset import VisionDataset
from .detection_dataset import DetectionDataset
from .diagnostic import (
    create_synthetic_dataset,
    create_synthetic_grayscale_dataset,
    create_temp_loader,
)
from .fetcher import DatasetData, ensure_dataset_npz, load_dataset
from .loader import DataLoaderFactory, get_dataloaders
from .transforms import get_augmentations_description, get_pipeline_transforms

__all__ = [
    "load_dataset",
    "DatasetData",
    "ensure_dataset_npz",
    "get_dataloaders",
    "DataLoaderFactory",
    "create_temp_loader",
    "show_sample_images",
    "show_samples_for_dataset",
    "get_augmentations_description",
    "get_pipeline_transforms",
    "VisionDataset",
    "DetectionDataset",
    "detection_collate_fn",
    "create_synthetic_dataset",
    "create_synthetic_grayscale_dataset",
]
