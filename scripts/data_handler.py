"""
Data Preparation Module for BloodMNIST

Handles the downloading, loading, and preprocessing of the BloodMNIST dataset.
This includes utilities for robust data fetching, data structuring, defining
a PyTorch Dataset, and creating DataLoaders with appropriate augmentation
(strong augmentation for training, standard normalization for validation/testing).
"""
# =========================================================================== #
#                                Standard Imports
# =========================================================================== #
from pathlib import Path
import time
from typing import Tuple, Final
import requests
from dataclasses import dataclass
import logging
import random

# =========================================================================== #
#                                Third-Party Imports
# =========================================================================== #
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# =========================================================================== #
#                                Internal Imports
# =========================================================================== #
from .utils import (
    Config, Logger,
    md5_checksum, validate_npz_keys,
    EXPECTED_MD5, URL, NPZ_PATH
    )

# Global logger instance
logger: Final[logging.Logger] = Logger().get_logger()

# =========================================================================== #
#                                 Data Utilities
# =========================================================================== #

def worker_init_fn(worker_id: int):
    """
    Initializes random number generators (PRNGs) for each DataLoader worker.
    This ensures data transformations are deterministic per worker process.

    Args:
        worker_id (int): ID of the worker process.
    """
    # Use the base configuration seed
    initial_seed = Config().seed
    
    # Create a unique seed for the worker
    worker_seed = initial_seed + worker_id 
    
    # Apply the seed to standard PRNGs used by transformations
    np.random.seed(worker_seed)
    random.seed(worker_seed) 
    torch.manual_seed(worker_seed)
    
    # PyTorch handles its own PRNG for the DataLoader internally 
    # based on this logic, so no need for an explicit torch.manual_seed.
    pass

def ensure_mnist_npz(target_npz: Path, retries: int = 5, delay: float = 5.0) -> Path:
    """
    Downloads the BloodMNIST dataset NPZ file robustly, with retries and MD5 validation.

    Args:
        target_npz (Path): The expected path for the dataset NPZ file.
        retries (int): Max number of download attempts.
        delay (float): Delay (seconds) between retries.

    Returns:
        Path: Path to the successfully validated .npz file.

    Raises:
        RuntimeError: If the dataset cannot be downloaded and verified.
    """
    def _is_valid(path: Path) -> bool:
        """Checks file existence, size, and MD5 checksum."""
        if not path.exists() or path.stat().st_size < 50_000:
            return False
        # Check for ZIP header (NPZ files are ZIP archives)
        if path.read_bytes()[:2] != b"PK":
            return False
        return md5_checksum(path) == EXPECTED_MD5

    if _is_valid(target_npz):
        logger.info(f"Valid dataset found: {target_npz}")
        return target_npz

    if target_npz.exists():
        logger.warning(f"Corrupted or incomplete dataset found, deleting: {target_npz}")
        target_npz.unlink()

    logger.info(f"Downloading BloodMNIST from {URL}")
    tmp_path = target_npz.with_suffix(".tmp")
    
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(URL, timeout=60)
            response.raise_for_status() 
            tmp_path.write_bytes(response.content)

            if not _is_valid(tmp_path):
                raise ValueError("Downloaded file failed validation (wrong size/header/MD5)")

            tmp_path.replace(target_npz) # Atomic move
            logger.info(f"Successfully downloaded and verified: {target_npz}")
            return target_npz

        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()

            if attempt == retries:
                logger.error(f"Failed to download dataset after {retries} attempts")
                raise RuntimeError("Could not download BloodMNIST dataset") from e

            logger.warning(f"Attempt {attempt}/{retries} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)

    raise RuntimeError("Unexpected error in dataset download logic.")


@dataclass(frozen=True)
class BloodMNISTData:
    """A container for the BloodMNIST dataset splits stored as NumPy arrays."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def load_bloodmnist(npz_path: Path = NPZ_PATH) -> BloodMNISTData:
    """
    Loads the dataset from the NPZ file, validates its keys, and returns
    the structured data splits.
    
    Args:
        npz_path (Path, optional): Path to the NPZ file.

    Returns:
        BloodMNISTData: The structured dataset splits.
    """
    path = ensure_mnist_npz(npz_path)

    logger.info(f"Loading dataset from {path}")

    # Use mmap_mode="r" for memory efficiency
    with np.load(npz_path, mmap_mode="r") as data:
        validate_npz_keys(data)
        logger.info(f"Keys in NPZ file: {data.files}")

        return BloodMNISTData(
            X_train=data["train_images"],
            X_val=data["val_images"],
            X_test=data["test_images"],
            y_train=data["train_labels"].ravel(), # Flatten labels
            y_val=data["val_labels"].ravel(),
            y_test=data["test_labels"].ravel(),
        )


class BloodMNISTDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """
    PyTorch Dataset for BloodMNIST data. Handles NumPy to Tensor conversion
    and optional image transformations.
    """
    def __init__(self,
                 images: np.ndarray,
                 labels: np.ndarray,
                 transform: transforms.Compose | None = None):
        """
        Args:
            images (np.ndarray): Image data (H, W, C).
            labels (np.ndarray): Label data.
            transform (transforms.Compose | None): Torchvision transformations.
        """
        # Normalize pixel values to [0.0, 1.0]
        self.images = images.astype(np.float32) / 255.0
        # Convert labels to int64 (long) for loss function
        self.labels = labels.astype(np.int64) 
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            # Apply transformation (e.g., augmentation for training)
            img = self.transform(img)
        else:
             # Convert NumPy array (H, W, C) to PyTorch Tensor (C, H, W)
             img = torch.from_numpy(img).permute(2, 0, 1)


        return img, torch.tensor(label, dtype=torch.long)

def get_dataloaders(
        data: BloodMNISTData,
        cfg: Config,
        ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders for train, validation, and test splits.

    Uses strong augmentation for training and standard normalization for validation/testing.

    Args:
        data (BloodMNISTData): The structured dataset splits.
        cfg (Config): Configuration containing batch_size and num_workers.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader).
    """
    # 1. Define Augmentations and Transformations
    # Strong augmentation for the training set
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomResizedCrop(28, scale=(0.9, 1.0)),
        transforms.ToTensor(), # Scales to [0, 1] and converts to (C, H, W)
        # ImageNet normalization
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Standard normalization for validation and test sets
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. Create Datasets
    train_ds = BloodMNISTDataset(data.X_train, data.y_train, transform=train_transform)
    val_ds   = BloodMNISTDataset(data.X_val,   data.y_val,   transform=val_transform)
    test_ds  = BloodMNISTDataset(data.X_test,  data.y_test,  transform=val_transform)

    # Apply worker_init_fn only if parallel workers are used
    init_fn = worker_init_fn if cfg.num_workers > 0 else None
    
    # 3. Create DataLoaders
    pin_memory = torch.cuda.is_available() # Pin memory if CUDA is available

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=init_fn
    )
    val_loader   = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=init_fn
    )
    test_loader  = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=init_fn
    )
    
    # Log dataset sizes for confirmation
    logger.info(f"Dataset loaded → Train:{len(train_ds)} | Val:{len(val_ds)} | Test:{len(test_ds)}")
    
    return train_loader, val_loader, test_loader