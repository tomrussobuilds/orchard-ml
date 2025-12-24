"""
PyTorch Dataset Definition Module

This module contains the custom Dataset class for MedMNIST, handling
the conversion from NumPy arrays to PyTorch tensors and applying 
image transformations for training and inference.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
from typing import Tuple
from pathlib import Path

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# =========================================================================== #
#                              Internal Imports                               #
# =========================================================================== #
from src.core import Config

# =========================================================================== #
#                                DATASET CLASS                                #
# =========================================================================== #

class MedMNISTDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Enhanced PyTorch Dataset for MedMNIST data. Supports:
    - Selective RAM loading (balanced efficiency)
    - Subsampling with fixed seed (deterministic)
    - Automatic handling of RGB/Grayscale
    """
    def __init__(
            self,
            path: Path,
            split: str = "train",
            transform: transforms.Compose | None = None,
            max_samples: int | None = None,
            cfg: Config = Config
            ):
        """
        Args:
            path (Path): Path to the .npz file.
            split (str): One of 'train', 'val', or 'test'.
            transform (transforms.Compose | None): Torchvision transformations.
            max_samples (int | None): Limits the dataset size if it exceeds this value.
            cfg (Config): Global configuration for seeding.
        """
        self.path = path
        self.transform = transform
        self.split = split
        
        # Load the specific split into RAM to avoid slow I/O during training
        with np.load(path) as data:
            # np.array() forces the data into RAM, releasing the file handle
            full_images = np.array(data[f"{split}_images"])
            full_labels = data[f"{split}_labels"].ravel().astype(np.int64)
            
            total_available = len(full_labels)
            indices = np.arange(total_available)
            
            # Manage deterministic subsampling
            if max_samples and max_samples < total_available:
                rng = np.random.default_rng(cfg.training.seed)
                rng.shuffle(indices)
                self.indices = indices[:max_samples]
            else:
                self.indices = indices
            
            # Store only the required subset in memory
            self.images = full_images[self.indices]
            self.labels = full_labels[self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample from the dataset. 
        Access is now O(1) as data is pre-loaded in RAM.
        """
        img = self.images[idx]
        label = self.labels[idx]

        # Apply transformation pipeline
        if self.transform:
            img = self.transform(img)
        else:
            # Fallback tensor conversion
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img)
            if img.ndim == 2:  # Grayscale
                img = img.unsqueeze(0)
            else:  # RGB
                img = img.permute(2, 0, 1)

        return img, torch.tensor(label, dtype=torch.long)