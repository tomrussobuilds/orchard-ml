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
from PIL import Image

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
            cfg: Config = Config()
            ):
        """
        Args:
            path (Path): Path to the .npz file.
            split (str): One of 'train', 'val', or 'test'.
            transform (transforms.Compose | None): Torchvision transformations.
            max_samples (int | None): Limits the dataset size if it exceeds this value.
            cfg (Config): Global configuration for seeding.
        """
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found in: {path}")
        self.path = path
        self.transform = transform
        self.split = split
        
        # Load the specific split into RAM to avoid slow I/O during training
        with np.load(path) as data:
            # Efficiently access arrays without full duplication
            raw_images = data[f"{split}_images"]
            raw_labels = data[f"{split}_labels"].ravel().astype(np.int64)
            
            total_available = len(raw_labels)
            
            # Manage deterministic subsampling efficiently
            if max_samples and max_samples < total_available:
                rng = np.random.default_rng(cfg.training.seed)
                # rng.choice is more memory efficient than shuffling all indices
                chosen_indices = rng.choice(total_available, size=max_samples, replace=False)
                self.images = raw_images[chosen_indices]
                self.labels = raw_labels[chosen_indices]
            else:
                # np.array() forces the data into RAM, releasing the file handle
                self.images = np.array(raw_images)
                self.labels = raw_labels

            # Standardize shape to (N, H, W, C) during initialization
            # This avoids repeated 'ndim' checks in __getitem__
            if self.images.ndim == 3:  # Grayscale: (N, H, W) -> (N, H, W, 1)
                self.images = np.expand_dims(self.images, axis=-1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample from the dataset. 
        Access is now O(1) as data is pre-loaded in RAM.
        """
        img = self.images[idx]
        label = self.labels[idx]

        # Apply transformation pipeline
        if self.transform:
            # torchvision transforms (v1) expect PIL Images
            # We squeeze only if it's grayscale (H, W, 1) -> (H, W)
            pil_img = Image.fromarray(img.squeeze() if img.shape[-1] == 1 else img)
            img = self.transform(pil_img)
        else:
            # Fallback tensor conversion: optimized HWC -> CHW
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        return img, torch.tensor(label, dtype=torch.long)