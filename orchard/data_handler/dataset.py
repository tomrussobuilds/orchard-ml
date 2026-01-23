"""
PyTorch Dataset Definition Module

This module contains the custom Dataset class for MedMNIST, handling
the conversion from NumPy arrays to PyTorch tensors and applying
image transformations for training and inference.

It implements selective RAM loading to balance I/O speed with memory
efficiency and ensures deterministic subsampling for reproducible research.
"""

# Standard Imports
from pathlib import Path
from typing import Final, Tuple

# Third-Party Imports
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Internal Imports
from orchard.core import Config


# DATASET CLASS
class MedMNISTDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Enhanced PyTorch Dataset for MedMNIST data.

    Features:
        - In-memory caching of specific splits to eliminate disk I/O bottlenecks.
        - Seed-aware deterministic subsampling for rapid smoke testing.
        - Automatic dimensionality standardization (N, H, W, C).
    """

    def __init__(
        self,
        path: Path,
        split: str = "train",
        transform: transforms.Compose | None = None,
        max_samples: int | None = None,
        cfg: Config = None,
    ):
        """
        Initializes the dataset by loading the specified .npz split into RAM.

        Args:
            path (Path): Path to the MedMNIST .npz archive.
            split (str): Dataset split to load ('train', 'val', or 'test').
            transform (transforms.Compose | None): Pipeline of Torchvision transforms.
            max_samples (int | None): If set, limits the number of samples (subsampling).
            cfg (Config): Global configuration used to extract the random seed.
        """
        if cfg is None:
            raise ValueError("A valid Config instance is required.")
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {path}")

        self.cfg: Final[Config] = cfg
        self.path: Final[Path] = path
        self.transform: Final[transforms.Compose | None] = transform
        self.split: Final[str] = split

        # Open NPZ once and load target arrays into system memory
        with np.load(path) as data:
            raw_images = data[f"{split}_images"]
            raw_labels = data[f"{split}_labels"].ravel().astype(np.int64)

            total_available = len(raw_labels)

            # Deterministic subsampling logic
            if max_samples and max_samples < total_available:
                rng = np.random.default_rng(cfg.training.seed)
                chosen_indices = rng.choice(total_available, size=max_samples, replace=False)
                self.images = raw_images[chosen_indices]
                self.labels = raw_labels[chosen_indices]
            else:
                self.images = np.array(raw_images)
                self.labels = raw_labels

            # Standard Imports
            # This ensures consistent PIL conversion regardless of source format
            if self.images.ndim == 3:  # (N, H, W) -> (N, H, W, 1)
                self.images = np.expand_dims(self.images, axis=-1)

    def __len__(self) -> int:
        """Returns the total number of samples currently in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a standardized sample-label pair.

        The image is converted to a PIL object to ensure compatibility with
        Torchvision V1 transforms before being returned as a PyTorch Tensor.
        """
        img = self.images[idx]
        label = self.labels[idx]

        pil_img = Image.fromarray(img.squeeze() if img.shape[-1] == 1 else img)

        if self.transform:
            img = self.transform(pil_img)
        else:
            img = transforms.functional.to_tensor(pil_img)

        return img, torch.tensor(label, dtype=torch.long)
