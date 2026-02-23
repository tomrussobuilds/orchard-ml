"""
PyTorch Dataset Definition Module.

This module contains the custom Dataset classes for NPZ-based vision datasets,
handling the conversion from NumPy arrays to PyTorch tensors and applying
image transformations for training and inference.

It implements selective RAM loading to balance I/O speed with memory
efficiency and ensures deterministic subsampling for reproducible research.

Key Components:
    VisionDataset: Full-featured dataset with transforms, subsampling, and PIL conversion.
    LazyNPZDataset: Memory-mapped dataset for lightweight health checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# DATASET CLASS
class VisionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Enhanced PyTorch Dataset for NPZ-based vision data.

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
        seed: int = 42,
    ) -> None:
        """
        Initializes the dataset by loading the specified .npz split into RAM.

        Args:
            path (Path): Path to the dataset .npz archive.
            split (str): Dataset split to load ('train', 'val', or 'test').
            transform (transforms.Compose | None): Pipeline of Torchvision transforms.
            max_samples (int | None): If set, limits the number of samples (subsampling).
            seed (int): Random seed for deterministic subsampling.
        """
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {path}")

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
                rng = np.random.default_rng(seed)
                chosen_indices = rng.choice(total_available, size=max_samples, replace=False)
                self.images = raw_images[chosen_indices]
                self.labels = raw_labels[chosen_indices]
            else:
                self.images = np.array(raw_images)
                self.labels = raw_labels

            # This ensures consistent PIL conversion regardless of source format
            if self.images.ndim == 3:  # (N, H, W) -> (N, H, W, 1)
                self.images = np.expand_dims(self.images, axis=-1)

    def __len__(self) -> int:
        """Returns the total number of samples currently in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a standardized sample-label pair.

        The image is converted to a PIL object to ensure compatibility with
        Torchvision V2 transforms before being returned as a PyTorch Tensor.

        Args:
            idx (int): Sample index.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A pair of (image, label) where
                image is a ``(C, H, W)`` float tensor and label is a scalar long tensor.
        """
        img = self.images[idx]
        label = self.labels[idx]

        pil_img = Image.fromarray(img.squeeze() if img.shape[-1] == 1 else img)

        if self.transform:
            img = self.transform(pil_img)
        else:
            img = transforms.functional.to_tensor(pil_img)

        return img, torch.tensor(label, dtype=torch.long)


class LazyNPZDataset(Dataset):
    """Memory-mapped PyTorch Dataset for lazy loading from ``.npz`` files.

    Uses NumPy memory-mapping (``mmap_mode="r"``) to avoid loading the entire
    archive into RAM, making it suitable for large-scale health checks and
    quick data integrity verification.

    Attributes:
        npz_path (Path): Filesystem path to the ``.npz`` archive.
        images (np.ndarray): Memory-mapped view of ``train_images``.
        labels (np.ndarray): Memory-mapped view of ``train_labels``.
    """

    def __init__(self, npz_path: Path) -> None:
        """Initializes the dataset with a memory-mapped ``.npz`` archive.

        Args:
            npz_path: Path to the ``.npz`` file containing
                ``train_images`` and ``train_labels`` arrays.
        """
        self.npz_path = npz_path
        self.data = np.load(npz_path, mmap_mode="r")
        self.images = self.data["train_images"]
        self.labels = self.data["train_labels"]

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Retrieves a normalized (C, H, W) tensor and its integer label.

        Args:
            idx: Sample index.

        Returns:
            A tuple of (image_tensor, label) where the image is scaled to [0, 1].

        Raises:
            ValueError: If the image has an unexpected number of dimensions.
        """
        img = self.images[idx]
        # Ensure channel dimension (C,H,W)
        if img.ndim == 2:  # (H,W) grayscale
            img = np.expand_dims(img, axis=0)
        elif img.ndim == 3:  # (H,W,C)
            img = np.transpose(img, (2, 0, 1))
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")

        img = torch.from_numpy(img).float() / 255.0
        label = int(self.labels[idx][0])
        return img, label
