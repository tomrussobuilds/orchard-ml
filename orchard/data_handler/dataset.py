"""
PyTorch Dataset Definition Module.

This module contains the custom Dataset class for NPZ-based vision datasets,
handling the conversion from NumPy arrays to PyTorch tensors and applying
image transformations for training and inference.

It supports two loading strategies via classmethod factories:

- ``from_npz``: Eager loading into RAM with transforms, subsampling, and PIL conversion.
- ``lazy``: Memory-mapped loading for large datasets or lightweight health checks.

Key Components:
    VisionDataset: Full-featured dataset with eager and lazy loading modes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# DATASET CLASS
class VisionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    PyTorch Dataset for NPZ-based vision data.

    The constructor accepts raw NumPy arrays directly (no I/O).
    Use the classmethod factories to load from disk:

    - ``VisionDataset.from_npz(...)`` — eager, full split into RAM.
    - ``VisionDataset.lazy(...)`` — memory-mapped, pages loaded on demand.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        *,
        transform: transforms.Compose | None = None,
    ) -> None:
        """
        Initializes the dataset from pre-loaded arrays.

        Args:
            images: Image array with shape ``(N, H, W)`` or ``(N, H, W, C)``.
            labels: Label array, any shape that flattens to ``(N,)``.
            transform: Pipeline of Torchvision transforms.
        """
        # Ensure consistent (N, H, W, C) for PIL conversion
        if images.ndim == 3:  # (N, H, W) -> (N, H, W, 1)
            images = np.expand_dims(images, axis=-1)

        self.images = images
        self.labels: np.ndarray = labels.ravel().astype(np.int64)
        self.transform = transform

        # Kept alive to prevent GC of mmap arrays (set by .lazy())
        self._npz_handle: np.lib.npyio.NpzFile | None = None
        # Index mapping for lazy subsampling (None = use all)
        self._indices: np.ndarray | None = None

    @classmethod
    def from_npz(
        cls,
        path: Path,
        split: str = "train",
        *,
        transform: transforms.Compose | None = None,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> VisionDataset:
        """
        Eagerly load a split from an NPZ archive into RAM.

        Args:
            path: Path to the dataset ``.npz`` archive.
            split: Dataset split to load (``train``, ``val``, or ``test``).
            transform: Pipeline of Torchvision transforms.
            max_samples: If set, limits the number of samples (subsampling).
            seed: Random seed for deterministic subsampling.
        """
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {path}")

        with np.load(path) as data:
            raw_images = data[f"{split}_images"]
            raw_labels = data[f"{split}_labels"]

            total_available = len(raw_labels)

            # Deterministic subsampling logic
            if max_samples and max_samples < total_available:
                rng = np.random.default_rng(seed)
                chosen = rng.choice(total_available, size=max_samples, replace=False)
                images = raw_images[chosen]
                labels = raw_labels[chosen]
            else:
                images = np.array(raw_images)
                labels = raw_labels

        return cls(images, labels, transform=transform)

    @classmethod
    def lazy(
        cls,
        path: Path,
        split: str = "train",
        *,
        transform: transforms.Compose | None = None,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> VisionDataset:
        """
        Memory-mapped load from an NPZ archive (no full RAM copy).

        Images are loaded page-by-page on demand. Suitable for large datasets
        that do not fit in RAM and for lightweight health checks.

        Args:
            path: Path to the ``.npz`` file.
            split: Dataset split to load (default ``train``).
            transform: Pipeline of Torchvision transforms.
            max_samples: If set, limits the number of samples (subsampling).
            seed: Random seed for deterministic subsampling.
        """
        data = np.load(path, mmap_mode="r")
        instance = cls(data[f"{split}_images"], data[f"{split}_labels"], transform=transform)
        instance._npz_handle = data

        if max_samples and max_samples < len(instance.labels):
            rng = np.random.default_rng(seed)
            instance._indices = rng.choice(len(instance.labels), size=max_samples, replace=False)
            # Eagerly subsample labels (small) so .labels and __len__ stay consistent
            instance.labels = instance.labels[instance._indices]

        return instance

    def __len__(self) -> int:
        """Returns the total number of samples currently in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a standardized sample-label pair.

        The image is converted to a PIL object to ensure compatibility with
        Torchvision V2 transforms before being returned as a PyTorch Tensor.

        Args:
            idx: Sample index.

        Returns:
            A pair of (image, label) where image is a ``(C, H, W)`` float
            tensor and label is a scalar long tensor.
        """
        # Remap index for lazy subsampling (images stay full mmap)
        img_idx = self._indices[idx] if self._indices is not None else idx
        img = self.images[img_idx]

        pil_img = Image.fromarray(img.squeeze() if img.shape[-1] == 1 else img)

        if self.transform:
            img_t = self.transform(pil_img)
        else:
            img_t = transforms.functional.to_tensor(pil_img)

        return img_t, torch.tensor(int(self.labels[idx]), dtype=torch.long)
