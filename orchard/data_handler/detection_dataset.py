"""
PyTorch Dataset for Object Detection.

Wraps image arrays and bounding-box annotations into the format expected
by torchvision detection models: ``(image_tensor, target_dict)`` where
``target_dict`` contains ``boxes`` and ``labels`` tensors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..core.paths import DEFAULT_SEED
from ..exceptions import OrchardDatasetError


class DetectionDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    """
    PyTorch Dataset for detection tasks with bounding-box annotations.

    Each sample returns an ``(image, target)`` pair where ``target`` is a
    dict with ``boxes`` (N, 4) in ``[x1, y1, x2, y2]`` format and
    ``labels`` (N,) as int64 class indices.
    """

    def __init__(
        self,
        images: npt.NDArray[Any],
        annotations: list[dict[str, npt.NDArray[Any]]],
        *,
        transform: transforms.Compose | None = None,
    ) -> None:
        """
        Initialize from pre-loaded arrays and annotation list.

        Args:
            images: Image array ``(N, H, W, C)`` or ``(N, H, W)``.
            annotations: Per-image annotation dicts with ``boxes`` and
                ``labels`` numpy arrays.
            transform: Torchvision transform pipeline for images.
        """
        if images.ndim == 3:
            images = np.expand_dims(images, axis=-1)

        self.images = images
        self.annotations = annotations
        self.transform = transform

    @classmethod
    def from_arrays(
        cls,
        images: npt.NDArray[Any],
        annotations: list[dict[str, npt.NDArray[Any]]],
        *,
        transform: transforms.Compose | None = None,
        max_samples: int | None = None,
        seed: int = DEFAULT_SEED,
    ) -> DetectionDataset:
        """
        Build a DetectionDataset with optional subsampling.

        Args:
            images: Image array ``(N, H, W, C)``.
            annotations: Per-image annotation dicts.
            transform: Transform pipeline.
            max_samples: Limit number of samples.
            seed: Random seed for deterministic subsampling.
        """
        if max_samples and max_samples < len(images):
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(images), size=max_samples, replace=False)
            images = images[indices]
            annotations = [annotations[i] for i in indices]

        return cls(images, annotations, transform=transform)

    @classmethod
    def from_npz(
        cls,
        image_path: Path,
        annotation_path: Path,
        split: str = "train",
        *,
        transform: transforms.Compose | None = None,
        max_samples: int | None = None,
        seed: int = DEFAULT_SEED,
    ) -> DetectionDataset:
        """
        Load a detection dataset from NPZ (images) and NPZ (annotations).

        The image NPZ has key ``{split}_images``. The annotation NPZ has
        keys ``{split}_boxes`` (list of (N_i, 4) arrays) and
        ``{split}_labels`` (list of (N_i,) arrays), stored as object arrays.

        Args:
            image_path: Path to images NPZ.
            annotation_path: Path to annotations NPZ.
            split: Dataset split (``train``, ``val``, ``test``).
            transform: Transform pipeline.
            max_samples: Limit number of samples.
            seed: Random seed.
        """
        if not image_path.exists():
            raise OrchardDatasetError(f"Image file not found: {image_path}")
        if not annotation_path.exists():
            raise OrchardDatasetError(f"Annotation file not found: {annotation_path}")

        with np.load(image_path) as img_data:
            images = np.array(img_data[f"{split}_images"])

        with np.load(annotation_path, allow_pickle=True) as ann_data:
            boxes_list = ann_data[f"{split}_boxes"]
            labels_list = ann_data[f"{split}_labels"]

        annotations: list[dict[str, npt.NDArray[Any]]] = [
            {"boxes": np.array(b, dtype=np.float32), "labels": np.array(lab, dtype=np.int64)}
            for b, lab in zip(boxes_list, labels_list)
        ]

        return cls.from_arrays(
            images, annotations, transform=transform, max_samples=max_samples, seed=seed
        )

    def __len__(self) -> int:
        """Returns the number of images in the dataset."""
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Retrieve an image and its bounding-box annotations.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor, target_dict) where target_dict has
            ``boxes`` (N, 4) float32 and ``labels`` (N,) int64.
        """
        img = self.images[idx]
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        pil_img = Image.fromarray(img)

        if self.transform:
            img_t: torch.Tensor = self.transform(pil_img)
        else:
            img_t = transforms.functional.to_tensor(pil_img)

        ann = self.annotations[idx]
        target = {
            "boxes": torch.as_tensor(ann["boxes"], dtype=torch.float32),
            "labels": torch.as_tensor(ann["labels"], dtype=torch.int64),
        }

        return img_t, target
