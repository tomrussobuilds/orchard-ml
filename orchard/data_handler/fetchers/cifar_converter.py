"""
CIFAR-10/100 Dataset Converter

Downloads CIFAR datasets via torchvision and converts them to NPZ format
compatible with the Orchard ML pipeline. Creates stratified train/val/test splits
from the original train/test partition.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np

from ...core.paths import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


def _create_stratified_split(
    images: np.ndarray,
    labels: np.ndarray,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits training data into train and validation sets using stratified sampling.

    Args:
        images: Training images (N, H, W, C)
        labels: Training labels (N,)
        val_ratio: Fraction of training data for validation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_images, train_labels, val_images, val_labels)
    """
    rng = np.random.default_rng(seed)

    train_imgs_list = []
    train_labels_list = []
    val_imgs_list = []
    val_labels_list = []

    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)

        n_val = max(1, int(len(indices) * val_ratio))

        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        train_imgs_list.append(images[train_idx])
        train_labels_list.append(labels[train_idx])
        val_imgs_list.append(images[val_idx])
        val_labels_list.append(labels[val_idx])

    # Concatenate and shuffle
    train_imgs = np.concatenate(train_imgs_list)
    train_labels = np.concatenate(train_labels_list)
    val_imgs = np.concatenate(val_imgs_list)
    val_labels = np.concatenate(val_labels_list)

    train_perm = rng.permutation(len(train_imgs))
    val_perm = rng.permutation(len(val_imgs))

    return (
        train_imgs[train_perm],
        train_labels[train_perm],
        val_imgs[val_perm],
        val_labels[val_perm],
    )


def _download_and_convert(metadata, cifar_cls) -> Path:
    """
    Downloads a CIFAR dataset via torchvision and converts to NPZ.

    Args:
        metadata: DatasetMetadata with path and name
        cifar_cls: torchvision dataset class (CIFAR10 or CIFAR100)

    Returns:
        Path to the generated NPZ file
    """
    target_npz = metadata.path
    download_dir = target_npz.parent / f".{metadata.name}_raw"

    logger.info(f"Downloading {metadata.display_name} via torchvision...")

    train_ds = cifar_cls(root=str(download_dir), train=True, download=True)
    test_ds = cifar_cls(root=str(download_dir), train=False, download=True)

    # Extract arrays (torchvision provides HWC uint8 numpy arrays)
    train_images = np.array(train_ds.data)  # (50000, 32, 32, 3)
    train_targets = np.array(train_ds.targets)  # (50000,)
    test_images = np.array(test_ds.data)  # (10000, 32, 32, 3)
    test_targets = np.array(test_ds.targets)  # (10000,)

    logger.info(
        f"Loaded {metadata.display_name}: "
        f"{len(train_images)} train + {len(test_images)} test images"
    )

    # Create stratified train/val split
    train_imgs, train_labels, val_imgs, val_labels = _create_stratified_split(
        train_images, train_targets
    )

    # Reshape labels to (N, 1) for NPZ format compatibility
    train_labels = train_labels.astype(np.int64).reshape(-1, 1)
    val_labels = val_labels.astype(np.int64).reshape(-1, 1)
    test_labels = test_targets.astype(np.int64).reshape(-1, 1)

    # Save as compressed NPZ
    target_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target_npz,
        train_images=train_imgs,
        train_labels=train_labels,
        val_images=val_imgs,
        val_labels=val_labels,
        test_images=test_images,
        test_labels=test_labels,
    )

    logger.info(
        f"{metadata.display_name} NPZ created: {target_npz} — "
        f"Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_images)}"
    )

    return target_npz


def ensure_cifar_npz(metadata) -> Path:
    """
    Ensures a CIFAR dataset is downloaded and converted to NPZ format.

    Supports both CIFAR-10 and CIFAR-100 via metadata.name routing.

    Args:
        metadata: DatasetMetadata with name ('cifar10' or 'cifar100') and path

    Returns:
        Path to validated NPZ file
    """
    target_npz = metadata.path

    if target_npz.exists():
        logger.info(f"{metadata.display_name} NPZ found: {target_npz}")
        return target_npz

    from torchvision.datasets import CIFAR10, CIFAR100

    if metadata.name == "cifar100":
        cifar_cls = CIFAR100
    else:
        cifar_cls = CIFAR10

    return _download_and_convert(metadata, cifar_cls)
