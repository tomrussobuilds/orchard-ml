"""
Galaxy10 DECals Dataset Converter.

Downloads and converts Galaxy10 DECals HDF5 dataset to NPZ format
compatible with the Orchard ML pipeline. Creates train/val/test splits.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import requests
from PIL import Image

from ...core.logger.styles import LogStyle
from ...core.metadata import DatasetMetadata
from ...core.paths import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


def download_galaxy10_h5(
    url: str,
    target_h5: Path,
    retries: int = 3,
    timeout: int = 600,
    chunk_size: int = 8192,
) -> None:
    """
    Downloads Galaxy10 HDF5 file with retry logic.

    Args:
        url: Download URL
        target_h5: Path to save HDF5 file
        retries: Number of download attempts
        timeout: Download timeout in seconds
        chunk_size: Streaming chunk size in bytes (default: TelemetryConfig.io_chunk_size).
    """
    if target_h5.exists():
        logger.info(f"{LogStyle.INDENT}{LogStyle.ARROW} {'HDF5 Cache':<18}: {target_h5.name}")
        return

    target_h5.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_h5.with_suffix(".tmp")

    for attempt in range(1, retries + 1):
        try:
            logger.info(
                f"{LogStyle.INDENT}{LogStyle.ARROW} {'Downloading':<18}: "
                f"Galaxy10 (attempt {attempt}/{retries})"
            )

            with requests.get(url, timeout=timeout, stream=True) as r:
                r.raise_for_status()

                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)

            tmp_path.replace(target_h5)
            logger.info(f"{LogStyle.INDENT}{LogStyle.SUCCESS} {'Downloaded':<18}: {target_h5.name}")
            return

        except OSError as e:
            if tmp_path.exists():
                tmp_path.unlink()

            if attempt == retries:
                raise RuntimeError(f"Failed to download Galaxy10 after {retries} attempts") from e

            logger.warning(f"Download attempt {attempt} failed: {e}")

    raise RuntimeError("Unexpected error in Galaxy10 download")  # pragma: no cover


def convert_galaxy10_to_npz(
    h5_path: Path,
    output_npz: Path,
    target_size: int = 224,
    seed: int = 42,
) -> None:
    """
    Converts Galaxy10 HDF5 to NPZ format with train/val/test splits.

    Args:
        h5_path: Path to downloaded HDF5 file
        output_npz: Path for output NPZ file
        target_size: Target image size (default 224)
        seed: Random seed for splits
    """
    logger.info(
        f"{LogStyle.INDENT}{LogStyle.ARROW} {'Converting':<18}: "
        f"Galaxy10 → NPZ ({target_size}x{target_size})"
    )

    with h5py.File(h5_path, "r") as f:
        images = np.array(f["images"])
        labels = np.array(f["ans"])

        logger.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Loaded':<18}: "
            f"{len(images)} images ({images.shape})"
        )

        # Resize if needed
        if images.shape[1] != target_size or images.shape[2] != target_size:
            logger.info(
                f"{LogStyle.INDENT}{LogStyle.ARROW} {'Resizing':<18}: "
                f"{images.shape[1]}x{images.shape[2]} → {target_size}x{target_size}"
            )
            resized_images = []

            for img in images:
                pil_img = Image.fromarray(img.astype(np.uint8))
                # Use Image.Resampling.BILINEAR for Pillow >= 10.0.0
                resample = getattr(Image, "Resampling", Image).BILINEAR
                pil_img = pil_img.resize((target_size, target_size), resample)
                resized_images.append(np.array(pil_img))

            images = np.array(resized_images, dtype=np.uint8)

        labels = labels.astype(np.int64).reshape(-1, 1)

        # Create splits (70/15/15)
        train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels = _create_splits(
            images, labels, seed=seed
        )

        # Save as NPZ
        np.savez_compressed(
            output_npz,
            train_images=train_imgs,
            train_labels=train_labels,
            val_images=val_imgs,
            val_labels=val_labels,
            test_images=test_imgs,
            test_labels=test_labels,
        )

        logger.info(f"{LogStyle.INDENT}{LogStyle.SUCCESS} {'NPZ Created':<18}: {output_npz.name}")
        logger.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Splits':<18}: "
            f"Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}"
        )


def _create_splits(
    images: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates stratified train/val/test splits.

    Args:
        images: All images (N, H, W, C)
        labels: All labels (N, 1)
        seed: Random seed
        train_ratio: Training split ratio
        val_ratio: Validation split ratio

    Returns:
        tuple of (train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels)
    """
    rng = np.random.default_rng(seed)

    train_imgs_list = []
    train_labels_list = []
    val_imgs_list = []
    val_labels_list = []
    test_imgs_list = []
    test_labels_list = []

    unique_labels = np.unique(labels)

    for label in unique_labels:
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)

        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        train_imgs_list.append(images[train_idx])
        train_labels_list.append(labels[train_idx])
        val_imgs_list.append(images[val_idx])
        val_labels_list.append(labels[val_idx])
        test_imgs_list.append(images[test_idx])
        test_labels_list.append(labels[test_idx])

    # Concatenate and shuffle
    train_imgs = np.concatenate(train_imgs_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)
    val_imgs = np.concatenate(val_imgs_list, axis=0)
    val_labels = np.concatenate(val_labels_list, axis=0)
    test_imgs = np.concatenate(test_imgs_list, axis=0)
    test_labels = np.concatenate(test_labels_list, axis=0)

    train_perm = rng.permutation(len(train_imgs))
    val_perm = rng.permutation(len(val_imgs))
    test_perm = rng.permutation(len(test_imgs))

    return (
        train_imgs[train_perm],
        train_labels[train_perm],
        val_imgs[val_perm],
        val_labels[val_perm],
        test_imgs[test_perm],
        test_labels[test_perm],
    )


def ensure_galaxy10_npz(metadata: DatasetMetadata) -> Path:
    """
    Ensures Galaxy10 is downloaded and converted to NPZ format.

    Args:
        metadata: DatasetMetadata with URL and path

    Returns:
        Path to validated NPZ file
    """
    from ...core import md5_checksum

    target_npz = metadata.path

    # Check if NPZ already exists
    if target_npz.exists():
        actual_md5 = md5_checksum(target_npz)
        if (
            actual_md5 == metadata.md5_checksum
            or metadata.md5_checksum == "placeholder_will_be_calculated_after_conversion"
        ):
            logger.debug(
                f"{LogStyle.INDENT}{LogStyle.ARROW} {'Dataset':<18}: "
                f"Galaxy10 found at {target_npz.name}"
            )
            return target_npz
        else:
            logger.warning("Galaxy10 NPZ MD5 mismatch, regenerating...")
            target_npz.unlink()

    # Download HDF5
    h5_path = target_npz.parent / "Galaxy10_DECals.h5"
    download_galaxy10_h5(metadata.url, h5_path)

    # Convert to NPZ
    target_size = metadata.native_resolution or 224
    convert_galaxy10_to_npz(
        h5_path=h5_path,
        output_npz=target_npz,
        target_size=target_size,
    )

    # Report MD5
    actual_md5 = md5_checksum(target_npz)
    logger.info(f"{LogStyle.INDENT}{LogStyle.ARROW} {'MD5':<18}: {actual_md5}")

    if metadata.md5_checksum == "placeholder_will_be_calculated_after_conversion":
        logger.info(
            f"{LogStyle.INDENT}{LogStyle.ARROW} {'Action Required':<18}: "
            f'Update metadata.md5_checksum = "{actual_md5}"'
        )

    return target_npz
