"""
PennFudan Pedestrian Detection Dataset Fetcher.

Downloads the PennFudan pedestrian dataset ZIP, extracts instance masks,
converts masks to bounding boxes, resizes images, and produces two NPZ
files (images + annotations) compatible with DetectionDataset.from_npz().
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import requests
from PIL import Image

from ...core.metadata import DatasetMetadata
from ...core.paths import DEFAULT_SEED, LOGGER_NAME
from ...core.paths.constants import LogStyle
from ...exceptions import OrchardDatasetError

logger = logging.getLogger(LOGGER_NAME)

_TARGET_SIZE = 224
_TRAIN_RATIO = 0.7
_VAL_RATIO = 0.15


def _download_zip(
    url: str,
    retries: int = 3,  # pragma: no mutate
    timeout: int = 120,  # pragma: no mutate
) -> zipfile.ZipFile:
    """
    Download a ZIP archive into memory with retry logic.

    Args:
        url: Download URL.
        retries: Number of attempts.
        timeout: Request timeout in seconds.

    Returns:
        In-memory ZipFile object.
    """
    for attempt in range(1, retries + 1):  # pragma: no mutate
        try:
            logger.info(
                "%s%s %-18s: PennFudan (attempt %d/%d)",
                LogStyle.INDENT,
                LogStyle.ARROW,
                "Downloading",
                attempt,
                retries,
            )
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return zipfile.ZipFile(io.BytesIO(resp.content))

        except OSError as e:
            if attempt == retries:
                raise OrchardDatasetError(
                    f"Failed to download PennFudan after {retries} attempts"
                ) from e
            logger.warning("Download attempt %d failed: %s", attempt, e)  # pragma: no mutate

    raise OrchardDatasetError(  # pragma: no cover  # pragma: no mutate
        "Unexpected error in PennFudan download"  # pragma: no mutate
    )


def _mask_to_boxes(mask: npt.NDArray[Any]) -> npt.NDArray[np.floating[Any]]:
    """
    Extract bounding boxes from an instance segmentation mask.

    Each unique non-zero pixel value represents one object instance.

    Args:
        mask: 2D array where each unique non-zero value is an instance.

    Returns:
        (N, 4) float32 array of [x1, y1, x2, y2] boxes.
    """
    instance_ids = np.unique(mask)
    instance_ids = instance_ids[instance_ids != 0]

    boxes = []
    for iid in instance_ids:
        positions = np.nonzero(mask == iid)
        y_min = int(np.min(positions[0]))
        y_max = int(np.max(positions[0]))
        x_min = int(np.min(positions[1]))
        x_max = int(np.max(positions[1]))
        if x_max > x_min and y_max > y_min:
            boxes.append([x_min, y_min, x_max, y_max])

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array(boxes, dtype=np.float32)


def _rescale_boxes(
    boxes: npt.NDArray[np.floating[Any]],
    orig_w: int,
    orig_h: int,
    target_size: int,
) -> npt.NDArray[np.floating[Any]]:
    """
    Rescale bounding boxes after image resize.

    Args:
        boxes: (N, 4) boxes in [x1, y1, x2, y2] format.
        orig_w: Original image width.
        orig_h: Original image height.
        target_size: Target square size.

    Returns:
        Rescaled (N, 4) float32 boxes.
    """
    if len(boxes) == 0:
        return boxes
    scale_x = target_size / orig_w
    scale_y = target_size / orig_h
    scaled: npt.NDArray[np.floating[Any]] = boxes.copy()
    scaled[:, 0] *= scale_x
    scaled[:, 2] *= scale_x
    scaled[:, 1] *= scale_y
    scaled[:, 3] *= scale_y
    return scaled


def _parse_pennfudan_zip(
    zf: zipfile.ZipFile,
    target_size: int = _TARGET_SIZE,
) -> tuple[npt.NDArray[Any], list[npt.NDArray[Any]], list[npt.NDArray[Any]]]:
    """
    Parse PennFudan ZIP: extract images, convert masks to boxes.

    Args:
        zf: Opened ZipFile.
        target_size: Target square resolution.

    Returns:
        Tuple of (images_array, boxes_list, labels_list).
    """
    image_files = sorted(
        n for n in zf.namelist() if n.startswith("PennFudanPed/PNGImages/") and n.endswith(".png")
    )
    mask_files = sorted(
        n for n in zf.namelist() if n.startswith("PennFudanPed/PedMasks/") and n.endswith(".png")
    )

    if len(image_files) != len(mask_files):
        raise OrchardDatasetError(
            f"PennFudan image/mask count mismatch: {len(image_files)} vs {len(mask_files)}"
        )

    resample = getattr(Image, "Resampling", Image).BILINEAR  # pragma: no mutate

    images = []
    all_boxes: list[npt.NDArray[Any]] = []
    all_labels: list[npt.NDArray[Any]] = []

    for img_name, mask_name in zip(image_files, mask_files):
        img = Image.open(io.BytesIO(zf.read(img_name))).convert("RGB")
        mask = np.array(Image.open(io.BytesIO(zf.read(mask_name))))

        orig_w, orig_h = img.size
        boxes = _mask_to_boxes(mask)
        boxes = _rescale_boxes(boxes, orig_w, orig_h, target_size)

        img_resized = img.resize((target_size, target_size), resample)  # pragma: no mutate
        images.append(np.array(img_resized, dtype=np.uint8))  # pragma: no mutate

        all_boxes.append(boxes)
        # All instances are class 1 (person); label 0 is reserved for background
        all_labels.append(np.ones(len(boxes), dtype=np.int64))

    images_array = np.array(images, dtype=np.uint8)  # pragma: no mutate
    return images_array, all_boxes, all_labels


def _split_indices(
    n: int,
    seed: int = DEFAULT_SEED,
    train_ratio: float = _TRAIN_RATIO,
    val_ratio: float = _VAL_RATIO,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """
    Create deterministic train/val/test index splits.

    Args:
        n: Total number of samples.
        seed: Random seed.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.

    Returns:
        Tuple of (train_indices, val_indices, test_indices).
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return indices[:n_train], indices[n_train : n_train + n_val], indices[n_train + n_val :]


def _save_detection_npz(
    images: npt.NDArray[Any],
    boxes_list: list[npt.NDArray[Any]],
    labels_list: list[npt.NDArray[Any]],
    image_path: Path,
    annotation_path: Path,
    seed: int = DEFAULT_SEED,
) -> None:
    """
    Save detection data as two NPZ files (images + annotations) with splits.

    Args:
        images: (N, H, W, C) uint8 array.
        boxes_list: Per-image bounding boxes.
        labels_list: Per-image labels.
        image_path: Output path for images NPZ.
        annotation_path: Output path for annotations NPZ.
        seed: Random seed for splits.
    """
    train_idx, val_idx, test_idx = _split_indices(len(images), seed)  # pragma: no mutate

    image_path.parent.mkdir(parents=True, exist_ok=True)  # pragma: no mutate

    np.savez_compressed(
        image_path,
        train_images=images[train_idx],
        val_images=images[val_idx],
        test_images=images[test_idx],
    )

    boxes_arr = np.array(boxes_list, dtype=object)
    labels_arr = np.array(labels_list, dtype=object)

    np.savez_compressed(
        annotation_path,
        train_boxes=boxes_arr[train_idx],
        train_labels=labels_arr[train_idx],
        val_boxes=boxes_arr[val_idx],
        val_labels=labels_arr[val_idx],
        test_boxes=boxes_arr[test_idx],
        test_labels=labels_arr[test_idx],
    )

    logger.info(
        "%s%s %-18s: Train: %d, Val: %d, Test: %d",
        LogStyle.INDENT,
        LogStyle.ARROW,
        "Splits",
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )


def ensure_pennfudan_npz(metadata: DatasetMetadata) -> Path:
    """
    Ensure PennFudan dataset is downloaded and converted to NPZ format.

    Args:
        metadata: DatasetMetadata with URL, path, and annotation_path.

    Returns:
        Path to the images NPZ file.
    """
    image_path = metadata.path
    annotation_path = metadata.annotation_path

    if annotation_path is None:
        raise OrchardDatasetError(
            "PennFudan metadata must have annotation_path set"  # pragma: no mutate
        )

    # Return cached if both NPZ files exist
    if image_path.exists() and annotation_path.exists():
        logger.info(
            "%s%s %-18s: PennFudan found at %s",
            LogStyle.INDENT,
            LogStyle.ARROW,
            "Dataset",
            image_path.name,
        )
        return image_path

    # Download and convert
    zf = _download_zip(metadata.url)
    images, boxes_list, labels_list = _parse_pennfudan_zip(zf)

    logger.info(
        "%s%s %-18s: %d images, %d total instances",
        LogStyle.INDENT,
        LogStyle.ARROW,
        "Parsed",
        len(images),
        sum(len(b) for b in boxes_list),
    )

    _save_detection_npz(images, boxes_list, labels_list, image_path, annotation_path)

    logger.info(
        "%s%s %-18s: %s + %s",
        LogStyle.INDENT,
        LogStyle.SUCCESS,
        "NPZ Created",
        image_path.name,
        annotation_path.name,
    )

    return image_path
