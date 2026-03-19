"""
Synthetic Detection Data for Testing.

Generates random images with bounding-box annotations for detection
task unit tests. Follows the same pattern as
:func:`create_synthetic_dataset` for classification.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ...core.paths import DEFAULT_SEED, MIN_SPLIT_SAMPLES

_SYNTHETIC_SEED = DEFAULT_SEED  # pragma: no mutate
_SYNTHETIC_PIXEL_RANGE = 255  # pragma: no mutate
_MIN_BOXES = 1  # pragma: no mutate
_MAX_BOXES = 5  # pragma: no mutate
_MIN_BOX_SIZE = 4  # pragma: no mutate


def _random_boxes(
    rng: np.random.Generator,
    num_boxes: int,
    img_size: int,
) -> npt.NDArray[np.float32]:
    """
    Generate random bounding boxes in [x1, y1, x2, y2] format.

    Args:
        rng: Seeded random generator.
        num_boxes: Number of boxes to generate.
        img_size: Image dimension (boxes clamped to this).

    Returns:
        Array of shape ``(num_boxes, 4)`` with valid box coordinates.
    """
    x1 = rng.integers(0, img_size - _MIN_BOX_SIZE, size=num_boxes)  # pragma: no mutate
    y1 = rng.integers(0, img_size - _MIN_BOX_SIZE, size=num_boxes)  # pragma: no mutate
    x2 = rng.integers(x1 + _MIN_BOX_SIZE, img_size, size=num_boxes)  # pragma: no mutate
    y2 = rng.integers(y1 + _MIN_BOX_SIZE, img_size, size=num_boxes)  # pragma: no mutate
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)  # pragma: no mutate


def _generate_split(
    rng: np.random.Generator,
    n_samples: int,
    resolution: int,
    channels: int,
    num_classes: int,
) -> tuple[npt.NDArray[Any], list[npt.NDArray[np.float32]], list[npt.NDArray[np.int64]]]:
    """
    Generate images, boxes, and labels for one split.

    Returns:
        Tuple of (images, boxes_list, labels_list).
    """
    images = rng.integers(  # pragma: no mutate
        0,
        _SYNTHETIC_PIXEL_RANGE,  # pragma: no mutate
        (n_samples, resolution, resolution, channels),  # pragma: no mutate
        dtype=np.uint8,  # pragma: no mutate
    )

    boxes_list: list[npt.NDArray[np.float32]] = []
    labels_list: list[npt.NDArray[np.int64]] = []

    for _ in range(n_samples):
        n_boxes = int(rng.integers(_MIN_BOXES, _MAX_BOXES + 1))  # pragma: no mutate
        boxes = _random_boxes(rng, n_boxes, resolution)
        labels = rng.integers(1, num_classes + 1, size=n_boxes).astype(
            np.int64
        )  # pragma: no mutate
        boxes_list.append(boxes)
        labels_list.append(labels)

    return images, boxes_list, labels_list


class SyntheticDetectionData:
    """
    Container for synthetic detection dataset paths and metadata.

    Attributes:
        image_path (Path): Path to images NPZ.
        annotation_path (Path): Path to annotations NPZ.
        num_classes (int): Number of object classes (excluding background).
        name (str): Dataset identifier.
    """

    def __init__(
        self,
        image_path: Path,
        annotation_path: Path,
        num_classes: int,
        name: str,
    ) -> None:
        self.image_path = image_path
        self.annotation_path = annotation_path
        self.num_classes = num_classes
        self.name = name


def create_synthetic_detection_dataset(
    num_classes: int = 4,  # pragma: no mutate
    samples: int = 50,  # pragma: no mutate
    resolution: int = 64,  # pragma: no mutate
    channels: int = 3,  # pragma: no mutate
    name: str = "synthetic_detection",  # pragma: no mutate
) -> SyntheticDetectionData:
    """
    Create a synthetic detection dataset for testing.

    Generates random images with random bounding boxes and saves them
    as NPZ files (images + annotations separately).

    Args:
        num_classes: Number of object categories (default: 4).
        samples: Number of training images (default: 50).
        resolution: Image size in pixels (default: 64).
        channels: Color channels (default: 3).
        name: Dataset identifier (default: "synthetic_detection").

    Returns:
        SyntheticDetectionData with paths to generated NPZ files.
    """
    rng = np.random.default_rng(_SYNTHETIC_SEED)  # pragma: no mutate

    train_imgs, train_boxes, train_labels = _generate_split(
        rng, samples, resolution, channels, num_classes
    )
    val_samples = max(MIN_SPLIT_SAMPLES, samples // 10)  # pragma: no mutate
    test_samples = max(MIN_SPLIT_SAMPLES, samples // 10)  # pragma: no mutate

    val_imgs, val_boxes, val_labels = _generate_split(
        rng, val_samples, resolution, channels, num_classes
    )
    test_imgs, test_boxes, test_labels = _generate_split(
        rng, test_samples, resolution, channels, num_classes
    )

    # Save images NPZ
    img_file = tempfile.NamedTemporaryFile(  # pragma: no mutate
        suffix=".npz", delete=False, prefix="det_images_"  # pragma: no mutate
    )
    img_path = Path(img_file.name)
    img_file.close()
    np.savez(
        img_path,
        train_images=train_imgs,
        val_images=val_imgs,
        test_images=test_imgs,
    )

    # Save annotations NPZ (object arrays for variable-length boxes)
    ann_file = tempfile.NamedTemporaryFile(  # pragma: no mutate
        suffix=".npz", delete=False, prefix="det_annotations_"  # pragma: no mutate
    )
    ann_path = Path(ann_file.name)
    ann_file.close()

    def _to_object_array(lst: list[npt.NDArray[Any]]) -> npt.NDArray[Any]:
        arr = np.empty(len(lst), dtype=object)
        for i, v in enumerate(lst):
            arr[i] = v
        return arr

    np.savez(
        ann_path,
        train_boxes=_to_object_array(train_boxes),
        train_labels=_to_object_array(train_labels),
        val_boxes=_to_object_array(val_boxes),
        val_labels=_to_object_array(val_labels),
        test_boxes=_to_object_array(test_boxes),
        test_labels=_to_object_array(test_labels),
    )

    return SyntheticDetectionData(
        image_path=img_path,
        annotation_path=ann_path,
        num_classes=num_classes,
        name=name,
    )
