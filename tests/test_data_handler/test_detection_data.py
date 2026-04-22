"""
Test Suite for Detection Data Loading.

Tests for detection collate function, DetectionDataset, and synthetic
detection data generation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import torch

from orchard.data_handler.collate import detection_collate_fn
from orchard.data_handler.detection_dataset import DetectionDataset
from orchard.data_handler.diagnostic.synthetic_detection import (
    SyntheticDetectionData,
    create_synthetic_detection_dataset,
)

# ── detection_collate_fn ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_collate_returns_lists() -> None:
    """Collate returns (list[Tensor], list[dict]) not stacked tensors."""
    batch: list[tuple[torch.Tensor, dict[str, Any]]] = [
        (
            torch.randn(3, 64, 64),
            {"boxes": torch.rand(2, 4), "labels": torch.tensor([1, 2])},
        ),
        (
            torch.randn(3, 64, 64),
            {"boxes": torch.rand(1, 4), "labels": torch.tensor([3])},
        ),
    ]

    images, targets = detection_collate_fn(batch)

    assert isinstance(images, list)
    assert isinstance(targets, list)
    assert len(images) == 2
    assert len(targets) == 2


@pytest.mark.unit
def test_collate_preserves_tensor_shapes() -> None:
    """Each image in the list keeps its original shape."""
    batch: list[tuple[torch.Tensor, dict[str, Any]]] = [
        (torch.randn(3, 32, 32), {"boxes": torch.rand(1, 4), "labels": torch.tensor([1])}),
        (torch.randn(3, 64, 64), {"boxes": torch.rand(2, 4), "labels": torch.tensor([1, 2])}),
    ]

    images, targets = detection_collate_fn(batch)

    assert images[0].shape == (3, 32, 32)
    assert images[1].shape == (3, 64, 64)


@pytest.mark.unit
def test_collate_preserves_target_dicts() -> None:
    """Target dicts are passed through unchanged."""
    boxes = torch.rand(3, 4)
    labels = torch.tensor([1, 2, 3])
    batch: list[tuple[torch.Tensor, dict[str, Any]]] = [
        (torch.randn(3, 64, 64), {"boxes": boxes, "labels": labels}),
    ]

    _, targets = detection_collate_fn(batch)

    assert torch.equal(targets[0]["boxes"], boxes)
    assert torch.equal(targets[0]["labels"], labels)


@pytest.mark.unit
def test_collate_empty_batch() -> None:
    """Collate handles empty batch."""
    images, targets = detection_collate_fn([])
    assert images == []
    assert targets == []


@pytest.mark.unit
def test_collate_single_item() -> None:
    """Collate handles single-item batch."""
    batch: list[tuple[torch.Tensor, dict[str, Any]]] = [
        (torch.randn(3, 64, 64), {"boxes": torch.rand(1, 4), "labels": torch.tensor([1])}),
    ]

    images, targets = detection_collate_fn(batch)
    assert len(images) == 1
    assert len(targets) == 1


# ── create_synthetic_detection_dataset ───────────────────────────────────────


@pytest.mark.unit
def test_synthetic_returns_data_object() -> None:
    """Factory returns a SyntheticDetectionData with valid paths."""
    data = create_synthetic_detection_dataset(num_classes=2, samples=20, resolution=32)

    assert isinstance(data, SyntheticDetectionData)
    assert data.image_path.exists()
    assert data.annotation_path.exists()
    assert data.num_classes == 2
    assert data.name == "synthetic_detection"


@pytest.mark.unit
def test_synthetic_images_shape() -> None:
    """Generated images have correct shape."""
    data = create_synthetic_detection_dataset(num_classes=2, samples=15, resolution=32, channels=3)

    with np.load(data.image_path) as f:
        train = f["train_images"]
        assert train.shape[0] == 15
        assert train.shape[1:3] == (32, 32)
        assert train.shape[3] == 3


@pytest.mark.unit
def test_synthetic_annotations_structure() -> None:
    """Generated annotations have boxes and labels for each image."""
    data = create_synthetic_detection_dataset(num_classes=3, samples=10, resolution=32)

    with np.load(data.annotation_path, allow_pickle=True) as f:
        boxes = f["train_boxes"]
        labels = f["train_labels"]

        assert len(boxes) == 10
        assert len(labels) == 10

        # Each annotation has valid structure
        for b, lab in zip(boxes, labels):
            assert b.ndim == 2
            assert b.shape[1] == 4
            assert len(lab) == len(b)
            # Labels are in [1, num_classes]
            assert all(1 <= x <= 3 for x in lab)


@pytest.mark.unit
def test_synthetic_boxes_valid_coordinates() -> None:
    """Generated boxes have x2 > x1 and y2 > y1."""
    data = create_synthetic_detection_dataset(num_classes=2, samples=20, resolution=64)

    with np.load(data.annotation_path, allow_pickle=True) as f:
        for boxes in f["train_boxes"]:
            assert np.all(boxes[:, 2] > boxes[:, 0]), "x2 must be > x1"
            assert np.all(boxes[:, 3] > boxes[:, 1]), "y2 must be > y1"


@pytest.mark.unit
def test_synthetic_has_val_and_test_splits() -> None:
    """Synthetic dataset generates val and test splits."""
    data = create_synthetic_detection_dataset(num_classes=2, samples=50)

    with np.load(data.image_path) as f:
        assert "val_images" in f
        assert "test_images" in f

    with np.load(data.annotation_path, allow_pickle=True) as f:
        assert "val_boxes" in f
        assert "test_labels" in f


@pytest.mark.unit
def test_synthetic_test_split_images_have_valid_shape() -> None:
    """Test split images are arrays with correct spatial dimensions."""
    data = create_synthetic_detection_dataset(num_classes=2, samples=20, resolution=32)

    with np.load(data.image_path) as f:
        test_imgs = f["test_images"]
        assert test_imgs.ndim == 4
        assert test_imgs.shape[1:3] == (32, 32)


@pytest.mark.unit
def test_synthetic_test_split_annotations_have_valid_structure() -> None:
    """Test split boxes and labels are non-empty object arrays."""
    data = create_synthetic_detection_dataset(num_classes=2, samples=20, resolution=32)

    with np.load(data.annotation_path, allow_pickle=True) as f:
        test_boxes = f["test_boxes"]
        test_labels = f["test_labels"]
        assert len(test_boxes) > 0
        assert len(test_labels) > 0
        for b, lab in zip(test_boxes, test_labels):
            assert b.ndim == 2
            assert b.shape[1] == 4
            assert len(lab) == len(b)


# ── DetectionDataset ─────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_det_data() -> SyntheticDetectionData:
    """Shared synthetic detection data fixture."""
    return create_synthetic_detection_dataset(num_classes=3, samples=20, resolution=32)


@pytest.mark.unit
def test_dataset_from_arrays() -> None:
    """DetectionDataset.from_arrays works with numpy data."""
    images = np.random.randint(0, 255, (5, 32, 32, 3), dtype=np.uint8)
    annotations: list[dict[str, npt.NDArray[Any]]] = [
        {
            "boxes": np.array([[10, 10, 20, 20]], dtype=np.float32),
            "labels": np.array([1], dtype=np.int64),
        }
        for _ in range(5)
    ]

    ds = DetectionDataset.from_arrays(images, annotations)
    assert len(ds) == 5


@pytest.mark.unit
def test_dataset_getitem_returns_tuple() -> None:
    """__getitem__ returns (tensor, dict) pair."""
    images = np.random.randint(0, 255, (3, 32, 32, 3), dtype=np.uint8)
    annotations: list[dict[str, npt.NDArray[Any]]] = [
        {
            "boxes": np.array([[5, 5, 25, 25]], dtype=np.float32),
            "labels": np.array([1], dtype=np.int64),
        }
        for _ in range(3)
    ]

    ds = DetectionDataset.from_arrays(images, annotations)
    img, target = ds[0]

    assert isinstance(img, torch.Tensor)
    assert img.shape[0] == 3  # CHW
    assert isinstance(target, dict)
    assert "boxes" in target
    assert "labels" in target
    assert target["boxes"].dtype == torch.float32
    assert target["labels"].dtype == torch.int64


@pytest.mark.unit
def test_dataset_from_npz(synthetic_det_data: SyntheticDetectionData) -> None:
    """DetectionDataset.from_npz loads from generated NPZ files."""
    ds = DetectionDataset.from_npz(
        synthetic_det_data.image_path,
        synthetic_det_data.annotation_path,
        split="train",
    )

    assert len(ds) == 20
    img, target = ds[0]
    assert isinstance(img, torch.Tensor)
    assert "boxes" in target


@pytest.mark.unit
def test_dataset_from_npz_val_split(synthetic_det_data: SyntheticDetectionData) -> None:
    """DetectionDataset.from_npz loads val split."""
    ds = DetectionDataset.from_npz(
        synthetic_det_data.image_path,
        synthetic_det_data.annotation_path,
        split="val",
    )

    assert len(ds) > 0


@pytest.mark.unit
def test_dataset_getitem_with_transform() -> None:
    """__getitem__ applies transform when provided."""
    from torchvision import transforms

    images = np.random.randint(0, 255, (3, 32, 32, 3), dtype=np.uint8)
    annotations: list[dict[str, npt.NDArray[Any]]] = [
        {
            "boxes": np.array([[5, 5, 25, 25]], dtype=np.float32),
            "labels": np.array([1], dtype=np.int64),
        }
        for _ in range(3)
    ]

    transform = transforms.Compose([transforms.ToTensor()])
    ds = DetectionDataset.from_arrays(images, annotations, transform=transform)
    img, target = ds[0]

    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 32, 32)
    assert "boxes" in target


@pytest.mark.unit
def test_dataset_subsampling() -> None:
    """from_arrays with max_samples limits dataset size."""
    images = np.random.randint(0, 255, (20, 32, 32, 3), dtype=np.uint8)
    annotations: list[dict[str, npt.NDArray[Any]]] = [
        {
            "boxes": np.array([[5, 5, 25, 25]], dtype=np.float32),
            "labels": np.array([1], dtype=np.int64),
        }
        for _ in range(20)
    ]

    ds = DetectionDataset.from_arrays(images, annotations, max_samples=5)
    assert len(ds) == 5


@pytest.mark.unit
def test_dataset_grayscale() -> None:
    """DetectionDataset converts grayscale (H, W) images to 3-channel RGB."""
    images = np.random.randint(0, 255, (3, 32, 32), dtype=np.uint8)
    annotations: list[dict[str, npt.NDArray[Any]]] = [
        {
            "boxes": np.array([[5, 5, 25, 25]], dtype=np.float32),
            "labels": np.array([1], dtype=np.int64),
        }
        for _ in range(3)
    ]

    ds = DetectionDataset(images, annotations)
    img, _ = ds[0]
    assert img.shape[0] == 3  # grayscale converted to RGB for pretrained backbone


@pytest.mark.unit
def test_dataset_with_dataloader(synthetic_det_data: SyntheticDetectionData) -> None:
    """DetectionDataset works with DataLoader + detection collate."""
    from torch.utils.data import DataLoader

    ds = DetectionDataset.from_npz(
        synthetic_det_data.image_path,
        synthetic_det_data.annotation_path,
        split="train",
    )

    loader = DataLoader(ds, batch_size=4, collate_fn=detection_collate_fn)
    batch = next(iter(loader))

    images, targets = batch
    assert isinstance(images, list)
    assert len(images) == 4
    assert isinstance(targets, list)
    assert "boxes" in targets[0]


@pytest.mark.unit
def test_dataset_from_npz_missing_image_raises(tmp_path: Any) -> None:
    """from_npz raises on missing image file."""
    from pathlib import Path

    with pytest.raises(Exception, match="not found"):
        DetectionDataset.from_npz(
            Path(tmp_path / "nonexistent.npz"),
            Path(tmp_path / "ann.npz"),
        )


@pytest.mark.unit
def test_dataset_from_npz_missing_annotation_raises(
    synthetic_det_data: SyntheticDetectionData, tmp_path: Any
) -> None:
    """from_npz raises on missing annotation file."""
    from pathlib import Path

    with pytest.raises(Exception, match="not found"):
        DetectionDataset.from_npz(
            synthetic_det_data.image_path,
            Path(tmp_path / "nonexistent_ann.npz"),
        )
