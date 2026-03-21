"""
Test Suite for PennFudan Pedestrian Detection Fetcher.

Tests mask-to-bbox conversion, box rescaling, ZIP parsing,
split creation, and NPZ saving — all without network access.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from orchard.data_handler.fetchers.pennfudan_fetcher import (
    _download_zip,
    _mask_to_boxes,
    _parse_pennfudan_zip,
    _rescale_boxes,
    _save_detection_npz,
    _split_indices,
    ensure_pennfudan_npz,
)

# ── _download_zip ────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_download_zip_success() -> None:
    """Successful download returns a ZipFile."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("test.txt", "hello")
    zip_bytes = buf.getvalue()

    mock_resp = MagicMock()
    mock_resp.content = zip_bytes
    mock_resp.raise_for_status = MagicMock()

    with patch(
        "orchard.data_handler.fetchers.pennfudan_fetcher.requests.get", return_value=mock_resp
    ):
        result = _download_zip("https://example.com/test.zip", retries=1)

    assert isinstance(result, zipfile.ZipFile)
    assert "test.txt" in result.namelist()


@pytest.mark.unit
def test_download_zip_retries_on_failure() -> None:
    """Download retries and raises after all attempts fail."""
    from orchard.exceptions import OrchardDatasetError

    with patch(
        "orchard.data_handler.fetchers.pennfudan_fetcher.requests.get",
        side_effect=OSError("connection failed"),
    ):
        with pytest.raises(OrchardDatasetError, match="Failed to download PennFudan"):
            _download_zip("https://example.com/test.zip", retries=2)


@pytest.mark.unit
def test_download_zip_partial_failure_recovers() -> None:
    """First attempt fails, second succeeds — download recovers."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("test.txt", "hello")
    zip_bytes = buf.getvalue()

    mock_success = MagicMock()
    mock_success.content = zip_bytes
    mock_success.raise_for_status = MagicMock()

    with patch(
        "orchard.data_handler.fetchers.pennfudan_fetcher.requests.get",
        side_effect=[OSError("fail"), mock_success],
    ) as mock_get:
        result = _download_zip("https://example.com/test.zip", retries=2)

    assert isinstance(result, zipfile.ZipFile)
    assert mock_get.call_count == 2


@pytest.mark.unit
def test_download_zip_single_retry_single_call() -> None:
    """With retries=1, exactly one request.get call is made on success."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("t.txt", "ok")

    mock_resp = MagicMock()
    mock_resp.content = buf.getvalue()
    mock_resp.raise_for_status = MagicMock()

    with patch(
        "orchard.data_handler.fetchers.pennfudan_fetcher.requests.get",
        return_value=mock_resp,
    ) as mock_get:
        _download_zip("https://example.com/t.zip", retries=1)

    assert mock_get.call_count == 1


@pytest.mark.unit
def test_download_zip_exhausts_all_retries() -> None:
    """All retries exhausted — exactly retries calls are made."""
    from orchard.exceptions import OrchardDatasetError

    with patch(
        "orchard.data_handler.fetchers.pennfudan_fetcher.requests.get",
        side_effect=OSError("fail"),
    ) as mock_get:
        with pytest.raises(OrchardDatasetError):
            _download_zip("https://example.com/x.zip", retries=3)

    assert mock_get.call_count == 3


@pytest.mark.unit
def test_download_zip_passes_url_and_timeout() -> None:
    """requests.get receives the correct URL and timeout."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("t.txt", "ok")

    mock_resp = MagicMock()
    mock_resp.content = buf.getvalue()
    mock_resp.raise_for_status = MagicMock()

    with patch(
        "orchard.data_handler.fetchers.pennfudan_fetcher.requests.get",
        return_value=mock_resp,
    ) as mock_get:
        _download_zip("https://example.com/data.zip", retries=1, timeout=30)

    mock_get.assert_called_once_with("https://example.com/data.zip", timeout=30)


# ── _mask_to_boxes ───────────────────────────────────────────────────────────


@pytest.mark.unit
def test_mask_to_boxes_single_instance() -> None:
    """Single instance mask produces one bounding box."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:50, 20:60] = 1
    boxes = _mask_to_boxes(mask)
    assert boxes.shape == (1, 4)
    assert boxes[0].tolist() == [20, 10, 59, 49]


@pytest.mark.unit
def test_mask_to_boxes_multiple_instances() -> None:
    """Multiple instance IDs produce multiple boxes."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 10:30] = 1
    mask[50:80, 50:80] = 2
    boxes = _mask_to_boxes(mask)
    assert boxes.shape == (2, 4)


@pytest.mark.unit
def test_mask_to_boxes_empty_mask() -> None:
    """All-zero mask produces empty boxes array."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    boxes = _mask_to_boxes(mask)
    assert boxes.shape == (0, 4)
    assert boxes.dtype == np.float32


@pytest.mark.unit
def test_mask_to_boxes_single_pixel_instance_skipped() -> None:
    """Single-pixel instance (x_max == x_min or y_max == y_min) is filtered out."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    # Instance 1: single pixel (should be skipped)
    mask[25, 25] = 1
    # Instance 2: single row (y_max == y_min, should be skipped)
    mask[10, 5:15] = 2
    # Instance 3: valid box
    mask[30:40, 30:40] = 3
    boxes = _mask_to_boxes(mask)
    assert boxes.shape == (1, 4)  # only instance 3


@pytest.mark.unit
def test_mask_to_boxes_dtype_float32() -> None:
    """Output boxes are float32."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[5:15, 5:15] = 1
    boxes = _mask_to_boxes(mask)
    assert boxes.dtype == np.float32


@pytest.mark.unit
def test_mask_to_boxes_single_column_filtered() -> None:
    """Single-column instance (x_max == x_min) is filtered out."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[5:15, 10] = 1  # single column: x_min == x_max == 10
    mask[20:35, 20:35] = 2  # valid box
    boxes = _mask_to_boxes(mask)
    assert boxes.shape == (1, 4)  # only instance 2


# ── _rescale_boxes ───────────────────────────────────────────────────────────


@pytest.mark.unit
def test_rescale_boxes_doubles_size() -> None:
    """Boxes scale correctly when image doubles in size."""
    boxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
    scaled = _rescale_boxes(boxes, orig_w=100, orig_h=100, target_size=200)
    np.testing.assert_allclose(scaled, [[20, 40, 60, 80]])


@pytest.mark.unit
def test_rescale_boxes_halves_size() -> None:
    """Boxes scale correctly when image halves."""
    boxes = np.array([[20, 40, 60, 80]], dtype=np.float32)
    scaled = _rescale_boxes(boxes, orig_w=200, orig_h=200, target_size=100)
    np.testing.assert_allclose(scaled, [[10, 20, 30, 40]])


@pytest.mark.unit
def test_rescale_boxes_empty() -> None:
    """Empty boxes array returns unchanged."""
    boxes = np.zeros((0, 4), dtype=np.float32)
    scaled = _rescale_boxes(boxes, orig_w=100, orig_h=100, target_size=224)
    assert scaled.shape == (0, 4)


@pytest.mark.unit
def test_rescale_boxes_asymmetric() -> None:
    """Non-square original image rescales x and y independently."""
    boxes = np.array([[0, 0, 100, 200]], dtype=np.float32)
    scaled = _rescale_boxes(boxes, orig_w=200, orig_h=400, target_size=100)
    np.testing.assert_allclose(scaled, [[0, 0, 50, 50]])


# ── _split_indices ───────────────────────────────────────────────────────────


@pytest.mark.unit
def test_split_indices_covers_all() -> None:
    """Train + val + test indices cover all samples without overlap."""
    train, val, test = _split_indices(100, seed=42)
    all_idx = np.concatenate([train, val, test])
    assert len(all_idx) == 100
    assert len(np.unique(all_idx)) == 100


@pytest.mark.unit
def test_split_indices_ratios() -> None:
    """Split ratios approximate 70/15/15."""
    train, val, test = _split_indices(100, seed=42)
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15


@pytest.mark.unit
def test_split_indices_deterministic() -> None:
    """Same seed produces same splits."""
    t1, v1, te1 = _split_indices(50, seed=123)
    t2, v2, te2 = _split_indices(50, seed=123)
    np.testing.assert_array_equal(t1, t2)
    np.testing.assert_array_equal(v1, v2)
    np.testing.assert_array_equal(te1, te2)


# ── _save_detection_npz ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_save_detection_npz_creates_files(tmp_path: Path) -> None:
    """Saving produces both image and annotation NPZ files."""
    images = np.random.randint(0, 255, (10, 32, 32, 3), dtype=np.uint8)
    boxes = [np.array([[0, 0, 10, 10]], dtype=np.float32) for _ in range(10)]
    labels = [np.array([1], dtype=np.int64) for _ in range(10)]

    img_path = tmp_path / "images.npz"
    ann_path = tmp_path / "annotations.npz"

    _save_detection_npz(images, boxes, labels, img_path, ann_path)

    assert img_path.exists()
    assert ann_path.exists()


@pytest.mark.unit
def test_save_detection_npz_split_keys(tmp_path: Path) -> None:
    """NPZ files contain correct split keys."""
    images = np.random.randint(0, 255, (10, 32, 32, 3), dtype=np.uint8)
    boxes = [np.array([[0, 0, 10, 10]], dtype=np.float32) for _ in range(10)]
    labels = [np.array([1], dtype=np.int64) for _ in range(10)]

    img_path = tmp_path / "images.npz"
    ann_path = tmp_path / "annotations.npz"

    _save_detection_npz(images, boxes, labels, img_path, ann_path)

    with np.load(img_path) as data:
        assert "train_images" in data
        assert "val_images" in data
        assert "test_images" in data

    with np.load(ann_path, allow_pickle=True) as data:
        assert "train_boxes" in data
        assert "train_labels" in data
        assert "val_boxes" in data
        assert "val_labels" in data
        assert "test_boxes" in data
        assert "test_labels" in data


@pytest.mark.unit
def test_save_detection_npz_split_sizes(tmp_path: Path) -> None:
    """Total samples across splits equals input length."""
    images = np.random.randint(0, 255, (20, 32, 32, 3), dtype=np.uint8)
    boxes = [np.array([[0, 0, 10, 10]], dtype=np.float32) for _ in range(20)]
    labels = [np.array([1], dtype=np.int64) for _ in range(20)]

    img_path = tmp_path / "images.npz"
    ann_path = tmp_path / "annotations.npz"

    _save_detection_npz(images, boxes, labels, img_path, ann_path)

    with np.load(img_path) as data:
        total = len(data["train_images"]) + len(data["val_images"]) + len(data["test_images"])
        assert total == 20

    with np.load(ann_path, allow_pickle=True) as data:
        total_ann = len(data["train_boxes"]) + len(data["val_boxes"]) + len(data["test_boxes"])
        assert total_ann == 20


@pytest.mark.unit
def test_save_detection_npz_all_annotation_keys(tmp_path: Path) -> None:
    """Annotation NPZ contains all 6 split keys (boxes + labels x 3 splits)."""
    images = np.random.randint(0, 255, (10, 32, 32, 3), dtype=np.uint8)
    boxes = [np.array([[0, 0, 10, 10]], dtype=np.float32) for _ in range(10)]
    labels = [np.array([1], dtype=np.int64) for _ in range(10)]

    img_path = tmp_path / "images.npz"
    ann_path = tmp_path / "annotations.npz"

    _save_detection_npz(images, boxes, labels, img_path, ann_path)

    expected = {
        "train_boxes",
        "train_labels",
        "val_boxes",
        "val_labels",
        "test_boxes",
        "test_labels",
    }
    with np.load(ann_path, allow_pickle=True) as data:
        assert set(data.files) == expected
        for split in ["train", "val", "test"]:
            assert len(data[f"{split}_labels"]) == len(data[f"{split}_boxes"])


@pytest.mark.unit
def test_save_detection_npz_ragged_boxes(tmp_path: Path) -> None:
    """Ragged boxes (variable count per image) are saved correctly."""
    images = np.random.randint(0, 255, (5, 32, 32, 3), dtype=np.uint8)
    boxes = [
        np.array([[0, 0, 10, 10]], dtype=np.float32),
        np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=np.float32),
        np.array([[0, 0, 10, 10]], dtype=np.float32),
        np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]], dtype=np.float32),
        np.array([[0, 0, 10, 10]], dtype=np.float32),
    ]
    labels = [np.ones(len(b), dtype=np.int64) for b in boxes]

    img_path = tmp_path / "images.npz"
    ann_path = tmp_path / "annotations.npz"

    _save_detection_npz(images, boxes, labels, img_path, ann_path)

    with np.load(ann_path, allow_pickle=True) as data:
        total_boxes = sum(len(data[f"{s}_boxes"]) for s in ["train", "val", "test"])
        total_labels = sum(len(data[f"{s}_labels"]) for s in ["train", "val", "test"])
        assert total_boxes == 5
        assert total_labels == 5


# ── _parse_pennfudan_zip ─────────────────────────────────────────────────────


def _make_fake_zip() -> zipfile.ZipFile:
    """Create an in-memory ZIP mimicking PennFudan structure."""
    from PIL import Image as PILImage

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(3):
            # RGB image
            img = PILImage.fromarray(np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8))
            img_buf = io.BytesIO()
            img.save(img_buf, format="PNG")
            zf.writestr(f"PennFudanPed/PNGImages/img_{i:03d}.png", img_buf.getvalue())

            # Instance mask (2 pedestrians)
            mask = np.zeros((100, 80), dtype=np.uint8)
            mask[10:40, 10:30] = 1
            mask[50:90, 40:70] = 2
            mask_img = PILImage.fromarray(mask)
            mask_buf = io.BytesIO()
            mask_img.save(mask_buf, format="PNG")
            zf.writestr(f"PennFudanPed/PedMasks/img_{i:03d}_mask.png", mask_buf.getvalue())

    buf.seek(0)
    return zipfile.ZipFile(buf, "r")


@pytest.mark.unit
def test_parse_zip_produces_correct_shapes() -> None:
    """Parsed ZIP returns images, boxes, and labels with matching lengths."""
    zf = _make_fake_zip()
    images, boxes_list, labels_list = _parse_pennfudan_zip(zf, target_size=64)

    assert images.shape == (3, 64, 64, 3)
    assert len(boxes_list) == 3
    assert len(labels_list) == 3


@pytest.mark.unit
def test_parse_zip_boxes_are_rescaled() -> None:
    """Bounding boxes are rescaled to target resolution."""
    zf = _make_fake_zip()
    _, boxes_list, _ = _parse_pennfudan_zip(zf, target_size=64)

    for boxes in boxes_list:
        assert boxes.shape[1] == 4
        # All coordinates should be within [0, 64]
        assert np.all(boxes >= 0)
        assert np.all(boxes <= 64)


@pytest.mark.unit
def test_parse_zip_labels_are_ones() -> None:
    """All labels are 1 (person class)."""
    zf = _make_fake_zip()
    _, _, labels_list = _parse_pennfudan_zip(zf, target_size=64)

    for labels in labels_list:
        assert np.all(labels == 1)


@pytest.mark.unit
def test_parse_zip_labels_dtype_int64() -> None:
    """Labels arrays have dtype int64."""
    zf = _make_fake_zip()
    _, _, labels_list = _parse_pennfudan_zip(zf, target_size=64)

    for labels in labels_list:
        assert labels.dtype == np.int64


@pytest.mark.unit
def test_parse_zip_rgba_converted_to_rgb() -> None:
    """RGBA images are converted to RGB (3 channels) by _parse_pennfudan_zip."""
    from PIL import Image as PILImage

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(2):
            img = PILImage.fromarray(
                np.random.randint(0, 255, (100, 80, 4), dtype=np.uint8), mode="RGBA"
            )
            img_buf = io.BytesIO()
            img.save(img_buf, format="PNG")
            zf.writestr(f"PennFudanPed/PNGImages/img_{i:03d}.png", img_buf.getvalue())

            mask = np.zeros((100, 80), dtype=np.uint8)
            mask[10:40, 10:30] = 1
            mask_img = PILImage.fromarray(mask)
            mask_buf = io.BytesIO()
            mask_img.save(mask_buf, format="PNG")
            zf.writestr(f"PennFudanPed/PedMasks/img_{i:03d}_mask.png", mask_buf.getvalue())

    buf.seek(0)
    zf_in = zipfile.ZipFile(buf, "r")
    images, _, _ = _parse_pennfudan_zip(zf_in, target_size=64)
    assert images.shape[-1] == 3  # RGB, not RGBA


@pytest.mark.unit
def test_parse_zip_mismatched_counts_raises() -> None:
    """Mismatched image/mask file counts raise OrchardDatasetError."""
    from orchard.exceptions import OrchardDatasetError

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        from PIL import Image as PILImage

        img = PILImage.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
        img_buf = io.BytesIO()
        img.save(img_buf, format="PNG")
        zf.writestr("PennFudanPed/PNGImages/img_000.png", img_buf.getvalue())
        # No mask file

    buf.seek(0)
    zf = zipfile.ZipFile(buf, "r")

    with pytest.raises(OrchardDatasetError, match="mismatch"):
        _parse_pennfudan_zip(zf)


# ── ensure_pennfudan_npz ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_ensure_pennfudan_cached(tmp_path: Path) -> None:
    """Returns immediately when both NPZ files exist."""
    img_path = tmp_path / "images.npz"
    ann_path = tmp_path / "annotations.npz"
    img_path.touch()
    ann_path.touch()

    meta = MagicMock()
    meta.path = img_path
    meta.annotation_path = ann_path

    result = ensure_pennfudan_npz(meta)
    assert result == img_path


@pytest.mark.unit
def test_ensure_pennfudan_no_annotation_path_raises() -> None:
    """Raises when metadata has no annotation_path."""
    from orchard.exceptions import OrchardDatasetError

    meta = MagicMock()
    meta.annotation_path = None

    with pytest.raises(OrchardDatasetError, match="annotation_path"):
        ensure_pennfudan_npz(meta)


@pytest.mark.unit
def test_ensure_pennfudan_downloads_and_converts(tmp_path: Path) -> None:
    """Full flow: download mock ZIP, parse, save NPZ."""
    img_path = tmp_path / "images.npz"
    ann_path = tmp_path / "annotations.npz"

    meta = MagicMock()
    meta.path = img_path
    meta.annotation_path = ann_path
    meta.url = "https://example.com/PennFudanPed.zip"

    fake_zip = _make_fake_zip()
    with patch(
        "orchard.data_handler.fetchers.pennfudan_fetcher._download_zip",
        return_value=fake_zip,
    ):
        result = ensure_pennfudan_npz(meta)

    assert result == img_path
    assert img_path.exists()
    assert ann_path.exists()

    with np.load(img_path) as data:
        assert "train_images" in data

    with np.load(ann_path, allow_pickle=True) as data:
        assert "train_boxes" in data
        assert "train_labels" in data


@pytest.mark.unit
def test_ensure_partial_cache_redownloads(tmp_path: Path) -> None:
    """When only image NPZ exists (not annotation), re-download is triggered."""
    img_path = tmp_path / "images.npz"
    ann_path = tmp_path / "annotations.npz"
    img_path.touch()  # only images exist, annotation missing

    meta = MagicMock()
    meta.path = img_path
    meta.annotation_path = ann_path
    meta.url = "https://example.com/PennFudanPed.zip"

    fake_zip = _make_fake_zip()
    with patch(
        "orchard.data_handler.fetchers.pennfudan_fetcher._download_zip",
        return_value=fake_zip,
    ) as mock_dl:
        ensure_pennfudan_npz(meta)

    mock_dl.assert_called_once()


@pytest.mark.unit
def test_ensure_passes_url_to_download(tmp_path: Path) -> None:
    """ensure_pennfudan_npz passes metadata.url to _download_zip."""
    img_path = tmp_path / "images.npz"
    ann_path = tmp_path / "annotations.npz"

    meta = MagicMock()
    meta.path = img_path
    meta.annotation_path = ann_path
    meta.url = "https://example.com/PennFudanPed.zip"

    fake_zip = _make_fake_zip()
    with patch(
        "orchard.data_handler.fetchers.pennfudan_fetcher._download_zip",
        return_value=fake_zip,
    ) as mock_dl:
        ensure_pennfudan_npz(meta)

    mock_dl.assert_called_once_with("https://example.com/PennFudanPed.zip")
