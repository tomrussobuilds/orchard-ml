"""
Pytest test suite for CIFAR-10/100 dataset converter.

Tests download-and-convert pipeline, stratified splitting,
and NPZ output format without performing real network calls.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from orchard.core.metadata import DatasetMetadata
from orchard.data_handler.fetchers.cifar_converter import (
    _create_stratified_split,
    _download_and_convert,
    ensure_cifar_npz,
)

# (cifar_cls, train_ds, test_ds) bundle returned by the mock factory.
MockBundle = tuple[Callable[..., MagicMock], MagicMock, MagicMock]


# FIXTURES
@pytest.fixture
def cifar10_metadata(tmp_path: Path) -> SimpleNamespace:
    """Minimal CIFAR-10 metadata stub."""
    return SimpleNamespace(
        name="cifar10",
        display_name="CIFAR-10",
        md5_checksum="",
        url="torchvision",
        path=tmp_path / "cifar10_32.npz",
        native_resolution=32,
    )


@pytest.fixture
def cifar100_metadata(tmp_path: Path) -> SimpleNamespace:
    """Minimal CIFAR-100 metadata stub."""
    return SimpleNamespace(
        name="cifar100",
        display_name="CIFAR-100",
        md5_checksum="",
        url="torchvision",
        path=tmp_path / "cifar100_32.npz",
        native_resolution=32,
    )


@pytest.fixture
def mock_cifar_cls() -> Callable[..., MockBundle]:
    """Creates a mock torchvision CIFAR class with fake data."""

    def _make_mock(num_classes: int = 10, train_size: int = 100, test_size: int = 20) -> MockBundle:
        train_ds = MagicMock()
        train_ds.data = np.random.randint(0, 255, (train_size, 32, 32, 3), dtype=np.uint8)
        train_ds.targets = list(np.random.randint(0, num_classes, train_size))

        test_ds = MagicMock()
        test_ds.data = np.random.randint(0, 255, (test_size, 32, 32, 3), dtype=np.uint8)
        test_ds.targets = list(np.random.randint(0, num_classes, test_size))

        def cifar_cls(root: Any, train: Any, download: Any) -> MagicMock:
            return train_ds if train else test_ds

        return cifar_cls, train_ds, test_ds

    return _make_mock


# TEST: _create_stratified_split
@pytest.mark.unit
class TestCreateStratifiedSplit:
    """Tests for _create_stratified_split."""

    def test_split_preserves_total_count(self) -> None:
        """Total samples after split should equal input count."""
        n = 1000
        images = np.random.rand(n, 32, 32, 3).astype(np.uint8)
        labels = np.random.randint(0, 10, n)

        train_imgs, train_labels, val_imgs, val_labels = _create_stratified_split(
            images, labels, val_ratio=0.15
        )

        assert len(train_imgs) + len(val_imgs) == n
        assert len(train_labels) + len(val_labels) == n

    def test_split_ratio_approximate(self) -> None:
        """Validation split should be approximately 15% of total."""
        n = 1000
        images = np.random.rand(n, 32, 32, 3).astype(np.uint8)
        labels = np.random.randint(0, 10, n)

        _, _, val_imgs, _ = _create_stratified_split(images, labels, val_ratio=0.15)

        actual_ratio = len(val_imgs) / n
        assert 0.12 < actual_ratio < 0.18

    def test_split_is_stratified(self) -> None:
        """Each class should be represented in both train and val sets."""
        n = 500
        images = np.random.rand(n, 32, 32, 3).astype(np.uint8)
        labels = np.repeat(np.arange(10), 50)

        _, train_labels, _, val_labels = _create_stratified_split(images, labels, val_ratio=0.2)

        train_classes = set(np.unique(train_labels))
        val_classes = set(np.unique(val_labels))

        assert train_classes == set(range(10))
        assert val_classes == set(range(10))

    def test_split_deterministic(self) -> None:
        """Same seed should produce identical splits."""
        n = 200
        images = np.random.rand(n, 32, 32, 3).astype(np.uint8)
        labels = np.random.randint(0, 5, n)

        result1 = _create_stratified_split(images, labels, seed=42)
        result2 = _create_stratified_split(images, labels, seed=42)

        np.testing.assert_array_equal(result1[0], result2[0])
        np.testing.assert_array_equal(result1[1], result2[1])

    def test_split_different_seeds_differ(self) -> None:
        """Different seeds should produce different label orderings."""
        n = 200
        images = np.random.randint(0, 255, (n, 32, 32, 3), dtype=np.uint8)
        labels = np.random.randint(0, 5, n)

        result1 = _create_stratified_split(images, labels, seed=42)
        result2 = _create_stratified_split(images, labels, seed=99)

        assert not np.array_equal(result1[1], result2[1])


# TEST: _download_and_convert
@pytest.mark.unit
class TestDownloadAndConvert:
    """Tests for _download_and_convert."""

    def test_creates_npz_with_correct_keys(
        self, cifar10_metadata: SimpleNamespace, mock_cifar_cls: Callable[..., MockBundle]
    ) -> None:
        """Converted NPZ should have all 6 standard keys."""
        cifar_cls, _, _ = mock_cifar_cls(num_classes=10, train_size=100, test_size=20)

        result = _download_and_convert(
            cast(DatasetMetadata, cifar10_metadata), cast(type, cifar_cls)
        )

        assert result.exists()
        with np.load(result) as data:
            assert set(data.keys()) == {
                "train_images",
                "train_labels",
                "val_images",
                "val_labels",
                "test_images",
                "test_labels",
            }

    def test_labels_shape_is_n_by_1(self, cifar10_metadata: Any, mock_cifar_cls: MagicMock) -> None:
        """Labels should be reshaped to (N, 1) for NPZ compatibility."""
        cifar_cls, _, _ = mock_cifar_cls(num_classes=10, train_size=100, test_size=20)

        result = _download_and_convert(cifar10_metadata, cifar_cls)

        with np.load(result) as data:
            assert data["train_labels"].ndim == 2
            assert data["train_labels"].shape[1] == 1
            assert data["val_labels"].ndim == 2
            assert data["val_labels"].shape[1] == 1
            assert data["test_labels"].ndim == 2
            assert data["test_labels"].shape[1] == 1

    def test_labels_dtype_is_int64(self, cifar10_metadata: Any, mock_cifar_cls: MagicMock) -> None:
        """Labels should be int64."""
        cifar_cls, _, _ = mock_cifar_cls(num_classes=10, train_size=100, test_size=20)

        result = _download_and_convert(cifar10_metadata, cifar_cls)

        with np.load(result) as data:
            assert data["train_labels"].dtype == np.int64
            assert data["val_labels"].dtype == np.int64
            assert data["test_labels"].dtype == np.int64

    def test_image_shape_is_32x32x3(self, cifar10_metadata: Any, mock_cifar_cls: MagicMock) -> None:
        """Images should be 32x32x3 HWC format."""
        cifar_cls, _, _ = mock_cifar_cls(num_classes=10, train_size=100, test_size=20)

        result = _download_and_convert(cifar10_metadata, cifar_cls)

        with np.load(result) as data:
            assert data["train_images"].shape[1:] == (32, 32, 3)
            assert data["test_images"].shape[1:] == (32, 32, 3)

    def test_train_val_split_sizes(self, cifar10_metadata: Any, mock_cifar_cls: MagicMock) -> None:
        """Train + val should equal original train size."""
        train_size = 100
        cifar_cls, _, _ = mock_cifar_cls(num_classes=10, train_size=train_size, test_size=20)

        result = _download_and_convert(cifar10_metadata, cifar_cls)

        with np.load(result) as data:
            total = len(data["train_images"]) + len(data["val_images"])
            assert total == train_size

    def test_test_set_preserved(self, cifar10_metadata: Any, mock_cifar_cls: MagicMock) -> None:
        """Test set should be passed through unchanged."""
        test_size = 20
        cifar_cls, _, _ = mock_cifar_cls(num_classes=10, train_size=100, test_size=test_size)

        result = _download_and_convert(cifar10_metadata, cifar_cls)

        with np.load(result) as data:
            assert len(data["test_images"]) == test_size
            assert len(data["test_labels"]) == test_size


# TEST: ensure_cifar_npz
@pytest.mark.unit
class TestEnsureCifarNpz:
    """Tests for ensure_cifar_npz."""

    def test_skips_download_when_npz_exists(self, cifar10_metadata: Any) -> None:
        """Existing NPZ should be returned without downloading."""
        cifar10_metadata.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cifar10_metadata.path,
            train_images=np.zeros((5, 32, 32, 3)),
            train_labels=np.zeros((5, 1)),
            val_images=np.zeros((2, 32, 32, 3)),
            val_labels=np.zeros((2, 1)),
            test_images=np.zeros((3, 32, 32, 3)),
            test_labels=np.zeros((3, 1)),
        )

        result = ensure_cifar_npz(cifar10_metadata)

        assert result == cifar10_metadata.path

    def test_cifar10_routes_to_cifar10_class(
        self, cifar10_metadata: SimpleNamespace, mock_cifar_cls: Callable[..., MockBundle]
    ) -> None:
        """CIFAR-10 metadata should route to torchvision.CIFAR10."""
        cifar_cls, _, _ = mock_cifar_cls(num_classes=10)

        with (
            patch("torchvision.datasets.CIFAR10", cifar_cls),
            patch("torchvision.datasets.CIFAR100", MagicMock()),
        ):
            result = ensure_cifar_npz(cast(DatasetMetadata, cifar10_metadata))

        assert result.exists()

    def test_cifar100_routes_to_cifar100_class(
        self, cifar100_metadata: SimpleNamespace, mock_cifar_cls: Callable[..., MockBundle]
    ) -> None:
        """CIFAR-100 metadata should route to torchvision.CIFAR100."""
        cifar_cls, _, _ = mock_cifar_cls(num_classes=100, train_size=200, test_size=40)

        with (
            patch("torchvision.datasets.CIFAR100", cifar_cls),
            patch("torchvision.datasets.CIFAR10", MagicMock()),
        ):
            result = ensure_cifar_npz(cast(DatasetMetadata, cifar100_metadata))

        assert result.exists()


# ─── MUTATION-KILLING TESTS ───


@pytest.mark.unit
class TestCreateStratifiedSplitMutations:
    """Kill surviving mutants in _create_stratified_split."""

    def test_default_val_ratio_is_015(self) -> None:
        """Default val_ratio=0.15 should produce ~15% validation split."""
        n = 1000
        images = np.random.randint(0, 255, (n, 32, 32, 3), dtype=np.uint8)
        labels = np.repeat(np.arange(10), 100)

        # Call WITHOUT specifying val_ratio — uses default 0.15
        _, _, val_imgs, _ = _create_stratified_split(images, labels)

        actual_ratio = len(val_imgs) / n
        # 0.15 default → ~15%. If mutated to 1.15, val would be ~100%.
        assert 0.12 < actual_ratio < 0.20

    def test_default_seed_is_42(self) -> None:
        """Default seed=42 should produce same result as explicit seed=42."""
        n = 200
        images = np.random.randint(0, 255, (n, 32, 32, 3), dtype=np.uint8)
        labels = np.random.randint(0, 5, n)

        result_default = _create_stratified_split(images, labels, val_ratio=0.15)
        result_explicit = _create_stratified_split(images, labels, val_ratio=0.15, seed=42)

        np.testing.assert_array_equal(result_default[0], result_explicit[0])

    def test_max_1_ensures_at_least_one_val_per_class(self) -> None:
        """max(1, ...) ensures each class has at least 1 validation sample."""
        # 5 classes with only 3 samples each → val_ratio=0.15 would give 0
        # but max(1, ...) forces at least 1
        n = 15
        images = np.random.randint(0, 255, (n, 32, 32, 3), dtype=np.uint8)
        labels = np.repeat(np.arange(5), 3)

        _, _, val_imgs, val_labels = _create_stratified_split(images, labels, val_ratio=0.15)

        # Each class should have exactly 1 val sample (max(1, int(3*0.15))=max(1,0)=1)
        for label in range(5):
            count = np.sum(val_labels == label)
            assert count >= 1, f"Class {label} has {count} val samples, expected >= 1"


@pytest.mark.unit
class TestDownloadAndConvertMutations:
    """Kill surviving mutants in _download_and_convert."""

    def test_cifar_cls_called_with_correct_args(self, cifar10_metadata: Any) -> None:
        """cifar_cls should be called with root=str(download_dir), train=T/F, download=True."""
        call_log = []

        def tracking_cls(root: Any, train: Any, download: Any) -> MagicMock:
            call_log.append({"root": root, "train": train, "download": download})
            ds = MagicMock()
            ds.data = np.random.randint(0, 255, (50, 32, 32, 3), dtype=np.uint8)
            ds.targets = list(np.random.randint(0, 10, 50))
            return ds

        _download_and_convert(cifar10_metadata, cast(type, tracking_cls))

        assert len(call_log) == 2

        # Train call
        assert call_log[0]["train"] is True
        assert call_log[0]["download"] is True
        assert isinstance(call_log[0]["root"], str)
        assert call_log[0]["root"] != "None"

        # Test call
        assert call_log[1]["train"] is False
        assert call_log[1]["download"] is True
        assert isinstance(call_log[1]["root"], str)
        assert call_log[1]["root"] == call_log[0]["root"]

    def test_download_dir_derived_from_metadata(self, cifar10_metadata: Any) -> None:
        """Download directory should be parent/.{name}_raw."""
        call_log = []

        def tracking_cls(root: Any, train: Any, download: Any) -> MagicMock:
            call_log.append(root)
            ds = MagicMock()
            ds.data = np.random.randint(0, 255, (50, 32, 32, 3), dtype=np.uint8)
            ds.targets = list(np.random.randint(0, 10, 50))
            return ds

        _download_and_convert(cifar10_metadata, cast(type, tracking_cls))

        expected_dir = str(cifar10_metadata.path.parent / ".cifar10_raw")
        assert call_log[0] == expected_dir

    def test_labels_reshaped_to_n_by_1_not_n_by_other(
        self, cifar10_metadata: SimpleNamespace, mock_cifar_cls: Callable[..., MockBundle]
    ) -> None:
        """Labels must be reshaped to (-1, 1), not (-2, 1) or other."""
        cifar_cls, _, _ = mock_cifar_cls(num_classes=10, train_size=100, test_size=20)

        result = _download_and_convert(
            cast(DatasetMetadata, cifar10_metadata), cast(type, cifar_cls)
        )

        with np.load(result) as data:
            for key in ["train_labels", "val_labels", "test_labels"]:
                assert data[key].shape[1] == 1
                assert data[key].shape[0] > 0
                # Verify values are correct (not garbage from wrong reshape)
                assert np.all(data[key] >= 0)
                assert np.all(data[key] < 10)

    def test_parent_dir_created_with_parents(
        self, tmp_path: Path, mock_cifar_cls: Callable[..., MockBundle]
    ) -> None:
        """mkdir should use parents=True for nested directories."""
        deep_path = tmp_path / "a" / "b" / "c" / "cifar10.npz"
        deep_metadata = SimpleNamespace(
            name="cifar10",
            display_name="CIFAR-10",
            md5_checksum="",
            url="torchvision",
            path=deep_path,
            native_resolution=32,
        )
        cifar_cls, _, _ = mock_cifar_cls(num_classes=10, train_size=50, test_size=10)

        result = _download_and_convert(cast(DatasetMetadata, deep_metadata), cast(type, cifar_cls))

        assert result.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
