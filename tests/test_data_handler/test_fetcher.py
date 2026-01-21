"""
Pytest test suite for MedMNIST dataset fetching and loading.

Covers download logic, retry behavior, NPZ validation,
and metadata extraction without performing real network calls.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
from types import SimpleNamespace

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
import numpy as np
import pytest
import requests

# =========================================================================== #
#                                Module Under Test                            #
# =========================================================================== #
from orchard.data_handler.fetcher import (
    _is_valid_npz,
    ensure_dataset_npz,
    load_medmnist,
    load_medmnist_health_check,
)

# =========================================================================== #
#                                   FIXTURES                                  #
# =========================================================================== #


@pytest.fixture
def metadata(tmp_path):
    """Minimal DatasetMetadata stub."""
    return SimpleNamespace(
        name="test_medmnist",
        url="https://example.com/fake.npz",
        md5_checksum="correct_md5",
        path=tmp_path / "dataset.npz",
    )


@pytest.fixture
def valid_npz_file(tmp_path):
    path = tmp_path / "valid.npz"
    np.savez(
        path,
        train_images=np.zeros((10, 28, 28, 3), dtype=np.uint8),
        train_labels=np.array([0, 1] * 5),
        val_images=np.zeros((4, 28, 28, 3), dtype=np.uint8),
        val_labels=np.array([0, 1, 0, 1]),
        test_images=np.zeros((4, 28, 28, 3), dtype=np.uint8),
        test_labels=np.array([0, 1, 0, 1]),
    )
    return path


@pytest.fixture
def monkeypatch_md5(monkeypatch):
    """Monkeypatch md5_checksum to be deterministic."""

    def fake_md5(path):
        return "correct_md5"

    monkeypatch.setattr(
        "orchard.data_handler.fetcher.md5_checksum",
        fake_md5,
    )


# =========================================================================== #
#                          TEST: _is_valid_npz                                 #
# =========================================================================== #


def test_is_valid_npz_true(valid_npz_file, monkeypatch_md5):
    """Valid NPZ with matching MD5 should return True."""
    assert _is_valid_npz(valid_npz_file, "correct_md5") is True


def test_is_valid_npz_false_on_missing_file(tmp_path):
    """Missing file should be invalid."""
    assert _is_valid_npz(tmp_path / "missing.npz", "x") is False


def test_is_valid_npz_false_on_bad_header(tmp_path, monkeypatch_md5):
    """Non-ZIP files should be rejected."""
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"NOT_A_ZIP")

    assert _is_valid_npz(bad, "correct_md5") is False


# =========================================================================== #
#                     TEST: ensure_dataset_npz                                 #
# =========================================================================== #


def test_ensure_dataset_npz_uses_existing_valid_file(metadata, valid_npz_file, monkeypatch_md5):
    """Existing valid dataset should not trigger download."""
    metadata.path.write_bytes(valid_npz_file.read_bytes())

    path = ensure_dataset_npz(metadata)

    assert path == metadata.path
    assert path.exists()


def test_ensure_dataset_npz_downloads_when_missing(
    metadata, tmp_path, monkeypatch_md5, monkeypatch
):
    """Missing dataset triggers download and validation."""

    def fake_stream_download(url, tmp_path):
        real_npz = tmp_path.with_suffix(".npz")
        np.savez(
            real_npz,
            train_images=np.zeros((5, 28, 28, 3)),
            train_labels=np.zeros(5),
            val_images=np.zeros((5, 28, 28, 3)),
            val_labels=np.zeros(5),
            test_images=np.zeros((5, 28, 28, 3)),
            test_labels=np.zeros(5),
        )
        tmp_path.write_bytes(real_npz.read_bytes())

    monkeypatch.setattr(
        "orchard.data_handler.fetcher._stream_download",
        fake_stream_download,
    )

    path = ensure_dataset_npz(metadata, retries=1)

    assert path.exists()
    assert path.suffix == ".npz"


def test_ensure_dataset_npz_retries_and_fails(metadata, monkeypatch):
    """Repeated failures should raise RuntimeError."""

    def always_fail(*args, **kwargs):
        raise requests.ConnectionError("network down")

    monkeypatch.setattr(
        "orchard.data_handler.fetcher._stream_download",
        always_fail,
    )

    monkeypatch.setattr(
        "orchard.data_handler.fetcher.time.sleep",
        lambda _: None,
    )

    with pytest.raises(RuntimeError):
        ensure_dataset_npz(metadata, retries=2, delay=0.01)


# =========================================================================== #
#                         TEST: load_medmnist                                  #
# =========================================================================== #


def test_load_medmnist_rgb(metadata, valid_npz_file, monkeypatch_md5, monkeypatch):
    """RGB dataset metadata should be inferred correctly."""

    metadata.path.write_bytes(valid_npz_file.read_bytes())

    monkeypatch.setattr(
        "orchard.data_handler.fetcher.ensure_dataset_npz",
        lambda _: metadata.path,
    )

    data = load_medmnist(metadata)

    assert data.name == metadata.name
    assert data.is_rgb is True
    assert data.num_classes == 2
    assert data.path == metadata.path


def test_load_medmnist_health_check_grayscale(metadata, tmp_path, monkeypatch_md5, monkeypatch):
    """Health check should work on grayscale datasets."""
    path = tmp_path / "gray.npz"

    np.savez(
        path,
        train_images=np.zeros((50, 28, 28), dtype=np.uint8),
        train_labels=np.arange(50),
        val_images=np.zeros((10, 28, 28), dtype=np.uint8),
        val_labels=np.arange(10),
        test_images=np.zeros((10, 28, 28), dtype=np.uint8),
        test_labels=np.arange(10),
    )

    metadata.path = path

    monkeypatch.setattr(
        "orchard.data_handler.fetcher.ensure_dataset_npz",
        lambda _: path,
    )

    data = load_medmnist_health_check(metadata, chunk_size=10)

    assert data.is_rgb is False
    assert data.num_classes == 10
