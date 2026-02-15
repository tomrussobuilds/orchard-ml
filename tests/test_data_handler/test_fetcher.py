"""
Pytest test suite for MedMNIST dataset fetching and loading.

Covers download logic, retry behavior, NPZ validation,
and metadata extraction without performing real network calls.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import requests

from orchard.data_handler.fetcher import (
    ensure_dataset_npz,
    load_dataset,
    load_dataset_health_check,
)
from orchard.data_handler.fetchers.medmnist_fetcher import (
    _is_valid_npz,
    _stream_download,
)


# FIXTURES
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
        "orchard.data_handler.fetchers.medmnist_fetcher.md5_checksum",
        fake_md5,
    )


# TEST: _is_valid_npz
@pytest.mark.unit
def test_is_valid_npz_true(valid_npz_file, monkeypatch_md5):
    """Valid NPZ with matching MD5 should return True."""
    assert _is_valid_npz(valid_npz_file, "correct_md5") is True


@pytest.mark.unit
def test_is_valid_npz_false_on_missing_file(tmp_path):
    """Missing file should be invalid."""
    assert _is_valid_npz(tmp_path / "missing.npz", "x") is False


@pytest.mark.unit
def test_is_valid_npz_false_on_bad_header(tmp_path, monkeypatch_md5):
    """Non-ZIP files should be rejected."""
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"NOT_A_ZIP")

    assert _is_valid_npz(bad, "correct_md5") is False


@pytest.mark.unit
def test_is_valid_npz_false_on_md5_mismatch(valid_npz_file, monkeypatch):
    """Valid NPZ file but wrong MD5 should return False."""

    def wrong_md5(path):
        return "wrong_md5"

    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher.md5_checksum",
        wrong_md5,
    )

    assert _is_valid_npz(valid_npz_file, "correct_md5") is False


@pytest.mark.unit
def test_is_valid_npz_ioerror(tmp_path, monkeypatch):
    """IOError during file reading should return False."""
    path = tmp_path / "test.npz"
    path.write_bytes(b"PK_some_content")

    def raise_ioerror(*args, **kwargs):
        raise IOError("Cannot read file")

    monkeypatch.setattr("builtins.open", raise_ioerror)

    assert _is_valid_npz(path, "any_md5") is False


# TEST: ensure_dataset_npz
@pytest.mark.unit
def test_ensure_dataset_npz_uses_existing_valid_file(metadata, valid_npz_file, monkeypatch_md5):
    """Existing valid dataset should not trigger download."""
    metadata.path.write_bytes(valid_npz_file.read_bytes())

    path = ensure_dataset_npz(metadata)

    assert path == metadata.path
    assert path.exists()


@pytest.mark.unit
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
        "orchard.data_handler.fetchers.medmnist_fetcher._stream_download",
        fake_stream_download,
    )

    path = ensure_dataset_npz(metadata, retries=1)

    assert path.exists()
    assert path.suffix == ".npz"


@pytest.mark.unit
def test_ensure_dataset_npz_removes_corrupted_file(
    metadata, tmp_path, monkeypatch_md5, monkeypatch
):
    """Corrupted existing file should be deleted before download."""
    metadata.path.parent.mkdir(parents=True, exist_ok=True)
    metadata.path.write_bytes(b"CORRUPTED")

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
        "orchard.data_handler.fetchers.medmnist_fetcher._stream_download",
        fake_stream_download,
    )

    path = ensure_dataset_npz(metadata, retries=1)

    assert path.exists()
    assert path.suffix == ".npz"


@pytest.mark.unit
def test_ensure_dataset_npz_md5_mismatch_raises_error(metadata, monkeypatch):
    """MD5 mismatch should raise ValueError and retry."""
    call_count = {"count": 0}

    def fake_stream_download(url, tmp_path):
        call_count["count"] += 1
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

    def fake_md5(path):
        return "wrong_md5"

    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher._stream_download",
        fake_stream_download,
    )
    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher.md5_checksum",
        fake_md5,
    )
    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher.time.sleep",
        lambda _: None,
    )

    with pytest.raises(RuntimeError):
        ensure_dataset_npz(metadata, retries=2, delay=0.01)

    assert call_count["count"] == 2


@pytest.mark.unit
def test_ensure_dataset_npz_retries_and_fails(metadata, monkeypatch):
    """Repeated failures should raise RuntimeError."""

    def always_fail(*args, **kwargs):
        raise requests.ConnectionError("network down")

    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher._stream_download",
        always_fail,
    )

    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher.time.sleep",
        lambda _: None,
    )

    with pytest.raises(RuntimeError):
        ensure_dataset_npz(metadata, retries=2, delay=0.01)


@pytest.mark.unit
def test_ensure_dataset_npz_rate_limit_429(metadata, monkeypatch):
    """Rate limit (429) should trigger exponential backoff."""
    call_count = {"count": 0}
    sleep_calls = []

    def fake_stream_download(url, tmp_path):
        call_count["count"] += 1
        exc = requests.HTTPError("429 Rate Limit")
        exc.response = SimpleNamespace(status_code=429)
        raise exc

    def fake_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher._stream_download",
        fake_stream_download,
    )
    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher.time.sleep",
        fake_sleep,
    )

    with pytest.raises(RuntimeError):
        ensure_dataset_npz(metadata, retries=3, delay=1.0)

    assert call_count["count"] == 3

    assert len(sleep_calls) == 2
    assert sleep_calls[0] == pytest.approx(1.0)
    assert sleep_calls[1] == pytest.approx(4.0)


@pytest.mark.unit
def test_ensure_dataset_npz_cleans_up_tmp_on_error(metadata, tmp_path, monkeypatch):
    """Temporary file should be cleaned up on download failure."""
    tmp_file_path = metadata.path.with_suffix(".tmp")

    def fake_stream_download(url, tmp_path):
        tmp_path.write_bytes(b"temp_data")
        raise requests.ConnectionError("network error")

    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher._stream_download",
        fake_stream_download,
    )
    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher.time.sleep",
        lambda _: None,
    )

    with pytest.raises(RuntimeError):
        ensure_dataset_npz(metadata, retries=1, delay=0.01)

    assert not tmp_file_path.exists()


@pytest.mark.unit
def test_ensure_dataset_npz_error_without_response_attribute(metadata, monkeypatch):
    """Error without response attribute should use normal delay."""
    sleep_calls = []

    def fake_stream_download(url, tmp_path):
        raise ValueError("Some error without response")

    def fake_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher._stream_download",
        fake_stream_download,
    )
    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher.time.sleep",
        fake_sleep,
    )

    with pytest.raises(RuntimeError):
        ensure_dataset_npz(metadata, retries=2, delay=2.0)

    assert sleep_calls[0] == pytest.approx(2.0)


# TEST: _stream_download
@pytest.mark.unit
def test_stream_download_success(tmp_path, monkeypatch):
    """Successful download should write content to file."""
    output_path = tmp_path / "output.npz"
    test_content = b"test_npz_content"

    class FakeResponse:
        def __init__(self):
            self.headers = {"Content-Type": "application/octet-stream"}

        def raise_for_status(self):
            """No-op: simulates a successful HTTP response (no error to raise)."""

        def iter_content(self, chunk_size):
            yield test_content

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_get(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr("requests.get", fake_get)

    _stream_download("https://example.com/file.npz", output_path)

    assert output_path.exists()
    assert output_path.read_bytes() == test_content


@pytest.mark.unit
def test_stream_download_html_content_raises_error(tmp_path, monkeypatch):
    """HTML content instead of NPZ should raise ValueError."""
    output_path = tmp_path / "output.npz"

    class FakeResponse:
        def __init__(self):
            self.headers = {"Content-Type": "text/html"}

        def raise_for_status(self):
            """No-op: simulates a successful HTTP response (no error to raise)."""

        def iter_content(self, chunk_size):
            yield b"<html>Not Found</html>"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_get(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr("requests.get", fake_get)

    with pytest.raises(ValueError, match="HTML page"):
        _stream_download("https://example.com/file.npz", output_path)


@pytest.mark.unit
def test_stream_download_skips_empty_chunks(tmp_path, monkeypatch):
    """Empty chunks should be skipped during download."""
    output_path = tmp_path / "output.npz"

    class FakeResponse:
        def __init__(self):
            self.headers = {"Content-Type": "application/octet-stream"}

        def raise_for_status(self):
            """No-op: simulates a successful HTTP response (no error to raise)."""

        def iter_content(self, chunk_size):
            yield b"first"
            yield b""
            yield None
            yield b"second"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_get(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr("requests.get", fake_get)

    _stream_download("https://example.com/file.npz", output_path)

    assert output_path.exists()
    assert output_path.read_bytes() == b"firstsecond"


@pytest.mark.unit
def test_stream_download_http_error(tmp_path, monkeypatch):
    """HTTP errors should be raised."""
    output_path = tmp_path / "output.npz"

    class FakeResponse:
        def raise_for_status(self):
            raise requests.HTTPError("404 Not Found")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_get(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr("requests.get", fake_get)

    with pytest.raises(requests.HTTPError):
        _stream_download("https://example.com/file.npz", output_path)


# TEST: load_dataset
@pytest.mark.unit
def test_load_medmnist_rgb(metadata, valid_npz_file, monkeypatch_md5, monkeypatch):
    """RGB dataset metadata should be inferred correctly."""

    metadata.path.write_bytes(valid_npz_file.read_bytes())

    monkeypatch.setattr(
        "orchard.data_handler.fetcher.ensure_dataset_npz",
        lambda _: metadata.path,
    )

    data = load_dataset(metadata)

    assert data.name == metadata.name
    assert data.is_rgb is True
    assert data.num_classes == 2
    assert data.path == metadata.path


@pytest.mark.unit
def test_load_medmnist_grayscale(metadata, tmp_path, monkeypatch_md5, monkeypatch):
    """Grayscale dataset should have is_rgb=False."""
    path = tmp_path / "gray.npz"

    np.savez(
        path,
        train_images=np.zeros((50, 28, 28), dtype=np.uint8),
        train_labels=np.array([0, 1, 2] * 16 + [0, 1]),
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

    data = load_dataset(metadata)

    assert data.is_rgb is False
    assert data.num_classes == 3


@pytest.mark.unit
def test_load_dataset_health_check_rgb(metadata, valid_npz_file, monkeypatch_md5, monkeypatch):
    """Health check should work on RGB datasets."""
    metadata.path.write_bytes(valid_npz_file.read_bytes())

    monkeypatch.setattr(
        "orchard.data_handler.fetcher.ensure_dataset_npz",
        lambda _: metadata.path,
    )

    data = load_dataset_health_check(metadata, chunk_size=5)

    assert data.is_rgb is True
    assert data.num_classes == 2
    assert data.path == metadata.path


@pytest.mark.unit
def test_load_dataset_health_check_grayscale(metadata, tmp_path, monkeypatch_md5, monkeypatch):
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

    data = load_dataset_health_check(metadata, chunk_size=10)

    assert data.is_rgb is False
    assert data.num_classes == 10


@pytest.mark.unit
def test_load_dataset_health_check_small_chunk(metadata, tmp_path, monkeypatch_md5, monkeypatch):
    """Health check with small chunk should count classes correctly."""
    path = tmp_path / "test.npz"

    labels = np.array([0, 1, 0] + [2, 3, 4] * 10)

    np.savez(
        path,
        train_images=np.zeros((len(labels), 28, 28, 3), dtype=np.uint8),
        train_labels=labels,
        val_images=np.zeros((10, 28, 28, 3), dtype=np.uint8),
        val_labels=np.arange(10),
        test_images=np.zeros((10, 28, 28, 3), dtype=np.uint8),
        test_labels=np.arange(10),
    )

    metadata.path = path

    monkeypatch.setattr(
        "orchard.data_handler.fetcher.ensure_dataset_npz",
        lambda _: path,
    )

    data = load_dataset_health_check(metadata, chunk_size=3)

    assert data.num_classes == 2


@pytest.mark.unit
def test_ensure_dataset_npz_galaxy10_converter_path(tmp_path, monkeypatch):
    """Galaxy10 dataset should trigger converter import and return path."""
    target_npz = tmp_path / "galaxy10.npz"

    dummy_data = {
        "train_images": np.zeros((5, 10, 10, 3), dtype=np.uint8),
        "train_labels": np.zeros((5, 1), dtype=np.int64),
        "val_images": np.zeros((2, 10, 10, 3), dtype=np.uint8),
        "val_labels": np.zeros((2, 1), dtype=np.int64),
        "test_images": np.zeros((3, 10, 10, 3), dtype=np.uint8),
        "test_labels": np.zeros((3, 1), dtype=np.int64),
    }
    np.savez_compressed(target_npz, **dummy_data)

    galaxy10_metadata = SimpleNamespace(
        name="galaxy10",
        url="https://example.com/galaxy10.h5",
        md5_checksum="test_md5",
        path=target_npz,
        native_resolution=224,
    )

    def mock_ensure_galaxy10_npz(metadata):
        return metadata.path

    monkeypatch.setattr(
        "orchard.data_handler.fetchers.ensure_galaxy10_npz",
        mock_ensure_galaxy10_npz,
    )

    result = ensure_dataset_npz(galaxy10_metadata)

    assert result == target_npz
