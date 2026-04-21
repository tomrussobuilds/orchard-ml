"""
Pytest test suite for dataset fetching and loading.

Covers download logic, retry behavior, NPZ validation,
and metadata extraction without performing real network calls.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import pytest
import requests

from orchard.data_handler.dispatcher import (
    ensure_dataset_npz,
    load_dataset,
    load_dataset_health_check,
)
from orchard.data_handler.fetchers.medmnist_fetcher import (
    _is_valid_npz,
    _retry_delay,
    _stream_download,
    ensure_medmnist_npz,
)
from orchard.exceptions import OrchardDatasetError


# FIXTURES
@pytest.fixture
def metadata(tmp_path: Path) -> SimpleNamespace:
    """Minimal DatasetMetadata stub."""
    return SimpleNamespace(
        name="test_medmnist",
        url="https://example.com/fake.npz",
        md5_checksum="correct_md5",
        path=tmp_path / "dataset.npz",
    )


@pytest.fixture
def valid_npz_file(tmp_path: Path) -> Path:
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
def monkeypatch_md5(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch md5_checksum to be deterministic."""

    def fake_md5(path: Any) -> str:
        return "correct_md5"

    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher.md5_checksum",
        fake_md5,
    )


# TEST: _is_valid_npz
@pytest.mark.unit
def test_is_valid_npz_true(valid_npz_file: Path, monkeypatch_md5: None) -> None:
    """Valid NPZ with matching MD5 should return True."""
    assert _is_valid_npz(valid_npz_file, "correct_md5") is True


@pytest.mark.unit
def test_is_valid_npz_false_on_missing_file(tmp_path: Path) -> None:
    """Missing file should be invalid."""
    assert _is_valid_npz(tmp_path / "missing.npz", "x") is False


@pytest.mark.unit
def test_is_valid_npz_false_on_bad_header(tmp_path: Path, monkeypatch_md5: None) -> None:
    """Non-ZIP files should be rejected."""
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"NOT_A_ZIP")

    assert _is_valid_npz(bad, "correct_md5") is False


@pytest.mark.unit
def test_is_valid_npz_false_on_md5_mismatch(
    valid_npz_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Valid NPZ file but wrong MD5 should return False."""

    def wrong_md5(path: Any) -> str:
        return "wrong_md5"

    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher.md5_checksum",
        wrong_md5,
    )

    assert _is_valid_npz(valid_npz_file, "correct_md5") is False


@pytest.mark.unit
def test_is_valid_npz_ioerror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """IOError during file reading should return False."""
    path = tmp_path / "test.npz"
    path.write_bytes(b"PK_some_content")

    def raise_ioerror(*args: object, **kwargs: object) -> None:
        raise IOError("Cannot read file")

    monkeypatch.setattr("builtins.open", raise_ioerror)

    assert _is_valid_npz(path, "any_md5") is False


# TEST: ensure_dataset_npz
@pytest.mark.unit
def test_ensure_dataset_npz_uses_existing_valid_file(
    metadata: SimpleNamespace, valid_npz_file: Path, monkeypatch_md5: None
) -> None:
    """Existing valid dataset should not trigger download."""
    metadata.path.write_bytes(valid_npz_file.read_bytes())

    path = ensure_dataset_npz(metadata)  # type: ignore[arg-type]

    assert path == metadata.path
    assert path.exists()


@pytest.mark.unit
def test_ensure_dataset_npz_downloads_when_missing(
    metadata: SimpleNamespace,
    tmp_path: Path,
    monkeypatch_md5: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing dataset triggers download and validation."""

    def fake_stream_download(url: Any, tmp_path: Path) -> None:
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

    path = ensure_dataset_npz(metadata, retries=1)  # type: ignore[arg-type]

    assert path.exists()
    assert path.suffix == ".npz"


@pytest.mark.unit
def test_ensure_dataset_npz_removes_corrupted_file(
    metadata: SimpleNamespace,
    tmp_path: Path,
    monkeypatch_md5: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Corrupted existing file should be deleted before download."""
    metadata.path.parent.mkdir(parents=True, exist_ok=True)
    metadata.path.write_bytes(b"CORRUPTED")

    def fake_stream_download(url: Any, tmp_path: Path) -> None:
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

    path = ensure_dataset_npz(metadata, retries=1)  # type: ignore[arg-type]

    assert path.exists()
    assert path.suffix == ".npz"


@pytest.mark.unit
def test_ensure_dataset_npz_md5_mismatch_raises_error(
    metadata: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MD5 mismatch should raise ValueError and retry."""
    call_count = {"count": 0}

    def fake_stream_download(url: Any, tmp_path: Path) -> None:
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

    def fake_md5(path: Any) -> str:
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

    with pytest.raises(OrchardDatasetError):
        ensure_dataset_npz(metadata, retries=2, delay=0.01)  # type: ignore[arg-type]

    assert call_count["count"] == 2


@pytest.mark.unit
def test_ensure_dataset_npz_retries_and_fails(
    metadata: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated failures should raise RuntimeError."""

    def always_fail(*args: object, **kwargs: object) -> None:
        raise requests.ConnectionError("network down")

    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher._stream_download",
        always_fail,
    )

    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher.time.sleep",
        lambda _: None,
    )

    with pytest.raises(OrchardDatasetError):
        ensure_dataset_npz(metadata, retries=2, delay=0.01)


@pytest.mark.unit
def test_ensure_dataset_npz_rate_limit_429(metadata: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Rate limit (429) should trigger exponential backoff."""
    call_count = {"count": 0}
    sleep_calls = []

    def fake_stream_download(url: Any, tmp_path: Path) -> None:
        call_count["count"] += 1
        exc = requests.HTTPError("429 Rate Limit")
        exc.response = SimpleNamespace(status_code=429)
        raise exc

    def fake_sleep(delay: Any) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher._stream_download",
        fake_stream_download,
    )
    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher.time.sleep",
        fake_sleep,
    )

    with pytest.raises(OrchardDatasetError):
        ensure_dataset_npz(metadata, retries=3, delay=1.0)

    assert call_count["count"] == 3

    assert len(sleep_calls) == 2
    assert sleep_calls[0] == pytest.approx(1.0)
    assert sleep_calls[1] == pytest.approx(4.0)


@pytest.mark.unit
def test_ensure_dataset_npz_cleans_up_tmp_on_error(
    metadata: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Temporary file should be cleaned up on download failure."""
    tmp_file_path = metadata.path.with_suffix(".tmp")

    def fake_stream_download(url: Any, tmp_path: Path) -> None:
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

    with pytest.raises(OrchardDatasetError):
        ensure_dataset_npz(metadata, retries=1, delay=0.01)  # type: ignore[arg-type]

    assert not tmp_file_path.exists()


@pytest.mark.unit
def test_ensure_dataset_npz_error_without_response_attribute(
    metadata: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Error without response attribute should use normal delay."""
    sleep_calls = []

    def fake_stream_download(url: Any, tmp_path: Path) -> None:
        raise OSError("Some error without response")

    def fake_sleep(delay: Any) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher._stream_download",
        fake_stream_download,
    )
    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher.time.sleep",
        fake_sleep,
    )

    with pytest.raises(OrchardDatasetError):
        ensure_dataset_npz(metadata, retries=2, delay=2.0)  # type: ignore[arg-type]

    assert sleep_calls[0] == pytest.approx(2.0)


# TEST: _stream_download
@pytest.mark.unit
def test_stream_download_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful download should write content to file."""
    output_path = tmp_path / "output.npz"
    test_content = b"test_npz_content"

    class FakeResponse:
        def __init__(self) -> None:
            self.headers = {"Content-type": "application/octet-stream"}

        def raise_for_status(self) -> None:
            """No-op: simulates a successful HTTP response (no error to raise)."""

        def iter_content(self, chunk_size: Any) -> Iterator[bytes]:
            yield test_content

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> Literal[False]:
            return False

    def fake_get(*args: object, **kwargs: object) -> FakeResponse:
        return FakeResponse()

    monkeypatch.setattr("requests.get", fake_get)

    _stream_download("https://example.com/file.npz", output_path)

    assert output_path.exists()
    assert output_path.read_bytes() == test_content


@pytest.mark.unit
def test_stream_download_html_content_raises_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HTML content instead of NPZ should raise ValueError."""
    output_path = tmp_path / "output.npz"

    class FakeResponse:
        def __init__(self) -> None:
            self.headers = {"Content-type": "text/html"}

        def raise_for_status(self) -> None:
            """No-op: simulates a successful HTTP response (no error to raise)."""

        def iter_content(self, chunk_size: Any) -> Iterator[bytes]:
            yield b"<html>Not Found</html>"

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> Literal[False]:
            return False

    def fake_get(*args: object, **kwargs: object) -> FakeResponse:
        return FakeResponse()

    monkeypatch.setattr("requests.get", fake_get)

    with pytest.raises(OrchardDatasetError, match="HTML page"):
        _stream_download("https://example.com/file.npz", output_path)


@pytest.mark.unit
def test_stream_download_skips_empty_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty chunks should be skipped during download."""
    output_path = tmp_path / "output.npz"

    class FakeResponse:
        def __init__(self) -> None:
            self.headers = {"Content-type": "application/octet-stream"}

        def raise_for_status(self) -> None:
            """No-op: simulates a successful HTTP response (no error to raise)."""

        def iter_content(self, chunk_size: Any) -> Iterator[bytes | None]:
            yield b"first"
            yield b""
            yield None
            yield b"second"

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> Literal[False]:
            return False

    def fake_get(*args: object, **kwargs: object) -> FakeResponse:
        return FakeResponse()

    monkeypatch.setattr("requests.get", fake_get)

    _stream_download("https://example.com/file.npz", output_path)

    assert output_path.exists()
    assert output_path.read_bytes() == b"firstsecond"


@pytest.mark.unit
def test_stream_download_http_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP errors should be raised."""
    output_path = tmp_path / "output.npz"

    class FakeResponse:
        def raise_for_status(self) -> None:
            raise requests.HTTPError("404 Not Found")

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> Literal[False]:
            return False

    def fake_get(*args: object, **kwargs: object) -> FakeResponse:
        return FakeResponse()

    monkeypatch.setattr("requests.get", fake_get)

    with pytest.raises(requests.HTTPError):
        _stream_download("https://example.com/file.npz", output_path)


# TEST: load_dataset
@pytest.mark.unit
def test_load_medmnist_rgb(
    metadata: SimpleNamespace,
    valid_npz_file: Path,
    monkeypatch_md5: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RGB dataset metadata should be inferred correctly."""

    metadata.path.write_bytes(valid_npz_file.read_bytes())

    monkeypatch.setattr(
        "orchard.data_handler.dispatcher.ensure_dataset_npz",
        lambda _: metadata.path,
    )

    data = load_dataset(metadata)  # type: ignore[arg-type]

    assert data.name == metadata.name
    assert data.is_rgb is True
    assert data.num_classes == 2
    assert data.path == metadata.path


@pytest.mark.unit
def test_load_medmnist_grayscale(
    metadata: SimpleNamespace,
    tmp_path: Path,
    monkeypatch_md5: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        "orchard.data_handler.dispatcher.ensure_dataset_npz",
        lambda _: path,
    )

    data = load_dataset(metadata)  # type: ignore[arg-type]

    assert data.is_rgb is False
    assert data.num_classes == 3


@pytest.mark.unit
def test_load_dataset_health_check_rgb(
    metadata: SimpleNamespace,
    valid_npz_file: Path,
    monkeypatch_md5: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Health check should work on RGB datasets."""
    metadata.path.write_bytes(valid_npz_file.read_bytes())

    monkeypatch.setattr(
        "orchard.data_handler.dispatcher.ensure_dataset_npz",
        lambda _: metadata.path,
    )

    data = load_dataset_health_check(metadata, chunk_size=5)  # type: ignore[arg-type]

    assert data.is_rgb is True
    assert data.num_classes == 2
    assert data.path == metadata.path


@pytest.mark.unit
def test_load_dataset_health_check_grayscale(
    metadata: SimpleNamespace,
    tmp_path: Path,
    monkeypatch_md5: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        "orchard.data_handler.dispatcher.ensure_dataset_npz",
        lambda _: path,
    )

    data = load_dataset_health_check(metadata, chunk_size=10)  # type: ignore[arg-type]

    assert data.is_rgb is False
    assert data.num_classes == 10


@pytest.mark.unit
def test_load_dataset_health_check_small_chunk(
    metadata: SimpleNamespace,
    tmp_path: Path,
    monkeypatch_md5: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        "orchard.data_handler.dispatcher.ensure_dataset_npz",
        lambda _: path,
    )

    data = load_dataset_health_check(metadata, chunk_size=3)  # type: ignore[arg-type]

    assert data.num_classes == 2


@pytest.mark.unit
@pytest.mark.parametrize("dataset_name", ["cifar10", "cifar100"])
def test_ensure_dataset_npz_cifar_routing(
    dataset_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CIFAR datasets should route to ensure_cifar_npz converter."""
    target_npz = tmp_path / f"{dataset_name}.npz"

    np.savez_compressed(
        target_npz,
        train_images=np.zeros((5, 32, 32, 3), dtype=np.uint8),
        train_labels=np.zeros((5, 1), dtype=np.int64),
        val_images=np.zeros((2, 32, 32, 3), dtype=np.uint8),
        val_labels=np.zeros((2, 1), dtype=np.int64),
        test_images=np.zeros((3, 32, 32, 3), dtype=np.uint8),
        test_labels=np.zeros((3, 1), dtype=np.int64),
    )

    cifar_metadata = SimpleNamespace(
        name=dataset_name,
        display_name=f"CIFAR-{dataset_name[-2:]}",
        url="torchvision",
        md5_checksum="",
        path=target_npz,
        native_resolution=32,
    )

    def mock_ensure_cifar_npz(metadata: Any) -> Path:
        return Path(metadata.path)

    monkeypatch.setattr(
        "orchard.data_handler.fetchers.ensure_cifar_npz",
        mock_ensure_cifar_npz,
    )

    result = ensure_dataset_npz(cifar_metadata)  # type: ignore[arg-type]

    assert result == target_npz


@pytest.mark.unit
def test_ensure_dataset_npz_galaxy10_converter_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Galaxy10 dataset should trigger converter import and return path."""
    target_npz = tmp_path / "galaxy10.npz"

    np.savez_compressed(
        target_npz,
        train_images=np.zeros((5, 10, 10, 3), dtype=np.uint8),
        train_labels=np.zeros((5, 1), dtype=np.int64),
        val_images=np.zeros((2, 10, 10, 3), dtype=np.uint8),
        val_labels=np.zeros((2, 1), dtype=np.int64),
        test_images=np.zeros((3, 10, 10, 3), dtype=np.uint8),
        test_labels=np.zeros((3, 1), dtype=np.int64),
    )

    galaxy10_metadata = SimpleNamespace(
        name="galaxy10",
        url="https://example.com/galaxy10.h5",
        md5_checksum="test_md5",
        path=target_npz,
        native_resolution=224,
    )

    def mock_ensure_galaxy10_npz(metadata: Any) -> Path:
        return Path(metadata.path)

    monkeypatch.setattr(
        "orchard.data_handler.fetchers.ensure_galaxy10_npz",
        mock_ensure_galaxy10_npz,
    )

    result = ensure_dataset_npz(galaxy10_metadata)  # type: ignore[arg-type]

    assert result == target_npz


# ─── MUTATION-KILLING TESTS: _stream_download ───


@pytest.mark.unit
def test_stream_download_passes_exact_kwargs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify requests.get receives url, headers, timeout, stream, allow_redirects."""
    output_path = tmp_path / "output.npz"
    captured: dict[str, Any] = {}

    class FakeResponse:
        def __init__(self) -> None:
            self.headers = {"Content-type": "application/octet-stream"}

        def raise_for_status(self) -> None:
            pass

        def iter_content(self, chunk_size: Any) -> Iterator[bytes]:
            yield b"data"

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> Literal[False]:
            return False

    def capturing_get(*args: object, **kwargs: object) -> FakeResponse:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeResponse()

    monkeypatch.setattr("requests.get", capturing_get)

    _stream_download("https://example.com/file.npz", output_path)

    assert captured["args"] == ("https://example.com/file.npz",)
    assert captured["kwargs"]["timeout"] == 60
    assert captured["kwargs"]["stream"] is True
    assert captured["kwargs"]["allow_redirects"] is True

    headers = captured["kwargs"]["headers"]
    assert headers is not None
    assert headers["User-Agent"] == "Wget/1.0"
    assert headers["Accept"] == "application/octet-stream"
    assert headers["Accept-Encoding"] == "identity"


@pytest.mark.unit
def test_stream_download_content_type_missing_no_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing Content-type header should not raise (not HTML)."""
    output_path = tmp_path / "output.npz"

    class FakeResponse:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def raise_for_status(self) -> None:
            pass

        def iter_content(self, chunk_size: Any) -> Iterator[bytes]:
            yield b"data"

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> Literal[False]:
            return False

    monkeypatch.setattr("requests.get", lambda *a, **kw: FakeResponse())

    _stream_download("https://example.com/file.npz", output_path)

    assert output_path.read_bytes() == b"data"


# ─── MUTATION-KILLING TESTS: _is_valid_npz ───


@pytest.mark.unit
def test_is_valid_npz_passes_path_to_md5(
    valid_npz_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """md5_checksum should receive the actual file path, not None."""
    received_paths: list[Any] = []

    def tracking_md5(path: Any) -> str:
        received_paths.append(path)
        return "correct_md5"

    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher.md5_checksum",
        tracking_md5,
    )

    _is_valid_npz(valid_npz_file, "correct_md5")

    assert received_paths == [valid_npz_file]


# ─── MUTATION-KILLING TESTS: _retry_delay ───


@pytest.mark.unit
def test_retry_delay_429_quadratic_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """429 response should produce quadratic delay and log the delay value."""
    exc = requests.HTTPError("429 Rate Limit")
    exc.response = SimpleNamespace(status_code=429)

    log_calls: list[str] = []
    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher.logger.warning",
        lambda msg, *args: log_calls.append(msg % args if args else msg),
    )

    result = _retry_delay(exc, base_delay=2.0, attempt=3)

    assert result == pytest.approx(18.0)
    assert any("18.0" in call for call in log_calls)


@pytest.mark.unit
def test_retry_delay_non_429_returns_base() -> None:
    """Non-429 error should return base delay without warning."""
    exc = OSError("generic")

    result = _retry_delay(exc, base_delay=5.0, attempt=3)

    assert result == pytest.approx(5.0)


# ─── MUTATION-KILLING TESTS: ensure_medmnist_npz ───


@pytest.mark.unit
def test_ensure_medmnist_npz_passes_url_to_stream_download(
    metadata: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_stream_download should receive metadata.url, not None."""
    received_urls: list[str] = []

    def tracking_download(url: Any, tmp_path: Path) -> None:
        received_urls.append(url)
        real_npz = tmp_path.with_suffix(".npz")
        np.savez(real_npz, train_images=np.zeros((2, 28, 28)))
        tmp_path.write_bytes(real_npz.read_bytes())

    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher._stream_download",
        tracking_download,
    )
    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher.md5_checksum",
        lambda _: "correct_md5",
    )

    ensure_medmnist_npz(metadata, retries=1)  # type: ignore[arg-type]

    assert received_urls == ["https://example.com/fake.npz"]


@pytest.mark.unit
def test_ensure_medmnist_npz_mkdir_parents(metadata: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Parent directory creation should use parents=True."""
    # Use a deeply nested path
    deep_path = metadata.path.parent / "sub" / "deep" / "dataset.npz"
    metadata.path = deep_path

    def fake_download(url: Any, tmp_path: Path) -> None:
        real_npz = tmp_path.with_suffix(".npz")
        np.savez(real_npz, train_images=np.zeros((2, 28, 28)))
        tmp_path.write_bytes(real_npz.read_bytes())

    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher._stream_download",
        fake_download,
    )
    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher.md5_checksum",
        lambda _: "correct_md5",
    )

    result = ensure_medmnist_npz(metadata, retries=1)

    assert result.exists()
    assert deep_path.parent.exists()


@pytest.mark.unit
def test_ensure_medmnist_npz_md5_mismatch_error_message(
    metadata: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MD5 mismatch should raise with specific error message."""

    def fake_download(url: Any, tmp_path: Path) -> None:
        real_npz = tmp_path.with_suffix(".npz")
        np.savez(real_npz, train_images=np.zeros((2, 28, 28)))
        tmp_path.write_bytes(real_npz.read_bytes())

    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher._stream_download",
        fake_download,
    )
    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher.md5_checksum",
        lambda _: "wrong_md5",
    )
    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher.time.sleep",
        lambda _: None,
    )

    with pytest.raises(OrchardDatasetError, match="Could not download"):
        ensure_medmnist_npz(metadata, retries=1, delay=0.01)  # type: ignore[arg-type]


@pytest.mark.unit
def test_ensure_medmnist_npz_corrupted_file_warning(
    metadata: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Corrupted file should log warning with the path."""
    metadata.path.parent.mkdir(parents=True, exist_ok=True)
    metadata.path.write_bytes(b"CORRUPTED")

    warning_calls: list[str] = []

    def fake_download(url: Any, tmp_path: Path) -> None:
        real_npz = tmp_path.with_suffix(".npz")
        np.savez(real_npz, train_images=np.zeros((2, 28, 28)))
        tmp_path.write_bytes(real_npz.read_bytes())

    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher._stream_download",
        fake_download,
    )
    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher.md5_checksum",
        lambda _: "correct_md5",
    )
    original_warning = __import__(
        "orchard.data_handler.fetchers.medmnist_fetcher", fromlist=["logger"]
    ).logger.warning

    def tracking_warning(msg: Any, *args: object) -> None:
        warning_calls.append(msg % args if args else msg)
        original_warning(msg, *args)

    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher.logger.warning",
        tracking_warning,
    )

    ensure_medmnist_npz(metadata, retries=1)  # type: ignore[arg-type]

    assert any(str(metadata.path) in call for call in warning_calls)


@pytest.mark.unit
def test_ensure_medmnist_npz_retry_logging(metadata: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry warnings should include attempt number, total retries, error, and delay."""
    call_count = {"n": 0}
    warning_calls: list[str] = []

    def failing_then_ok(url: Any, tmp_path: Path) -> None:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise OSError("network glitch")
        real_npz = tmp_path.with_suffix(".npz")
        np.savez(real_npz, train_images=np.zeros((2, 28, 28)))
        tmp_path.write_bytes(real_npz.read_bytes())

    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher._stream_download",
        failing_then_ok,
    )
    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher.md5_checksum",
        lambda _: "correct_md5",
    )
    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher.time.sleep",
        lambda _: None,
    )
    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher.logger.warning",
        lambda msg, *args: warning_calls.append(msg % args if args else msg),
    )

    ensure_medmnist_npz(metadata, retries=2, delay=3.0)

    combined = " ".join(warning_calls)
    assert "1/2" in combined
    assert "network glitch" in combined
    assert "3.0" in combined


@pytest.mark.unit
def test_ensure_medmnist_npz_final_failure_error_log(
    metadata: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Final failure should log error with retry count and raise with dataset name."""
    error_calls: list[str] = []

    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher._stream_download",
        lambda *a: (_ for _ in ()).throw(OSError("fail")),
    )
    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher.time.sleep",
        lambda _: None,
    )
    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher.logger.error",
        lambda msg, *args: error_calls.append(msg % args if args else msg),
    )

    with pytest.raises(OrchardDatasetError, match="test_medmnist"):
        ensure_medmnist_npz(metadata, retries=2, delay=0.01)  # type: ignore[arg-type]

    combined = " ".join(error_calls)
    assert "2" in combined


@pytest.mark.unit
def test_ensure_medmnist_npz_md5_failure_error_log(
    metadata: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MD5 validation failure should log error with expected and actual MD5."""
    error_calls: list[str] = []

    def fake_download(url: Any, tmp_path: Path) -> None:
        real_npz = tmp_path.with_suffix(".npz")
        np.savez(real_npz, train_images=np.zeros((2, 28, 28)))
        tmp_path.write_bytes(real_npz.read_bytes())

    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher._stream_download",
        fake_download,
    )
    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher.md5_checksum",
        lambda _: "bad_hash_xyz",
    )
    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher.time.sleep",
        lambda _: None,
    )
    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher.logger.error",
        lambda msg, *args: error_calls.append(msg % args if args else msg),
    )

    with pytest.raises(OrchardDatasetError):
        ensure_medmnist_npz(metadata, retries=1, delay=0.01)  # type: ignore[arg-type]

    combined = " ".join(error_calls)
    assert "correct_md5" in combined
    assert "bad_hash_xyz" in combined


@pytest.mark.unit
def test_ensure_medmnist_npz_retries_exact_count(
    metadata: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ensure_medmnist_npz should retry exactly `retries` times."""
    call_count = {"n": 0}

    def counting_download(url: Any, tmp_path: Path) -> None:
        call_count["n"] += 1
        raise OSError("fail")

    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher._stream_download",
        counting_download,
    )
    monkeypatch.setattr(
        "orchard.data_handler.fetchers.medmnist_fetcher.time.sleep",
        lambda _: None,
    )

    with pytest.raises(OrchardDatasetError):
        ensure_medmnist_npz(metadata, retries=3, delay=0.01)

    assert call_count["n"] == 3


# ── PennFudan dispatcher route ───────────────────────────────────────────────


@pytest.mark.unit
def test_ensure_dataset_npz_routes_pennfudan(tmp_path: Path) -> None:
    """ensure_dataset_npz routes 'pennfudan' to ensure_pennfudan_npz."""
    from unittest.mock import patch

    metadata = SimpleNamespace(name="pennfudan", path=tmp_path / "img.npz")

    with patch(
        "orchard.data_handler.fetchers.ensure_pennfudan_npz",
        return_value=tmp_path / "img.npz",
    ) as mock_pf:
        result = ensure_dataset_npz(metadata)  # type: ignore[arg-type]

    mock_pf.assert_called_once_with(metadata)
    assert result == tmp_path / "img.npz"


@pytest.mark.unit
def test_load_and_inspect_detection_uses_metadata_num_classes(tmp_path: Path) -> None:
    """Detection path reads num_classes from metadata, not from labels."""
    from unittest.mock import patch

    from orchard.data_handler.dispatcher import _load_and_inspect

    # Create a minimal images-only NPZ (no train_labels)
    images = np.random.randint(0, 255, (5, 32, 32, 3), dtype=np.uint8)
    npz_path = tmp_path / "images.npz"
    np.savez(npz_path, train_images=images)

    metadata = SimpleNamespace(
        name="pennfudan",
        path=npz_path,
        annotation_path=tmp_path / "annotations.npz",
        num_classes=1,
    )

    with patch(
        "orchard.data_handler.dispatcher.ensure_dataset_npz",
        return_value=npz_path,
    ):
        result = _load_and_inspect(metadata)  # type: ignore[arg-type]

    assert result.num_classes == 1
    assert result.annotation_path == tmp_path / "annotations.npz"
    assert result.is_rgb is True


# ─── MUTATION-KILLING TESTS: ensure_dataset_npz / _load_and_inspect / health_check ───


@pytest.mark.unit
def test_ensure_dataset_npz_default_retries_and_delay(
    metadata: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default retries=5 and delay=5.0 are forwarded to ensure_medmnist_npz."""
    captured: dict[str, Any] = {}

    def fake_medmnist(meta: Any, retries: int = 5, delay: float = 5.0) -> Path:
        captured["retries"] = retries
        captured["delay"] = delay
        return Path(meta.path)

    monkeypatch.setattr("orchard.data_handler.fetchers.ensure_medmnist_npz", fake_medmnist)

    # Call without explicit retries/delay so the defaults are used
    ensure_dataset_npz(metadata)  # type: ignore[arg-type]

    # Kill mutmut_1: retries 5 → 6
    assert captured["retries"] == 5
    # Kill mutmut_2: delay 5.0 → 6.0
    assert captured["delay"] == pytest.approx(5.0)


@pytest.mark.unit
def test_load_and_inspect_passes_metadata_not_none(
    metadata: SimpleNamespace,
    valid_npz_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_load_and_inspect passes the metadata object (not None) to ensure_dataset_npz."""
    from orchard.data_handler.dispatcher import _load_and_inspect

    metadata.path.write_bytes(valid_npz_file.read_bytes())
    called_with: dict[str, Any] = {}

    def capturing_ensure(meta: Any) -> Path:
        called_with["meta"] = meta
        return Path(metadata.path)

    monkeypatch.setattr("orchard.data_handler.dispatcher.ensure_dataset_npz", capturing_ensure)

    _load_and_inspect(metadata)  # type: ignore[arg-type]

    # Kill mutmut_2: ensure_dataset_npz(metadata) → ensure_dataset_npz(None)
    assert called_with["meta"] is not None


@pytest.mark.unit
def test_load_and_inspect_4d_grayscale_is_not_rgb(
    metadata: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """4D grayscale (N, H, W, 1) must yield is_rgb=False — kills and→or mutant."""
    path = tmp_path / "gray4d.npz"
    np.savez(
        path,
        train_images=np.zeros((20, 28, 28, 1), dtype=np.uint8),
        train_labels=np.arange(20),
        val_images=np.zeros((5, 28, 28, 1), dtype=np.uint8),
        val_labels=np.arange(5),
        test_images=np.zeros((5, 28, 28, 1), dtype=np.uint8),
        test_labels=np.arange(5),
    )
    metadata.path = path

    monkeypatch.setattr(
        "orchard.data_handler.dispatcher.ensure_dataset_npz",
        lambda _: path,
    )

    data = load_dataset(metadata)  # type: ignore[arg-type]

    # ndim==4 and shape[-1]==1: and→False, or→True
    assert data.is_rgb is False


@pytest.mark.unit
def test_load_dataset_health_check_default_chunk_size_is_100(
    metadata: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """load_dataset_health_check defaults chunk_size to 100 (not 101)."""
    from orchard.data_handler.dispatcher import DatasetData

    captured: dict[str, Any] = {}

    def fake_load_and_inspect(meta: Any, chunk_size: int | None = None) -> DatasetData:
        captured["chunk_size"] = chunk_size
        return DatasetData(path=Path("/fake"), name="test", is_rgb=True, num_classes=2)

    monkeypatch.setattr("orchard.data_handler.dispatcher._load_and_inspect", fake_load_and_inspect)

    load_dataset_health_check(metadata)  # type: ignore[arg-type]

    # Kill mutmut_1: chunk_size 100 → 101
    assert captured["chunk_size"] == 100
