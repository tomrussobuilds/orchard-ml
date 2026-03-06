"""
MedMNIST Dataset Fetcher.

Downloads MedMNIST NPZ files with robust retry logic, MD5 verification,
and atomic file operations. Follows the same pattern as ``galaxy10_converter``
to keep each domain's fetch logic self-contained.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

from ...core import DatasetMetadata, LogStyle, md5_checksum
from ...core.paths import LOGGER_NAME
from ...exceptions import OrchardDatasetError

logger = logging.getLogger(LOGGER_NAME)


def ensure_medmnist_npz(
    metadata: DatasetMetadata,
    retries: int = 5,
    delay: float = 5.0,
) -> Path:
    """
    Downloads a MedMNIST NPZ file with retries and MD5 validation.

    Implements a three-phase strategy:
        1. Return immediately if a valid local copy already exists.
        2. Delete any corrupted local copy.
        3. Stream-download with retry loop and atomic file replacement.

    Args:
        metadata (DatasetMetadata): Metadata containing URL, MD5, name and target path.
        retries (int): Max number of download attempts.
        delay (float): Base delay (seconds) between retries (quadratic backoff on 429).

    Returns:
        Path: Path to the successfully validated .npz file.

    Raises:
        OrchardDatasetError: If all download attempts fail.
    """
    target_npz = metadata.path

    # 1. Validation of existing file
    if _is_valid_npz(target_npz, metadata.md5_checksum):
        logger.debug(
            "%s%s %-18s: '%s' found at %s",
            LogStyle.INDENT,
            LogStyle.ARROW,
            "Dataset",
            metadata.name,
            target_npz.name,
        )
        return target_npz

    # 2. Cleanup corrupted file
    if target_npz.exists():
        logger.warning("Corrupted dataset found, deleting: %s", target_npz)
        target_npz.unlink()

    # 3. Download logic with retries
    logger.info("%s%s %-18s: %s", LogStyle.INDENT, LogStyle.ARROW, "Downloading", metadata.name)
    target_npz.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_npz.with_suffix(".tmp")

    for attempt in range(1, retries + 1):
        try:
            _stream_download(metadata.url, tmp_path)

            if not _is_valid_npz(tmp_path, metadata.md5_checksum):
                actual_md5 = md5_checksum(tmp_path)
                logger.error("MD5 mismatch: expected %s, got %s", metadata.md5_checksum, actual_md5)
                raise OrchardDatasetError("Downloaded file failed MD5 or header validation")

            # Atomic move
            tmp_path.replace(target_npz)
            logger.info(
                "%s%s %-18s: %s", LogStyle.INDENT, LogStyle.SUCCESS, "Verified", metadata.name
            )
            return target_npz

        except (OrchardDatasetError, OSError) as e:
            if tmp_path.exists():
                tmp_path.unlink()

            if attempt == retries:
                logger.error("Download failed after %d attempts", retries)
                raise OrchardDatasetError(f"Could not download {metadata.name}") from e

            actual_delay = _retry_delay(e, delay, attempt)
            logger.warning(
                "Attempt %d/%d failed: %s. Retrying in %ss...", attempt, retries, e, actual_delay
            )
            time.sleep(actual_delay)

    raise OrchardDatasetError("Unexpected error in dataset download logic.")  # pragma: no cover


# PRIVATE HELPERS
def _retry_delay(exc: Exception, base_delay: float, attempt: int) -> float:
    """
    Compute retry delay with quadratic backoff for 429 responses.
    """
    if hasattr(exc, "response") and exc.response is not None and exc.response.status_code == 429:
        delay = base_delay * (attempt**2)
        logger.warning("Rate limited (429). Waiting %ss before retrying...", delay)
        return delay
    return base_delay


def _is_valid_npz(path: Path, expected_md5: str) -> bool:
    """
    Checks file existence, header (ZIP/NPZ), and MD5 checksum.
    """
    if not path.exists():
        return False
    try:
        # Check for ZIP header (NPZ files are ZIP archives)
        with open(path, "rb") as f:
            if f.read(2) != b"PK":
                return False
    except IOError:
        return False

    return md5_checksum(path) == expected_md5


def _stream_download(url: str, tmp_path: Path, chunk_size: int = 8192) -> None:
    """
    Executes the streaming GET request and writes to a temporary file.
    """
    headers = {
        "User-Agent": "Wget/1.0",
        "Accept": "application/octet-stream",
        "Accept-Encoding": "identity",
    }

    with requests.get(url, headers=headers, timeout=60, stream=True, allow_redirects=True) as r:
        r.raise_for_status()

        content_type = r.headers.get("Content-type", "")
        if "text/html" in content_type:
            raise OrchardDatasetError("Downloaded file is an HTML page, not the expected NPZ file.")

        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
