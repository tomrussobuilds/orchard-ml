"""
Test Suite for Distributed Environment Detection Utilities.

Tests rank, local_rank, world_size detection from environment variables
and the is_distributed / is_main_process helpers.
"""

from __future__ import annotations

import pytest

from orchard.core.environment.distributed import (
    get_local_rank,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
)


# GET_RANK
@pytest.mark.unit
def test_get_rank_defaults_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_rank returns 0 when RANK env var is not set."""
    monkeypatch.delenv("RANK", raising=False)
    assert get_rank() == 0


@pytest.mark.unit
def test_get_rank_reads_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_rank reads RANK env var correctly."""
    monkeypatch.setenv("RANK", "3")
    assert get_rank() == 3


@pytest.mark.unit
def test_get_rank_returns_zero_for_rank_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_rank returns 0 when RANK=0 is set."""
    monkeypatch.setenv("RANK", "0")
    assert get_rank() == 0


# GET_LOCAL_RANK
@pytest.mark.unit
def test_get_local_rank_defaults_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_local_rank returns 0 when LOCAL_RANK env var is not set."""
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert get_local_rank() == 0


@pytest.mark.unit
def test_get_local_rank_reads_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_local_rank reads LOCAL_RANK env var correctly."""
    monkeypatch.setenv("LOCAL_RANK", "2")
    assert get_local_rank() == 2


# GET_WORLD_SIZE
@pytest.mark.unit
def test_get_world_size_defaults_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_world_size returns 1 when WORLD_SIZE env var is not set."""
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert get_world_size() == 1


@pytest.mark.unit
def test_get_world_size_reads_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_world_size reads WORLD_SIZE env var correctly."""
    monkeypatch.setenv("WORLD_SIZE", "8")
    assert get_world_size() == 8


# IS_DISTRIBUTED
@pytest.mark.unit
def test_is_distributed_false_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test is_distributed returns False when no RANK/LOCAL_RANK set."""
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert is_distributed() is False


@pytest.mark.unit
def test_is_distributed_true_with_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test is_distributed returns True when RANK is set."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert is_distributed() is True


@pytest.mark.unit
def test_is_distributed_true_with_local_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test is_distributed returns True when LOCAL_RANK is set."""
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setenv("LOCAL_RANK", "0")
    assert is_distributed() is True


@pytest.mark.unit
def test_is_distributed_true_with_both(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test is_distributed returns True when both RANK and LOCAL_RANK are set."""
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    assert is_distributed() is True


# IS_MAIN_PROCESS
@pytest.mark.unit
def test_is_main_process_true_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test is_main_process returns True when no RANK set (single-process)."""
    monkeypatch.delenv("RANK", raising=False)
    assert is_main_process() is True


@pytest.mark.unit
def test_is_main_process_true_for_rank_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test is_main_process returns True when RANK=0."""
    monkeypatch.setenv("RANK", "0")
    assert is_main_process() is True


@pytest.mark.unit
def test_is_main_process_false_for_non_zero_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test is_main_process returns False when RANK>0."""
    monkeypatch.setenv("RANK", "1")
    assert is_main_process() is False


@pytest.mark.unit
def test_is_main_process_false_for_high_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test is_main_process returns False for high rank values."""
    monkeypatch.setenv("RANK", "7")
    assert is_main_process() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
