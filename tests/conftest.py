"""
Pytest Configuration and Shared Fixtures for Orchard ML Test Suite.

This module provides reusable test fixtures for configuration testing, including:
- Mock dataset metadata for different resolutions (28×28, 224×224)
- CLI argument namespaces for training and optimization
- Temporary YAML configuration files for integration tests

Fixtures are automatically discovered by pytest across all test modules.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from orchard.core.metadata import DatasetMetadata


def mutmut_safe_env(**extra: str) -> dict[str, str]:
    """
    Build an env dict that preserves ``MUTANT_UNDER_TEST``.

    Use with ``patch.dict(os.environ, mutmut_safe_env(...), clear=True)``
    so that mutmut v3 trampolines keep working even when the test
    needs a clean environment.
    """
    env: dict[str, str] = {}
    mut = os.environ.get("MUTANT_UNDER_TEST")
    if mut is not None:
        env["MUTANT_UNDER_TEST"] = mut
    env.update(extra)
    return env


# DATASET METADATA FIXTURES
@pytest.fixture
def mock_metadata_28(tmp_path):
    """Mock 28×28 dataset metadata for testing low-resolution workflows."""
    return DatasetMetadata(
        name="bloodmnist",
        display_name="BloodMNIST",
        md5_checksum="test123",
        url="https://example.com/bloodmnist.npz",
        path=tmp_path / "bloodmnist_28.npz",
        classes=[f"class_{i}" for i in range(8)],
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
        in_channels=3,
        native_resolution=28,
        is_anatomical=False,
        is_texture_based=False,
    )


@pytest.fixture
def mock_metadata_224(tmp_path):
    """Mock 224×224 dataset metadata for testing high-resolution workflows."""
    return DatasetMetadata(
        name="organcmnist",
        display_name="OrganCMNIST",
        md5_checksum="test456",
        url="https://example.com/organcmnist.npz",
        path=tmp_path / "organcmnist_224.npz",
        classes=[f"organ_{i}" for i in range(11)],
        mean=(0.5,),
        std=(0.5,),
        in_channels=1,
        native_resolution=224,
        is_anatomical=True,
        is_texture_based=False,
    )


@pytest.fixture
def mock_grayscale_metadata(tmp_path):
    """Mock grayscale dataset metadata for testing channel conversion logic."""
    return DatasetMetadata(
        name="pneumoniamnist",
        display_name="PneumoniaMNIST",
        md5_checksum="test789",
        url="https://example.com/pneumoniamnist.npz",
        path=tmp_path / "pneumoniamnist_28.npz",
        classes=["normal", "pneumonia"],
        mean=(0.5,),
        std=(0.5,),
        in_channels=1,
        native_resolution=28,
        is_anatomical=True,
        is_texture_based=False,
    )


@pytest.fixture
def mock_metadata_many_classes(tmp_path):
    """Mock dataset with many classes for min dataset size validation tests."""
    return DatasetMetadata(
        name="organamnist",
        display_name="OrganAMNIST",
        md5_checksum="test_many",
        url="https://example.com/organamnist.npz",
        path=tmp_path / "organamnist_28.npz",
        classes=[f"organ_{i}" for i in range(50)],
        mean=(0.5,),
        std=(0.5,),
        in_channels=1,
        native_resolution=28,
        is_anatomical=True,
        is_texture_based=False,
    )


# MINIMAL CONFIG
@pytest.fixture
def minimal_config():
    """Minimal valid Config for testing."""
    from orchard.core import Config

    return Config(
        dataset={"name": "bloodmnist", "resolution": 28},
        architecture={"name": "resnet_18", "pretrained": False},
        training={
            "epochs": 25,
            "batch_size": 16,
            "learning_rate": 0.001,
            "use_amp": False,
        },
        hardware={"device": "cpu", "project_name": "test-project"},
        telemetry={"output_dir": "./outputs"},
    )


# FILESYSTEM ISOLATION
@pytest.fixture()
def safe_outputs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Redirect OUTPUTS_ROOT to tmp_path so RunPaths.create() without
    an explicit base_dir never writes to the real filesystem.
    """
    monkeypatch.setattr("orchard.core.paths.run_paths.OUTPUTS_ROOT", tmp_path)
    return tmp_path
