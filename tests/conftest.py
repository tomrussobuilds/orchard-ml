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


# YAML CONFIGURATION FIXTURES
@pytest.fixture
def temp_yaml_config(tmp_path):
    """Valid YAML configuration file for integration testing."""
    yaml_content = """
dataset:
  name: bloodmnist
  resolution: 28
model:
  name: resnet_18
  pretrained: true
training:
  epochs: 60
  batch_size: 128
  learning_rate: 0.008
optuna:
  study_name: yaml_test_study
  n_trials: 20
"""
    yaml_file = tmp_path / "test_config.yaml"
    yaml_file.write_text(yaml_content)
    return yaml_file


@pytest.fixture
def temp_invalid_yaml(tmp_path):
    """Invalid YAML configuration for validation error testing."""
    yaml_content = """
training:
  epochs: 60
  min_lr: 10.0  # Invalid: min_lr must be < learning_rate
  learning_rate: 0.001
"""
    yaml_file = tmp_path / "invalid.yaml"
    yaml_file.write_text(yaml_content)
    return yaml_file


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
        telemetry={"data_dir": "./dataset", "output_dir": "./outputs"},
    )
