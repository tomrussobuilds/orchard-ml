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
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from orchard.core import Config, OptunaConfig, TrainingConfig
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


# CONFIG FACTORIES
def make_training_config(**overrides: Any) -> TrainingConfig:
    """
    Build a real TrainingConfig with test-friendly defaults.

    Overrides ``use_amp=False`` (no CUDA in tests) and ``mixup_alpha=0``
    (no augmentation noise).  Callers can override any field.
    """
    defaults: dict[str, Any] = {
        "use_amp": False,
        "mixup_alpha": 0,
        "mixup_epochs": 0,
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def make_optuna_config(**overrides: Any) -> OptunaConfig:
    """
    Build a real OptunaConfig with test-friendly defaults.

    Overrides nothing beyond Pydantic defaults; callers supply
    only the fields they care about.
    """
    return OptunaConfig(**overrides)


def make_dummy_loader(
    in_features: int = 10, num_classes: int = 2, n_samples: int = 4, batch_size: int = 2
) -> DataLoader[Any]:
    """
    Build a real DataLoader with random tensors.

    Useful wherever tests patch ``train_one_epoch`` / ``validate_epoch``
    so the loader is never actually iterated, but Pylance still wants a
    correctly-typed object.
    """
    x = torch.randn(n_samples, in_features)
    y = torch.randint(0, num_classes, (n_samples,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


class TrainingBundle:
    """
    Convenience container for the objects that almost every
    TrialTrainingExecutor / ModelTrainer test needs.

    Attributes:
        model: A small ``nn.Linear`` module.
        optimizer: SGD on ``model.parameters()``.
        scheduler: ``StepLR`` with ``step_size=1``.
        criterion: ``CrossEntropyLoss``.
        train_loader: Dummy ``DataLoader``.
        val_loader: Dummy ``DataLoader``.
        device: ``torch.device("cpu")``.
    """

    def __init__(self, in_features: int = 10, num_classes: int = 2) -> None:
        self.model = nn.Linear(in_features, num_classes)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader: DataLoader[Any] = make_dummy_loader(in_features, num_classes)
        self.val_loader: DataLoader[Any] = make_dummy_loader(in_features, num_classes)
        self.device = torch.device("cpu")


# DATASET METADATA FIXTURES
@pytest.fixture
def mock_metadata_28(tmp_path: Path) -> DatasetMetadata:
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
def mock_metadata_224(tmp_path: Path) -> DatasetMetadata:
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
def mock_grayscale_metadata(tmp_path: Path) -> DatasetMetadata:
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
def mock_metadata_many_classes(tmp_path: Path) -> DatasetMetadata:
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
def minimal_config() -> Config:
    """Minimal valid Config for testing."""
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
