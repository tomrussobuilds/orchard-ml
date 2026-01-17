"""
Pytest Configuration and Shared Fixtures for VisionForge Test Suite.

This module provides reusable test fixtures for configuration testing, including:
- Mock dataset metadata for different resolutions (28×28, 224×224)
- CLI argument namespaces for training and optimization
- Temporary YAML configuration files for integration tests

Fixtures are automatically discovered by pytest across all test modules.
"""

# =========================================================================== #
#                              STANDARD LIBRARY                               #
# =========================================================================== #
import argparse
from pathlib import Path

# =========================================================================== #
#                            THIRD-PARTY IMPORTS                              #
# =========================================================================== #
import pytest

# =========================================================================== #
#                              LOCAL IMPORTS                                  #
# =========================================================================== #
from orchard.core.metadata import DatasetMetadata


# =========================================================================== #
#                          DATASET METADATA FIXTURES                          #
# =========================================================================== #

@pytest.fixture
def mock_metadata_28():
    """Mock 28×28 dataset metadata for testing low-resolution workflows."""
    return DatasetMetadata(
        name="bloodmnist",
        display_name="BloodMNIST",
        md5_checksum="test123",
        url="https://example.com/bloodmnist.npz",
        path=Path("/tmp/bloodmnist_28.npz"),
        classes=[f"class_{i}" for i in range(8)],
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
        in_channels=3,
        native_resolution=28,
        is_anatomical=False,
        is_texture_based=False
    )


@pytest.fixture
def mock_metadata_224():
    """Mock 224×224 dataset metadata for testing high-resolution workflows."""
    return DatasetMetadata(
        name="organcmnist",
        display_name="OrganCMNIST",
        md5_checksum="test456",
        url="https://example.com/organcmnist.npz",
        path=Path("/tmp/organcmnist_224.npz"),
        classes=[f"organ_{i}" for i in range(11)],
        mean=(0.5,),
        std=(0.5,),
        in_channels=1,
        native_resolution=224,
        is_anatomical=True,
        is_texture_based=False
    )


@pytest.fixture
def mock_grayscale_metadata():
    """Mock grayscale dataset metadata for testing channel conversion logic."""
    return DatasetMetadata(
        name="pneumoniamnist",
        display_name="PneumoniaMNIST",
        md5_checksum="test789",
        url="https://example.com/pneumoniamnist.npz",
        path=Path("/tmp/pneumoniamnist_28.npz"),
        classes=["normal", "pneumonia"],
        mean=(0.5,),
        std=(0.5,),
        in_channels=1,
        native_resolution=28,
        is_anatomical=True,
        is_texture_based=False
    )


# =========================================================================== #
#                        CLI ARGUMENT FIXTURES                                #
# =========================================================================== #

@pytest.fixture
def basic_args():
    """Standard CLI arguments for training pipeline testing."""
    return argparse.Namespace(
        # Dataset
        dataset="bloodmnist",
        data_dir="./dataset",
        resolution=28,
        max_samples=None,
        use_weighted_sampler=True,
        force_rgb=True,
        img_size=None,
        # Model
        model_name="resnet_18_adapted",
        pretrained=True,
        # Training
        epochs=60,
        batch_size=128,
        learning_rate=0.001,
        min_lr=1e-6,
        weight_decay=5e-4,
        momentum=0.9,
        scheduler_patience=5,
        cosine_fraction=0.8,
        use_amp=False,
        # Regularization
        mixup_alpha=0.0,
        mixup_epochs=None,
        label_smoothing=0.0,
        dropout=0.0,
        # Augmentation
        hflip=0.5,
        rotation_angle=10,
        jitter_val=0.2,
        min_scale=0.95,
        no_tta=False,
        # System
        seed=42,
        reproducible=False,
        num_workers=4,
        device=None,
        # Paths
        output_root=None,
        config=None,  # Not loading from YAML
    )


@pytest.fixture
def optuna_args():
    """CLI arguments for Optuna hyperparameter optimization testing."""
    return argparse.Namespace(
        dataset="bloodmnist",
        resolution=28,
        study_name="test_study",
        n_trials=10,
        epochs=15,
        metric_name="auc",
        direction="maximize"
    )


# =========================================================================== #
#                        YAML CONFIGURATION FIXTURES                          #
# =========================================================================== #

@pytest.fixture
def temp_yaml_config(tmp_path):
    """Valid YAML configuration file for integration testing."""
    yaml_content = """
dataset:
  name: bloodmnist
  resolution: 28
model:
  name: resnet_18_adapted
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