"""
Pytest guard for smoke_test.py and health_check.py.

These scripts are standalone CI utilities (not part of the pytest suite)
that run real training. This module validates their imports and Config
construction so breakage is caught by pytest before CI runs them.
"""

from __future__ import annotations

from orchard.core import Config
from orchard.data_handler.synthetic import create_synthetic_dataset
from tests.health_check import _build_health_check_config
from tests.smoke_test import _build_smoke_config

# --- smoke_test ---


def test_smoke_config_builds_default():
    """Default smoke config should produce a valid frozen Config."""
    cfg = _build_smoke_config()
    assert isinstance(cfg, Config)
    assert cfg.dataset.name == "bloodmnist"
    assert cfg.architecture.name == "mini_cnn"
    assert cfg.training.epochs == 1
    assert cfg.hardware.device == "cpu"


def test_smoke_config_custom_architecture():
    """Smoke config should accept a custom architecture name."""
    cfg = _build_smoke_config(architecture="resnet_18")
    assert cfg.architecture.name == "resnet_18"


def test_smoke_config_custom_dataset():
    """Smoke config should accept a custom dataset name."""
    cfg = _build_smoke_config(dataset="dermamnist")
    assert cfg.dataset.name == "dermamnist"


# --- health_check ---


def test_health_check_config_28():
    """Health check config at 28px should use mini_cnn."""
    cfg = _build_health_check_config("bloodmnist", 28)
    assert isinstance(cfg, Config)
    assert cfg.architecture.name == "mini_cnn"
    assert cfg.dataset.resolution == 28


def test_health_check_config_224():
    """Health check config at 224px should use efficientnet_b0."""
    cfg = _build_health_check_config("bloodmnist", 224)
    assert cfg.architecture.name == "efficientnet_b0"
    assert cfg.dataset.resolution == 224


# --- synthetic (CI data path) ---


def test_synthetic_dataset_plugs_into_smoke_config():
    """Synthetic dataset should be compatible with smoke config pipeline."""
    cfg = _build_smoke_config()
    data = create_synthetic_dataset(
        num_classes=8,
        samples=32,
        resolution=cfg.dataset.resolution,
        channels=3,
    )
    assert data.path.exists()
    assert data.num_classes == 8
    assert data.is_rgb is True
