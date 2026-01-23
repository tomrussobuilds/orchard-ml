"""
Pytest test suite for data transformation pipelines.

Tests augmentation description generation and torchvision v2
training/validation pipelines for both RGB and Grayscale datasets.
"""

# Standard Imports
from types import SimpleNamespace

# Third-Party Imports
import pytest
import torch
from torchvision.transforms import v2

# MODULE UNDER TEST
from orchard.data_handler import (
    get_augmentations_description,
    get_pipeline_transforms,
)


# FIXTURES
@pytest.fixture
def base_cfg():
    """Minimal configuration stub matching the required Config interface."""
    return SimpleNamespace(
        augmentation=SimpleNamespace(
            hflip=0.5,
            rotation_angle=15,
            jitter_val=0.2,
            min_scale=0.8,
        ),
        dataset=SimpleNamespace(
            img_size=(224, 224),
        ),
        training=SimpleNamespace(
            mixup_alpha=0.4,
        ),
    )


@pytest.fixture
def rgb_metadata():
    """DatasetMetadata stub for RGB datasets."""
    return SimpleNamespace(
        in_channels=3,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )


@pytest.fixture
def grayscale_metadata():
    """DatasetMetadata stub for Grayscale datasets."""
    return SimpleNamespace(
        in_channels=1,
        mean=[0.5],
        std=[0.25],
    )


@pytest.fixture
def dummy_image_rgb():
    """Dummy RGB image tensor (C, H, W)."""
    return torch.randint(0, 255, (3, 256, 256), dtype=torch.uint8)


@pytest.fixture
def dummy_image_gray():
    """Dummy Grayscale image tensor (H, W)."""
    return torch.randint(0, 255, (256, 256), dtype=torch.uint8)


# TEST: AUGMENTATION DESCRIPTION
def test_get_augmentations_description_contains_all_fields(base_cfg):
    """Augmentation description should include all configured operations."""
    descr = get_augmentations_description(base_cfg)

    assert "HFlip" in descr
    assert "Rotation" in descr
    assert "Jitter" in descr
    assert "ResizedCrop" in descr
    assert "MixUp" in descr
    assert "α=0.4" in descr


def test_get_augmentations_description_without_mixup(base_cfg):
    """MixUp should be omitted when alpha <= 0."""
    base_cfg.training.mixup_alpha = 0.0

    descr = get_augmentations_description(base_cfg)

    assert "MixUp" not in descr


# TEST: PIPELINE CONSTRUCTION
def test_pipeline_returns_compose_objects(base_cfg, rgb_metadata):
    """Pipeline factory should return torchvision v2 Compose objects."""
    train_tf, val_tf = get_pipeline_transforms(base_cfg, rgb_metadata)

    assert isinstance(train_tf, v2.Compose)
    assert isinstance(val_tf, v2.Compose)


def test_rgb_pipeline_does_not_include_grayscale(base_cfg, rgb_metadata):
    """RGB datasets should not include Grayscale promotion."""
    train_tf, val_tf = get_pipeline_transforms(base_cfg, rgb_metadata)

    train_types = [type(t) for t in train_tf.transforms]
    val_types = [type(t) for t in val_tf.transforms]

    assert v2.Grayscale not in train_types
    assert v2.Grayscale not in val_types


def test_grayscale_pipeline_includes_grayscale_promotion(base_cfg, grayscale_metadata):
    """Grayscale datasets must be promoted to 3 channels."""
    train_tf, val_tf = get_pipeline_transforms(base_cfg, grayscale_metadata)

    train_types = [type(t) for t in train_tf.transforms]
    val_types = [type(t) for t in val_tf.transforms]

    assert v2.Grayscale in train_types
    assert v2.Grayscale in val_types


def test_normalization_stats_replicated_for_grayscale(base_cfg, grayscale_metadata):
    """Grayscale mean/std should be replicated to 3 channels."""
    train_tf, _ = get_pipeline_transforms(base_cfg, grayscale_metadata)

    normalize = next(t for t in train_tf.transforms if isinstance(t, v2.Normalize))

    assert normalize.mean == [0.5, 0.5, 0.5]
    assert normalize.std == [0.25, 0.25, 0.25]


#                  TEST: PIPELINE EXECUTION (SMOKE TEST)                       #
def test_train_pipeline_executes_on_rgb_image(base_cfg, rgb_metadata, dummy_image_rgb):
    """Training pipeline should run end-to-end on RGB input."""
    train_tf, _ = get_pipeline_transforms(base_cfg, rgb_metadata)

    out = train_tf(dummy_image_rgb)

    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 3
    assert out.dtype == torch.float32


def test_val_pipeline_executes_on_grayscale_image(base_cfg, grayscale_metadata, dummy_image_gray):
    """Validation pipeline should run end-to-end on Grayscale input."""
    _, val_tf = get_pipeline_transforms(base_cfg, grayscale_metadata)

    out = val_tf(dummy_image_gray)

    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 3
    assert out.dtype == torch.float32
