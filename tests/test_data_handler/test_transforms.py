"""
Pytest test suite for data transformation pipelines.

Tests augmentation description generation and torchvision v2
training/validation pipelines for both RGB and Grayscale datasets.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torchvision.transforms import v2

# Internal Import
from orchard.data_handler import (
    get_augmentations_description,
    get_pipeline_transforms,
)


# FIXTURES
@pytest.fixture
def aug_cfg():
    """Minimal augmentation config stub."""
    return SimpleNamespace(
        hflip=0.5,
        rotation_angle=15,
        jitter_val=0.2,
        min_scale=0.8,
    )


@pytest.fixture
def img_size():
    return (224, 224)


@pytest.fixture
def mixup_alpha():
    return 0.4


@pytest.fixture
def rgb_metadata():
    """DatasetMetadata stub for RGB datasets."""
    return SimpleNamespace(
        in_channels=3,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        is_anatomical=False,
        is_texture_based=False,
    )


@pytest.fixture
def grayscale_metadata():
    """DatasetMetadata stub for Grayscale datasets."""
    return SimpleNamespace(
        in_channels=1,
        mean=[0.5],
        std=[0.25],
        is_anatomical=False,
        is_texture_based=False,
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
def test_get_augmentations_description_contains_all_fields(aug_cfg, img_size, mixup_alpha):
    """Augmentation description should include all configured operations."""
    descr = get_augmentations_description(aug_cfg, img_size, mixup_alpha)

    assert "HFlip" in descr
    assert "Rotation" in descr
    assert "Jitter" in descr
    assert "ResizedCrop" in descr
    assert "MixUp" in descr
    assert "α=0.4" in descr


def test_get_augmentations_description_without_mixup(aug_cfg, img_size):
    """MixUp should be omitted when alpha <= 0."""
    descr = get_augmentations_description(aug_cfg, img_size, 0.0)

    assert "MixUp" not in descr


# TEST: PIPELINE CONSTRUCTION
def test_pipeline_returns_compose_objects(aug_cfg, img_size, rgb_metadata):
    """Pipeline factory should return torchvision v2 Compose objects."""
    train_tf, val_tf = get_pipeline_transforms(aug_cfg, img_size, rgb_metadata)

    assert isinstance(train_tf, v2.Compose)
    assert isinstance(val_tf, v2.Compose)


def test_rgb_pipeline_does_not_include_grayscale(aug_cfg, img_size, rgb_metadata):
    """RGB datasets should not include Grayscale promotion."""
    train_tf, val_tf = get_pipeline_transforms(aug_cfg, img_size, rgb_metadata)

    train_types = [type(t) for t in train_tf.transforms]
    val_types = [type(t) for t in val_tf.transforms]

    assert v2.Grayscale not in train_types
    assert v2.Grayscale not in val_types


def test_grayscale_pipeline_includes_grayscale_promotion(aug_cfg, img_size, grayscale_metadata):
    """Grayscale datasets must be promoted to 3 channels."""
    train_tf, val_tf = get_pipeline_transforms(aug_cfg, img_size, grayscale_metadata)

    train_types = [type(t) for t in train_tf.transforms]
    val_types = [type(t) for t in val_tf.transforms]

    assert v2.Grayscale in train_types
    assert v2.Grayscale in val_types


def test_normalization_stats_replicated_for_grayscale(aug_cfg, img_size, grayscale_metadata):
    """Grayscale mean/std should be replicated to 3 channels."""
    train_tf, _ = get_pipeline_transforms(aug_cfg, img_size, grayscale_metadata)

    normalize = next(t for t in train_tf.transforms if isinstance(t, v2.Normalize))

    assert normalize.mean == [0.5, 0.5, 0.5]
    assert normalize.std == [0.25, 0.25, 0.25]


# TEST: PIPELINE EXECUTION (SMOKE TEST)
def test_train_pipeline_executes_on_rgb_image(aug_cfg, img_size, rgb_metadata, dummy_image_rgb):
    """Training pipeline should run end-to-end on RGB input."""
    train_tf, _ = get_pipeline_transforms(aug_cfg, img_size, rgb_metadata)

    out = train_tf(dummy_image_rgb)

    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 3
    assert out.dtype == torch.float32


def test_val_pipeline_executes_on_grayscale_image(
    aug_cfg, img_size, grayscale_metadata, dummy_image_gray
):
    """Validation pipeline should run end-to-end on Grayscale input."""
    _, val_tf = get_pipeline_transforms(aug_cfg, img_size, grayscale_metadata)

    out = val_tf(dummy_image_gray)

    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 3
    assert out.dtype == torch.float32


# TEST: DOMAIN-AWARE AUGMENTATION
def test_anatomical_disables_flip_and_rotation(aug_cfg, img_size, rgb_metadata):
    """Anatomical datasets should not have RandomHorizontalFlip or RandomRotation."""
    rgb_metadata.is_anatomical = True
    train_tf, _ = get_pipeline_transforms(aug_cfg, img_size, rgb_metadata)

    train_types = [type(t) for t in train_tf.transforms]
    assert v2.RandomHorizontalFlip not in train_types
    assert v2.RandomRotation not in train_types


def test_texture_based_disables_color_jitter(aug_cfg, img_size, rgb_metadata):
    """Texture-based datasets should not have ColorJitter."""
    rgb_metadata.is_texture_based = True
    train_tf, _ = get_pipeline_transforms(aug_cfg, img_size, rgb_metadata)

    train_types = [type(t) for t in train_tf.transforms]
    assert v2.ColorJitter not in train_types


def test_standard_dataset_has_all_augmentations(aug_cfg, img_size, rgb_metadata):
    """Non-anatomical, non-texture datasets should have full augmentations."""
    train_tf, _ = get_pipeline_transforms(aug_cfg, img_size, rgb_metadata)

    train_types = [type(t) for t in train_tf.transforms]
    assert v2.RandomHorizontalFlip in train_types
    assert v2.RandomRotation in train_types
    assert v2.ColorJitter in train_types


def test_anatomical_texture_minimal_augmentation(aug_cfg, img_size, rgb_metadata):
    """Anatomical + texture datasets get minimal augmentation (crop + normalize only)."""
    rgb_metadata.is_anatomical = True
    rgb_metadata.is_texture_based = True
    train_tf, _ = get_pipeline_transforms(aug_cfg, img_size, rgb_metadata)

    train_types = [type(t) for t in train_tf.transforms]
    assert v2.RandomHorizontalFlip not in train_types
    assert v2.RandomRotation not in train_types
    assert v2.ColorJitter not in train_types
    assert v2.RandomResizedCrop in train_types
    assert v2.Normalize in train_types


def test_augmentation_description_anatomical(aug_cfg, img_size, mixup_alpha):
    """Anatomical datasets should omit HFlip and Rotation from description."""
    meta = SimpleNamespace(is_anatomical=True, is_texture_based=False)
    descr = get_augmentations_description(aug_cfg, img_size, mixup_alpha, ds_meta=meta)

    assert "HFlip" not in descr
    assert "Rotation" not in descr
    assert "Jitter" in descr


def test_augmentation_description_texture(aug_cfg, img_size, mixup_alpha):
    """Texture-based datasets should omit Jitter from description."""
    meta = SimpleNamespace(is_anatomical=False, is_texture_based=True)
    descr = get_augmentations_description(aug_cfg, img_size, mixup_alpha, ds_meta=meta)

    assert "Jitter" not in descr
    assert "HFlip" in descr
