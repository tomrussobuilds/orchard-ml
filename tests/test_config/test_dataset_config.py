"""
Test Suite for DatasetConfig.

Tests dataset configuration validation, metadata injection,
force_rgb logic, and resolution handling.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orchard.core.config import DatasetConfig


# UNIT TESTS: CONSTRUCTION
@pytest.mark.unit
def test_dataset_config_defaults():
    """Test DatasetConfig with default values."""
    config = DatasetConfig()

    assert config.name == "bloodmnist"
    assert config.use_weighted_sampler is True
    assert config.max_samples is None
    assert config.force_rgb is True
    assert config.resolution == 28


@pytest.mark.unit
def test_dataset_config_with_metadata(mock_metadata_28):
    """Test DatasetConfig with explicit metadata."""
    config = DatasetConfig(name="bloodmnist", metadata=mock_metadata_28, resolution=28)

    assert config.dataset_name == "bloodmnist"
    assert config.num_classes == 8
    assert config.in_channels == 3


@pytest.mark.unit
def test_img_size_auto_sync_from_resolution():
    """Test that img_size auto-syncs with resolution when not provided."""
    config = DatasetConfig(resolution=224)

    assert config.img_size == 224


@pytest.mark.unit
def test_img_size_explicit_override():
    """Test that explicit img_size overrides resolution."""
    config = DatasetConfig(resolution=28, img_size=64)

    assert config.img_size == 64
    assert config.resolution == 28


# UNIT TESTS: FORCE_RGB LOGIC
def test_force_rgb_disabled_keeps_grayscale(mock_grayscale_metadata):
    """Test force_rgb=False preserves grayscale channels."""
    config = DatasetConfig(metadata=mock_grayscale_metadata, force_rgb=False)

    assert len(config.mean) == 1, "Mean should be 1-tuple for grayscale"
    assert len(config.std) == 1, "Std should be 1-tuple for grayscale"

    assert config.mean == mock_grayscale_metadata.mean
    assert config.std == mock_grayscale_metadata.std

    assert config.effective_in_channels == 1
    assert config.processing_mode == "NATIVE-GRAY"


def test_force_rgb_grayscale_to_rgb(mock_grayscale_metadata):
    """Test force_rgb converts grayscale mean/std to RGB."""
    config = DatasetConfig(metadata=mock_grayscale_metadata, force_rgb=True)

    assert len(config.mean) == 3, "Mean should be 3-tuple for RGB"
    assert len(config.std) == 3, "Std should be 3-tuple for RGB"

    assert (
        config.mean[0] == config.mean[1] == config.mean[2]
    ), "All RGB channels should have same mean (replicated from grayscale)"
    assert (
        config.std[0] == config.std[1] == config.std[2]
    ), "All RGB channels should have same std (replicated from grayscale)"

    assert config.effective_in_channels == 3
    assert config.processing_mode == "RGB-PROMOTED"


def test_force_rgb_native_rgb_noeffect(mock_metadata_28):
    """Test force_rgb has no effect on native RGB datasets."""
    config = DatasetConfig(metadata=mock_metadata_28, force_rgb=True)

    assert len(config.mean) == 3, "Mean should be 3-tuple for RGB"
    assert len(config.std) == 3, "Std should be 3-tuple for RGB"

    assert config.mean == mock_metadata_28.mean
    assert config.std == mock_metadata_28.std

    assert config.effective_in_channels == 3
    assert config.processing_mode == "NATIVE-RGB"


# UNIT TESTS: PROPERTIES
@pytest.mark.unit
def test_ensure_metadata_lazy_loading():
    """Test metadata lazy loading via _ensure_metadata."""
    config = DatasetConfig(name="bloodmnist", resolution=28)

    assert config.metadata is None

    config.dataset_name

    assert config.metadata is not None
    assert config.metadata.name == "bloodmnist"


@pytest.mark.unit
def test_ensure_metadata_rewrites_path_for_custom_data_root(tmp_path):
    """_ensure_metadata rewrites metadata path when data_root differs from DATASET_DIR."""
    config = DatasetConfig(name="bloodmnist", resolution=28, data_root=tmp_path)

    meta = config._ensure_metadata

    assert meta.path.parent == tmp_path
    assert meta.path.name == "bloodmnist_28.npz"


@pytest.mark.unit
def test_processing_mode_classification(mock_grayscale_metadata, mock_metadata_28):
    """Test processing_mode property returns correct classification."""
    config_promoted = DatasetConfig(metadata=mock_grayscale_metadata, force_rgb=True)
    assert config_promoted.processing_mode == "RGB-PROMOTED"

    config_gray = DatasetConfig(metadata=mock_grayscale_metadata, force_rgb=False)
    assert config_gray.processing_mode == "NATIVE-GRAY"

    config_rgb = DatasetConfig(metadata=mock_metadata_28, force_rgb=True)
    assert config_rgb.processing_mode == "NATIVE-RGB"


@pytest.mark.unit
def test_effective_is_anatomical_with_metadata(mock_metadata_28):
    """Test effective_is_anatomical returns metadata flag when present."""
    config = DatasetConfig(metadata=mock_metadata_28)
    assert config.effective_is_anatomical == mock_metadata_28.is_anatomical


@pytest.mark.unit
def test_effective_is_anatomical_without_metadata():
    """Test effective_is_anatomical lazy-loads from registry when metadata is None."""
    config = DatasetConfig()  # default dataset = bloodmnist (is_anatomical=False)
    assert config.effective_is_anatomical is False


@pytest.mark.unit
def test_effective_is_texture_based_with_metadata(mock_metadata_28):
    """Test effective_is_texture_based returns metadata flag when present."""
    config = DatasetConfig(metadata=mock_metadata_28)
    assert config.effective_is_texture_based == mock_metadata_28.is_texture_based


@pytest.mark.unit
def test_effective_is_texture_based_without_metadata():
    """Test effective_is_texture_based lazy-loads from registry when metadata is None."""
    config = DatasetConfig()  # default dataset = bloodmnist (is_texture_based=False)
    assert config.effective_is_texture_based is False


# EDGE CASES & REGRESSION TESTS
@pytest.mark.unit
def test_frozen_immutability():
    """Test DatasetConfig is frozen (immutable)."""
    config = DatasetConfig()

    with pytest.raises(ValidationError):
        config.name = "different_name"


@pytest.mark.unit
def test_invalid_resolution_rejected():
    """Test invalid resolution values are rejected."""
    with pytest.raises(ValidationError):
        DatasetConfig(resolution=0)

    with pytest.raises(ValidationError):
        DatasetConfig(resolution=2000)


@pytest.mark.unit
def test_sync_validator_runs_before_frozen():
    """Test sync_img_size_with_resolution runs during construction."""
    config = DatasetConfig(resolution=224)

    assert config.img_size == 224


# UNIT TESTS: MAX_SAMPLES VALIDATION
@pytest.mark.unit
def test_max_samples_none_allowed():
    """Test max_samples=None (load all) is valid."""
    config = DatasetConfig(max_samples=None)

    assert config.max_samples is None


@pytest.mark.unit
def test_max_samples_valid_value():
    """Test max_samples with valid value >= 20."""
    config = DatasetConfig(max_samples=100)

    assert config.max_samples == 100


@pytest.mark.unit
def test_max_samples_minimum_boundary():
    """Test max_samples=20 is accepted (boundary)."""
    config = DatasetConfig(max_samples=20)

    assert config.max_samples == 20


@pytest.mark.unit
def test_max_samples_too_small_rejected():
    """Test max_samples < 20 is rejected."""
    with pytest.raises(ValidationError, match="max_samples=19 is too small"):
        DatasetConfig(max_samples=19)


@pytest.mark.unit
def test_max_samples_one_rejected():
    """Test max_samples=1 is rejected."""
    with pytest.raises(ValidationError, match="max_samples=1 is too small"):
        DatasetConfig(max_samples=1)
