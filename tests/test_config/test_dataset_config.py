"""
Test Suite for DatasetConfig.

Tests dataset configuration validation, metadata injection,
force_rgb logic, and resolution handling.
"""

# Standard Imports
import argparse

# Third-Party Imports
import pytest
from pydantic import ValidationError

# Internal Imports
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

    # img_size should default to resolution
    assert config.img_size == 224


@pytest.mark.unit
def test_img_size_explicit_override():
    """Test that explicit img_size overrides resolution."""
    config = DatasetConfig(resolution=28, img_size=64)

    # Explicit img_size takes precedence
    assert config.img_size == 64
    assert config.resolution == 28


# UNIT TESTS: FORCE_RGB LOGIC
def test_force_rgb_disabled_keeps_grayscale(mock_grayscale_metadata):
    """Test force_rgb=False preserves grayscale channels."""
    config = DatasetConfig(metadata=mock_grayscale_metadata, force_rgb=False)

    # Verify grayscale is preserved (single-channel)
    assert len(config.mean) == 1, "Mean should be 1-tuple for grayscale"
    assert len(config.std) == 1, "Std should be 1-tuple for grayscale"

    # Verify the values match what's in the mock metadata
    assert config.mean == mock_grayscale_metadata.mean
    assert config.std == mock_grayscale_metadata.std

    assert config.effective_in_channels == 1
    assert config.processing_mode == "NATIVE-GRAY"


def test_force_rgb_grayscale_to_rgb(mock_grayscale_metadata):
    """Test force_rgb converts grayscale mean/std to RGB."""
    config = DatasetConfig(metadata=mock_grayscale_metadata, force_rgb=True)

    # Verify mean/std were expanded from 1 channel to 3 channels
    assert len(config.mean) == 3, "Mean should be 3-tuple for RGB"
    assert len(config.std) == 3, "Std should be 3-tuple for RGB"

    # Verify all channels have the same value (replicated from grayscale)
    assert (
        config.mean[0] == config.mean[1] == config.mean[2]
    ), "All RGB channels should have same mean (replicated from grayscale)"
    assert (
        config.std[0] == config.std[1] == config.std[2]
    ), "All RGB channels should have same std (replicated from grayscale)"

    # Verify behavior
    assert config.effective_in_channels == 3
    assert config.processing_mode == "RGB-PROMOTED"


def test_force_rgb_native_rgb_noeffect(mock_metadata_28):
    """Test force_rgb has no effect on native RGB datasets."""
    config = DatasetConfig(metadata=mock_metadata_28, force_rgb=True)

    # Native RGB should remain unchanged
    assert len(config.mean) == 3, "Mean should be 3-tuple for RGB"
    assert len(config.std) == 3, "Std should be 3-tuple for RGB"

    # Verify values match the mock metadata (no transformation)
    assert config.mean == mock_metadata_28.mean
    assert config.std == mock_metadata_28.std

    assert config.effective_in_channels == 3
    assert config.processing_mode == "NATIVE-RGB"


# UNIT TESTS: PROPERTIES
@pytest.mark.unit
def test_ensure_metadata_lazy_loading():
    """Test metadata lazy loading via _ensure_metadata."""
    # Create without metadata
    config = DatasetConfig(name="bloodmnist", resolution=28)

    # Metadata should be None initially
    assert config.metadata is None

    # Accessing property should trigger loading
    config.dataset_name

    # Metadata should now be populated
    assert config.metadata is not None
    assert config.metadata.name == "bloodmnist"


@pytest.mark.unit
def test_processing_mode_classification(mock_grayscale_metadata, mock_metadata_28):
    """Test processing_mode property returns correct classification."""
    # Grayscale + force_rgb
    config_promoted = DatasetConfig(metadata=mock_grayscale_metadata, force_rgb=True)
    assert config_promoted.processing_mode == "RGB-PROMOTED"

    # Grayscale native
    config_gray = DatasetConfig(metadata=mock_grayscale_metadata, force_rgb=False)
    assert config_gray.processing_mode == "NATIVE-GRAY"

    # Native RGB
    config_rgb = DatasetConfig(metadata=mock_metadata_28, force_rgb=True)
    assert config_rgb.processing_mode == "NATIVE-RGB"


# UNIT TESTS: FROM_ARGS FACTORY
@pytest.mark.unit
def test_from_args_basic(basic_args):
    """Test DatasetConfig.from_args() with basic arguments."""
    config = DatasetConfig.from_args(basic_args)

    assert config.dataset_name == "bloodmnist"
    assert config.resolution == 28
    assert config.use_weighted_sampler is True


@pytest.mark.unit
def test_from_args_force_rgb_auto_enable_for_grayscale_pretrained():
    """Test force_rgb auto-enables for grayscale + pretrained."""
    args = argparse.Namespace(
        dataset="pneumoniamnist", resolution=28, pretrained=True, force_rgb=None
    )

    config = DatasetConfig.from_args(args)

    assert config.force_rgb is True


@pytest.mark.unit
def test_from_args_force_rgb_explicit_override():
    """Test explicit force_rgb CLI argument overrides auto-logic."""
    args = argparse.Namespace(
        dataset="pneumoniamnist",
        resolution=28,
        pretrained=True,
        force_rgb=False,  # Explicitly disabled
    )

    config = DatasetConfig.from_args(args)

    # Explicit CLI value should override
    assert config.force_rgb is False


@pytest.mark.unit
def test_from_args_max_samples_zero_becomes_none():
    """Test max_samples=0 converts to None."""
    args = argparse.Namespace(dataset="bloodmnist", resolution=28, max_samples=0)

    config = DatasetConfig.from_args(args)

    # Zero should convert to None (unlimited)
    assert config.max_samples is None


@pytest.mark.unit
def test_from_args_max_samples_positive():
    """Test max_samples with positive value."""
    args = argparse.Namespace(dataset="bloodmnist", resolution=28, max_samples=1000)

    config = DatasetConfig.from_args(args)

    assert config.max_samples == 1000


# INTEGRATION TESTS: VALIDATION
@pytest.mark.integration
def test_dataset_not_found_raises_error():
    """Test error raised for non-existent dataset."""
    args = argparse.Namespace(dataset="nonexistent_dataset", resolution=28, pretrained=True)

    with pytest.raises(KeyError, match="not found"):
        DatasetConfig.from_args(args)


@pytest.mark.integration
def test_resolution_224_loads_correct_metadata():
    """Test resolution=224 loads high-res metadata."""
    args = argparse.Namespace(dataset="organcmnist", resolution=224, pretrained=True)

    config = DatasetConfig.from_args(args)

    assert config.resolution == 224
    assert config.metadata.native_resolution == 224


#                         EDGE CASES & REGRESSION TESTS                       #
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
        DatasetConfig(resolution=0)  # Too small

    with pytest.raises(ValidationError):
        DatasetConfig(resolution=2000)  # Too large


@pytest.mark.unit
def test_sync_validator_runs_before_frozen():
    """Test sync_img_size_with_resolution runs during construction."""
    # This should NOT raise even though we're modifying values
    # because validator runs BEFORE frozen=True takes effect
    config = DatasetConfig(resolution=224)

    assert config.img_size == 224  # Auto-synced
