"""
Unit tests for Dataset Metadata schemas.

Tests DatasetMetadata base model and DatasetRegistryWrapper
for validation, property methods, and error handling.
"""

# Standard Imports
from pathlib import Path

# Third-Party Imports
import pytest

# Internal Imports
from orchard.core.metadata import DatasetMetadata, DatasetRegistryWrapper


# DATASET METADATA TESTS
@pytest.mark.unit
def test_dataset_metadata_repr_all_components():
    """Test DatasetMetadata __repr__ includes all components."""
    metadata = DatasetMetadata(
        name="testmnist",
        display_name="Test MNIST Dataset",
        md5_checksum="abc123",
        url="http://example.com/test.npz",
        path=Path("/data/test.npz"),
        classes=["A", "B"],
        in_channels=1,
        native_resolution=224,
        mean=(0.5,),
        std=(0.2,),
        is_anatomical=True,
        is_texture_based=False,
    )

    # Call repr explicitly
    result = metadata.__repr__()

    # Verify all parts of the formatted string
    assert result.startswith("<DatasetMetadata:")
    assert "Test MNIST Dataset" in result
    assert "224x224" in result
    assert "2 classes" in result
    assert result.endswith(">")


# DATASET REGISTRY WRAPPER TESTS
@pytest.mark.unit
def test_registry_wrapper_get_dataset_not_found():
    """Test DatasetRegistryWrapper.get_dataset raises KeyError for unknown dataset."""
    wrapper = DatasetRegistryWrapper(resolution=28)

    with pytest.raises(KeyError) as exc_info:
        wrapper.get_dataset("nonexistent_dataset")

    # Verify error message contains available datasets
    error_msg = str(exc_info.value)
    assert "nonexistent_dataset" in error_msg
    assert "not found" in error_msg
    assert "Available:" in error_msg


@pytest.mark.unit
def test_registry_wrapper_invalid_resolution():
    """Test DatasetRegistryWrapper raises ValueError for invalid resolution."""
    with pytest.raises(ValueError) as exc_info:
        DatasetRegistryWrapper(resolution=64)

    # Verify error message
    error_msg = str(exc_info.value)
    assert "Unsupported resolution 64" in error_msg
    assert "[28, 224]" in error_msg


@pytest.mark.unit
def test_registry_wrapper_resolution_28():
    """Test DatasetRegistryWrapper loads 28x28 registry correctly."""
    wrapper = DatasetRegistryWrapper(resolution=28)

    assert wrapper.resolution == 28
    assert len(wrapper.registry) > 0

    # Verify all metadata have resolution 28
    for metadata in wrapper.registry.values():
        assert metadata.native_resolution == 28


@pytest.mark.unit
def test_registry_wrapper_resolution_224():
    """Test DatasetRegistryWrapper loads 224x224 registry correctly."""
    wrapper = DatasetRegistryWrapper(resolution=224)

    assert wrapper.resolution == 224
    assert len(wrapper.registry) > 0

    # Verify all metadata have resolution 224
    for metadata in wrapper.registry.values():
        assert metadata.native_resolution == 224


@pytest.mark.unit
def test_registry_wrapper_get_dataset_returns_deep_copy():
    """Test get_dataset returns independent copy of metadata."""
    wrapper = DatasetRegistryWrapper(resolution=28)

    # Get first available dataset
    available_datasets = list(wrapper.registry.keys())
    assert len(available_datasets) > 0

    dataset_name = available_datasets[0]

    # Get metadata twice
    meta1 = wrapper.get_dataset(dataset_name)
    meta2 = wrapper.get_dataset(dataset_name)

    # Should be equal but not the same object
    assert meta1 == meta2
    assert meta1 is not meta2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
