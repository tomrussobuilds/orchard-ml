"""
Unit tests for Dataset Metadata schemas.

Tests DatasetMetadata base model and DatasetRegistryWrapper
for validation, property methods, and error handling.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from orchard.core.metadata import DatasetMetadata, DatasetRegistryWrapper


# DATASET METADATA TESTS
@pytest.mark.unit
def test_dataset_metadata_repr_all_components():
    """Test DatasetMetadata __repr__ includes all components."""
    metadata = DatasetMetadata(
        name="testmnist",
        display_name="Test MNIST Dataset",
        md5_checksum="abc123",
        url="https://example.com/test.npz",
        path=Path("/data/test.npz"),
        classes=["A", "B"],
        in_channels=1,
        native_resolution=224,
        mean=(0.5,),
        std=(0.2,),
        is_anatomical=True,
        is_texture_based=False,
    )

    result = metadata.__repr__()

    assert result.startswith("<DatasetMetadata:")
    assert "Test MNIST Dataset" in result
    assert "224x224" in result
    assert "2 classes" in result
    assert result.endswith(">")


@pytest.mark.unit
def test_registry_wrapper_get_dataset_not_found():
    """Test DatasetRegistryWrapper.get_dataset raises KeyError for unknown dataset."""
    wrapper = DatasetRegistryWrapper(resolution=28)

    with pytest.raises(KeyError) as exc_info:
        wrapper.get_dataset("nonexistent_dataset")

    error_msg = str(exc_info.value)
    assert "nonexistent_dataset" in error_msg
    assert "not found" in error_msg
    assert "Available:" in error_msg


@pytest.mark.unit
def test_registry_wrapper_invalid_resolution():
    """Test DatasetRegistryWrapper raises ValueError for invalid resolution."""
    with pytest.raises(ValueError) as exc_info:
        DatasetRegistryWrapper(resolution=999)

    error_msg = str(exc_info.value)
    assert "Unsupported resolution 999" in error_msg
    assert "[28, 32, 64, 128, 224]" in error_msg


@pytest.mark.unit
def test_registry_wrapper_resolution_28():
    """Test DatasetRegistryWrapper loads 28x28 registry correctly."""
    wrapper = DatasetRegistryWrapper(resolution=28)

    assert wrapper.resolution == 28
    assert len(wrapper.registry) > 0

    for metadata in wrapper.registry.values():
        assert metadata.native_resolution == 28


@pytest.mark.unit
def test_registry_wrapper_resolution_32():
    """Test DatasetRegistryWrapper loads 32x32 registry correctly."""
    wrapper = DatasetRegistryWrapper(resolution=32)

    assert wrapper.resolution == 32
    assert len(wrapper.registry) > 0

    for metadata in wrapper.registry.values():
        assert metadata.native_resolution == 32


@pytest.mark.unit
def test_registry_wrapper_resolution_32_contains_cifar():
    """Test 32x32 registry contains CIFAR-10 and CIFAR-100."""
    wrapper = DatasetRegistryWrapper(resolution=32)

    assert "cifar10" in wrapper.registry
    assert "cifar100" in wrapper.registry
    assert wrapper.registry["cifar10"].in_channels == 3
    assert wrapper.registry["cifar100"].in_channels == 3
    assert len(wrapper.registry["cifar10"].classes) == 10
    assert len(wrapper.registry["cifar100"].classes) == 100


@pytest.mark.unit
def test_registry_wrapper_resolution_64():
    """Test DatasetRegistryWrapper loads 64x64 registry correctly."""
    wrapper = DatasetRegistryWrapper(resolution=64)

    assert wrapper.resolution == 64
    assert len(wrapper.registry) > 0

    for metadata in wrapper.registry.values():
        assert metadata.native_resolution == 64


@pytest.mark.unit
def test_registry_wrapper_resolution_128():
    """Test DatasetRegistryWrapper loads 128x128 registry correctly."""
    wrapper = DatasetRegistryWrapper(resolution=128)

    assert wrapper.resolution == 128
    assert len(wrapper.registry) == 11

    for metadata in wrapper.registry.values():
        assert metadata.native_resolution == 128


@pytest.mark.unit
def test_registry_wrapper_resolution_224():
    """Test DatasetRegistryWrapper loads 224x224 registry correctly."""
    wrapper = DatasetRegistryWrapper(resolution=224)

    assert wrapper.resolution == 224
    assert len(wrapper.registry) > 0

    for metadata in wrapper.registry.values():
        assert metadata.native_resolution == 224


@pytest.mark.unit
def test_registry_wrapper_get_dataset_returns_deep_copy():
    """Test get_dataset returns independent copy of metadata."""
    wrapper = DatasetRegistryWrapper(resolution=28)

    available_datasets = list(wrapper.registry.keys())
    assert len(available_datasets) > 0

    dataset_name = available_datasets[0]

    meta1 = wrapper.get_dataset(dataset_name)
    meta2 = wrapper.get_dataset(dataset_name)

    assert meta1 == meta2
    assert meta1 is not meta2


@pytest.mark.unit
def test_dataset_metadata_normalization_info_property():
    """Test DatasetMetadata.normalization_info property (line 59 in base.py)."""
    metadata = DatasetMetadata(
        name="testmnist",
        display_name="Test Dataset",
        md5_checksum="abc123",
        url="https://example.com/test.npz",
        path=Path("/data/test.npz"),
        classes=["A", "B"],
        in_channels=3,
        native_resolution=28,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        is_anatomical=True,
        is_texture_based=False,
    )
    result = metadata.normalization_info

    assert "Mean: (0.485, 0.456, 0.406)" in result
    assert "Std: (0.229, 0.224, 0.225)" in result
    assert "|" in result


@pytest.mark.unit
def test_dataset_metadata_resolution_str_none():
    """Test resolution_str returns 'unknown' when native_resolution is None."""
    metadata = DatasetMetadata(
        name="testmnist",
        display_name="Test Dataset",
        md5_checksum="abc123",
        url="https://example.com/test.npz",
        path=Path("/data/test.npz"),
        classes=["A", "B"],
        in_channels=1,
        native_resolution=None,
        mean=(0.5,),
        std=(0.2,),
        is_anatomical=False,
        is_texture_based=False,
    )
    assert metadata.resolution_str == "unknown"


@pytest.mark.unit
def test_registry_wrapper_empty_source_registry():
    """Test DatasetRegistryWrapper raises ValueError when source registry is empty."""
    empty_table = {28: ({},), 32: ({},), 64: ({},), 128: ({},), 224: ({}, {})}
    with patch("orchard.core.metadata.wrapper._RESOLUTION_REGISTRIES", empty_table):
        with pytest.raises(ValueError) as exc_info:
            DatasetRegistryWrapper(resolution=28)

        error_msg = str(exc_info.value)
        assert "Dataset registry for resolution 28 is empty" in error_msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
