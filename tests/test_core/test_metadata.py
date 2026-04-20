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
def test_dataset_metadata_repr_all_components() -> None:
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
def test_dataset_metadata_annotation_path_default_none() -> None:
    """annotation_path defaults to None for classification datasets."""
    metadata = DatasetMetadata(
        name="testmnist",
        display_name="Test",
        md5_checksum="abc",
        url="https://example.com/test.npz",
        path=Path("/data/test.npz"),
        classes=["A", "B"],
        in_channels=1,
        mean=(0.5,),
        std=(0.2,),
    )
    assert metadata.annotation_path is None


@pytest.mark.unit
def test_dataset_metadata_annotation_path_detection() -> None:
    """annotation_path can be set for detection datasets."""
    metadata = DatasetMetadata(
        name="pennfudan",
        display_name="PennFudan",
        md5_checksum="abc",
        url="https://example.com/pennfudan.zip",
        path=Path("/data/pennfudan_images.npz"),
        classes=["person"],
        in_channels=3,
        native_resolution=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        is_anatomical=False,
        is_texture_based=False,
        annotation_path=Path("/data/pennfudan_annotations.npz"),
    )
    assert metadata.annotation_path == Path("/data/pennfudan_annotations.npz")
    assert metadata.num_classes == 1


@pytest.mark.unit
def test_registry_wrapper_get_dataset_not_found() -> None:
    """Test DatasetRegistryWrapper.get_dataset raises KeyError for unknown dataset."""
    wrapper = DatasetRegistryWrapper(resolution=28)

    with pytest.raises(KeyError) as exc_info:
        wrapper.get_dataset("nonexistent_dataset")

    error_msg = str(exc_info.value)
    assert "nonexistent_dataset" in error_msg
    assert "not found" in error_msg
    assert "Available:" in error_msg
    # Kill mutant: available = None → message must contain a real list, not "None"
    assert "[" in error_msg and "]" in error_msg


@pytest.mark.unit
def test_registry_wrapper_invalid_resolution() -> None:
    """Test DatasetRegistryWrapper raises ValueError for invalid resolution."""
    with pytest.raises(ValueError) as exc_info:
        DatasetRegistryWrapper(resolution=999)

    error_msg = str(exc_info.value)
    assert "Unsupported resolution 999" in error_msg
    assert "[28, 32, 64, 128, 224]" in error_msg


@pytest.mark.unit
def test_registry_wrapper_resolution_28() -> None:
    """Test DatasetRegistryWrapper loads 28x28 registry correctly."""
    wrapper = DatasetRegistryWrapper(resolution=28)

    assert wrapper.resolution == 28
    assert len(wrapper.registry) > 0

    for metadata in wrapper.registry.values():
        assert metadata.native_resolution == 28


@pytest.mark.unit
def test_registry_wrapper_resolution_32() -> None:
    """Test DatasetRegistryWrapper loads 32x32 registry correctly."""
    wrapper = DatasetRegistryWrapper(resolution=32)

    assert wrapper.resolution == 32
    assert len(wrapper.registry) > 0

    for metadata in wrapper.registry.values():
        assert metadata.native_resolution == 32


@pytest.mark.unit
def test_registry_wrapper_resolution_32_contains_cifar() -> None:
    """Test 32x32 registry contains CIFAR-10 and CIFAR-100."""
    wrapper = DatasetRegistryWrapper(resolution=32)

    assert "cifar10" in wrapper.registry
    assert "cifar100" in wrapper.registry
    assert wrapper.registry["cifar10"].in_channels == 3
    assert wrapper.registry["cifar100"].in_channels == 3
    assert len(wrapper.registry["cifar10"].classes) == 10
    assert len(wrapper.registry["cifar100"].classes) == 100


@pytest.mark.unit
def test_registry_wrapper_resolution_64() -> None:
    """Test DatasetRegistryWrapper loads 64x64 registry correctly."""
    wrapper = DatasetRegistryWrapper(resolution=64)

    assert wrapper.resolution == 64
    assert len(wrapper.registry) > 0

    for metadata in wrapper.registry.values():
        assert metadata.native_resolution == 64


@pytest.mark.unit
def test_registry_wrapper_resolution_128() -> None:
    """Test DatasetRegistryWrapper loads 128x128 registry correctly."""
    wrapper = DatasetRegistryWrapper(resolution=128)

    assert wrapper.resolution == 128
    assert len(wrapper.registry) == 11

    for metadata in wrapper.registry.values():
        assert metadata.native_resolution == 128


@pytest.mark.unit
def test_registry_wrapper_resolution_224() -> None:
    """Test DatasetRegistryWrapper loads 224x224 registry correctly."""
    wrapper = DatasetRegistryWrapper(resolution=224)

    assert wrapper.resolution == 224
    assert len(wrapper.registry) > 0

    for metadata in wrapper.registry.values():
        assert metadata.native_resolution == 224


@pytest.mark.unit
def test_registry_wrapper_get_dataset_returns_deep_copy() -> None:
    """Test get_dataset returns independent copy of metadata."""
    wrapper = DatasetRegistryWrapper(resolution=28)

    available_datasets = list(wrapper.registry.keys())
    assert len(available_datasets) > 0

    dataset_name = available_datasets[0]

    meta1 = wrapper.get_dataset(dataset_name)
    meta2 = wrapper.get_dataset(dataset_name)

    assert meta1 == meta2
    assert meta1 is not meta2
    # Kill mutant: copy.copy → copy.deepcopy — nested mutable fields must be independent
    assert meta1.classes is not meta2.classes


@pytest.mark.unit
def test_dataset_metadata_normalization_info_property() -> None:
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
def test_dataset_metadata_resolution_str_none() -> None:
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
def test_registry_wrapper_empty_source_registry() -> None:
    """Test DatasetRegistryWrapper raises ValueError when source registry is empty."""
    empty_table: dict[int, tuple[dict[str, DatasetMetadata], ...]] = {
        28: ({},),
        32: ({},),
        64: ({},),
        128: ({},),
        224: ({}, {}),
    }
    with patch("orchard.core.metadata.wrapper._CLASSIFICATION_REGISTRIES", empty_table):
        with pytest.raises(ValueError) as exc_info:
            DatasetRegistryWrapper(resolution=28)

        error_msg = str(exc_info.value)
        assert "No datasets available at resolution 28" in error_msg


@pytest.mark.unit
def test_detection_registry_wrapper_unsupported_resolution_raises() -> None:
    """DetectionRegistryWrapper raises ValueError for resolution without detection datasets."""
    from orchard.core.metadata.wrapper import DetectionRegistryWrapper

    with pytest.raises(ValueError, match="No datasets available"):
        DetectionRegistryWrapper(resolution=28)


@pytest.mark.unit
def test_detection_registry_wrapper_has_pennfudan() -> None:
    """DetectionRegistryWrapper at 224 contains pennfudan."""
    from orchard.core.metadata.wrapper import DetectionRegistryWrapper

    wrapper = DetectionRegistryWrapper(resolution=224)
    meta = wrapper.get_dataset("pennfudan")
    assert meta.name == "pennfudan"
    assert meta.classes == ["person"]
    assert meta.annotation_path is not None


@pytest.mark.unit
def test_get_registry_detection_route() -> None:
    """get_registry with task_type='detection' returns DetectionRegistryWrapper."""
    from orchard.core.metadata.wrapper import (
        _DETECTION_REGISTRIES,
        DetectionRegistryWrapper,
        get_registry,
    )

    # Temporarily populate detection registry so construction succeeds
    fake_meta = DatasetMetadata(
        name="fake_det",
        display_name="Fake",
        md5_checksum="",
        url="",
        path=Path("/tmp/fake.npz"),
        classes=["obj"],
        in_channels=3,
        mean=(0.5, 0.5, 0.5),
        std=(0.2, 0.2, 0.2),
        native_resolution=224,
    )
    original_224 = _DETECTION_REGISTRIES.get(224)
    _DETECTION_REGISTRIES[224] = ({"fake_det": fake_meta},)
    try:
        wrapper = get_registry(224, "detection")
        assert isinstance(wrapper, DetectionRegistryWrapper)
        assert "fake_det" in wrapper.registry
    finally:
        if original_224 is not None:
            _DETECTION_REGISTRIES[224] = original_224
        else:
            del _DETECTION_REGISTRIES[224]


@pytest.mark.unit
def test_get_registry_default_task_type_is_classification() -> None:
    """get_registry without task_type defaults to classification."""
    from orchard.core.metadata.wrapper import ClassificationRegistryWrapper, get_registry

    wrapper = get_registry(28)
    assert isinstance(wrapper, ClassificationRegistryWrapper)


@pytest.mark.unit
def test_get_registry_unknown_task_type_raises() -> None:
    """get_registry raises ValueError for unknown task_type strings."""
    from orchard.core.metadata.wrapper import get_registry

    with pytest.raises(ValueError, match="Unknown task_type"):
        get_registry(28, "segmentation")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
