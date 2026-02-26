"""
Dataset Metadata Base Definitions.

Defines dataset metadata schema using Pydantic for immutability, type safety,
and seamless integration with the global configuration engine.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


# METADATA SCHEMA
class DatasetMetadata(BaseModel):
    """
    Immutable metadata container for a dataset entry.

    Ensures dataset-specific constants are grouped and frozen throughout
    pipeline execution. Serves as static definition feeding into dynamic
    DatasetConfig.

    Attributes:
        name: Short identifier (e.g., ``'pathmnist'``, ``'galaxy10'``).
        display_name: Human-readable name for reporting.
        md5_checksum: MD5 hash for download integrity verification.
        url: Source URL for dataset download.
        path: Local path to the ``.npz`` archive.
        classes: Class labels in index order.
        in_channels: Number of image channels (1=grayscale, 3=RGB).
        native_resolution: Native pixel resolution (e.g., 28, 224).
        mean: Channel-wise normalization mean.
        std: Channel-wise normalization standard deviation.
        is_anatomical: Whether images have fixed anatomical orientation.
        is_texture_based: Whether classification relies on texture patterns.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    # Identity
    name: str = Field(..., description="Short identifier (e.g., 'pathmnist')")
    display_name: str = Field(..., description="Full name for reporting")

    # Source
    md5_checksum: str = Field(..., description="MD5 hash for integrity")
    url: str = Field(..., description="Source URL for downloads")
    path: Path = Field(..., description="Path to .npz archive")

    # Classification
    classes: list[str] = Field(..., description="Class labels in index order")

    # Image properties
    in_channels: int = Field(..., description="1 for grayscale, 3 for RGB")
    native_resolution: int | None = Field(
        default=None, description="Native pixel resolution (28, 32, 64, 128, or 224)"
    )

    # Normalization
    mean: tuple[float, ...] = Field(..., description="Channel-wise mean")
    std: tuple[float, ...] = Field(..., description="Channel-wise std")

    # Behavioral flags
    is_anatomical: bool = Field(
        default=True, description="Fixed anatomical orientation (e.g., ChestMNIST)"
    )
    is_texture_based: bool = Field(
        default=True, description="Classification relies on texture (e.g., PathMNIST)"
    )

    @property
    def normalization_info(self) -> str:
        """Formatted mean/std for reporting."""
        return f"Mean: {self.mean} | Std: {self.std}"

    @property
    def resolution_str(self) -> str:
        """Formatted resolution string (e.g., '28x28', '224x224')."""
        return f"{self.native_resolution}x{self.native_resolution}"

    @property
    def num_classes(self) -> int:
        """Total number of target classes."""
        return len(self.classes)

    def __repr__(self) -> str:
        return (
            f"<DatasetMetadata: {self.display_name} "
            f"({self.resolution_str}, {self.num_classes} classes)>"
        )
