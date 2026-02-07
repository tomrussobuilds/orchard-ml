"""
Dataset Metadata Base Definitions.

Defines dataset metadata schema using Pydantic for immutability, type safety,
and seamless integration with the global configuration engine.
"""

from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


# METADATA SCHEMA
class DatasetMetadata(BaseModel):
    """
    Metadata container for a MedMNIST dataset.

    Ensures dataset-specific constants are grouped and immutable throughout
    pipeline execution. Serves as static definition feeding into dynamic
    DatasetConfig.
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
    classes: List[str] = Field(..., description="Class labels in index order")

    # Image properties
    in_channels: int = Field(..., description="1 for grayscale, 3 for RGB")
    native_resolution: Optional[int] = Field(
        default=None, description="Native pixel resolution (28 or 224)"
    )

    # Normalization
    mean: Tuple[float, ...] = Field(..., description="Channel-wise mean")
    std: Tuple[float, ...] = Field(..., description="Channel-wise std")

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
