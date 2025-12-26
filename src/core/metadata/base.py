"""
Dataset Metadata Base Definitions

This module defines the schema for dataset metadata using Pydantic 
to ensure immutability, type safety, and seamless integration 
with the global configuration engine.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
from typing import List, Tuple
from pathlib import Path

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
from pydantic import BaseModel, Field, ConfigDict

# =========================================================================== #
#                                METADATA SCHEMA                              #
# =========================================================================== #

class DatasetMetadata(BaseModel):
    """
    Metadata container for a MedMNIST dataset.
    
    This structure ensures that all dataset-specific constants are grouped
    and immutable throughout the execution of the pipeline. It serves as
    the static definition that feeds into the dynamic DatasetConfig.
    """
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True
    )

    name: str = Field(..., description="Short identifier (e.g., 'pathmnist')")
    display_name: str = Field(..., description="Full dataset name for reporting")
    md5_checksum: str = Field(..., description="MD5 hash for archive integrity")
    url: str = Field(..., description="Source URL for automated downloads")
    path: Path = Field(..., description="Relative or absolute path to the .npz archive")
    
    classes: List[str] = Field(..., description="List of class labels in index order")
    in_channels: int = Field(..., description="1 for Grayscale, 3 for RGB")
    
    # Normalization parameters
    mean: Tuple[float, ...] = Field(..., description="Channel-wise mean for normalization")
    std: Tuple[float, ...] = Field(..., description="Channel-wise std for normalization")
    
    # Behavioral flags
    is_anatomical: bool = Field(
        default=True, 
        description="True if the dataset has a fixed anatomical orientation (e.g., ChestMNIST)"
    )

    def __repr__(self) -> str:
        return f"<DatasetMetadata: {self.display_name} ({len(self.classes)} classes)>"