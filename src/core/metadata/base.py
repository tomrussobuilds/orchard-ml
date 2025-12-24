"""
Dataset Metadata Base Definitions

This module defines the schema for dataset metadata using NamedTuples 
to ensure immutability and type safety across the Orchard.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
from typing import NamedTuple, List
from pathlib import Path

# =========================================================================== #
#                                METADATA SCHEMA                              #
# =========================================================================== #


class DatasetMetadata(NamedTuple):
    """
    Metadata container for a MedMNIST dataset.
    
    This structure ensures that all dataset-specific constants are grouped
    and immutable throughout the execution of the pipeline.
    """
    name: str
    display_name: str
    md5_checksum: str
    url: str
    path: Path
    classes: List[str]
    mean: tuple
    std: tuple
    in_channels: int