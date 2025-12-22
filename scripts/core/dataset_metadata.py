"""
Dataset Registry and Metadata Module

This module centralizes the metadata for all supported datasets (e.g., BloodMNIST, 
DermaMNIST). It uses a NamedTuple-based registry to ensure immutability, 
type safety, and easy scalability when adding new MedMNIST datasets.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
from typing import NamedTuple, List, Dict, Final
from pathlib import Path

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from .constants import DATASET_DIR

# =========================================================================== #
#                                DATASET METADATA                             #
# =========================================================================== #

class DatasetMetadata(NamedTuple):
    """
    Immutable container for dataset-specific metadata.

    Attributes:
        name (str): Lowercase unique identifier (e.g., 'bloodmnist').
        display_name (str): Human-readable name for plots and reports.
        md5_checksum (str): MD5 hash to verify file integrity.
        url (str): Direct download link to the .npz file.
        classes (List[str]): Ordered list of class labels.
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

    @property
    def num_classes(self) -> int:
        """Infers the number of classes."""
        return len(self.classes)

DATASET_REGISTRY: Final[Dict[str, DatasetMetadata]] = {
    "bloodmnist": DatasetMetadata(
        name="bloodmnist",
        display_name="BloodMNIST",
        md5_checksum="7053d0359d879ad8a5505303e11de1dc",
        url="https://zenodo.org/record/5208230/files/bloodmnist.npz?download=1",
        path=DATASET_DIR / "bloodmnist.npz",
        classes=[
            "basophil",
            "eosinophil",
            "erythroblast",
            "immature granulocyte",
            "lymphocyte",
            "monocyte",
            "neutrophil",
            "platelet"
        ],
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
        in_channels=3,
    ),
    
    "dermamnist": DatasetMetadata(
        name="dermamnist",
        display_name="DermaMNIST",
        md5_checksum="0744692d530f8e62bc4730caddabc09d",
        url="https://zenodo.org/record/5208230/files/dermamnist.npz?download=1",
        path=DATASET_DIR / "dermamnist.npz",
        classes=[
            "actinic keratoses",
            "basal cell carcinoma",
            "benign keratosis",
            "dermatofibroma",
            "melanocytic nevi",
            "melanoma",
            "vascular lesions"
        ],
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        in_channels=3,
    ),
}