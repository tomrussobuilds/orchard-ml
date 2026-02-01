"""
Space and Celestial Bodies Dataset Registry Definitions.

Contains DatasetMetadata for astronomical imagery and galaxy classification.
Currently supports Galaxy10 DECals at 224x224 resolution (resized from 256x256 native).
"""

from typing import Dict, Final

from ...paths import DATASET_DIR
from ..base import DatasetMetadata

# SPACE DATASET REGISTRY (224x224)
REGISTRY_224: Final[Dict[str, DatasetMetadata]] = {
    "galaxy10": DatasetMetadata(
        name="galaxy10",
        display_name="Galaxy10 DECals",
        md5_checksum="d288075a410ae999036bec39a53d8552",
        url="https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5?download=1",
        path=DATASET_DIR / "galaxy10_224.npz",
        classes=[
            "DisturbedGalaxies",
            "MergingGalaxies",
            "RoundSmoothGalaxies",
            "InBetweenRoundSmoothGalaxies",
            "CigarShapedSmoothGalaxies",
            "BarredSpiralGalaxies",
            "UnbarredTightSpiralGalaxies",
            "UnbarredLooseSpiralGalaxies",
            "EdgeOnWithoutBulge",
            "EdgeOnWithBulge",
        ],
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        in_channels=3,
        native_resolution=224,
        is_anatomical=False,
        is_texture_based=True,
    ),
}
