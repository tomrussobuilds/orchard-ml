"""
PennFudan Pedestrian Detection Dataset Registry Definition.

Contains DatasetMetadata for the Penn-Fudan pedestrian detection dataset
at 224x224 resolution. Images are resized and bounding boxes rescaled
during the fetch/conversion step.

Source: https://www.cis.upenn.edu/~jshi/ped_html/
"""

from __future__ import annotations

from typing import Final

from ....paths import DATASET_DIR
from ...base import DatasetMetadata

REGISTRY_224: Final[dict[str, DatasetMetadata]] = {
    "pennfudan": DatasetMetadata(
        name="pennfudan",
        display_name="PennFudan Pedestrians",
        md5_checksum="",
        url="https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip",
        path=DATASET_DIR / "pennfudan_224_images.npz",
        classes=["person"],
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        in_channels=3,
        native_resolution=224,
        # Classification-only flags; detection pipeline never reads them
        is_anatomical=False,
        is_texture_based=False,
        annotation_path=DATASET_DIR / "pennfudan_224_annotations.npz",
    ),
}
