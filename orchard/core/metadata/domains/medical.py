"""
Medical Imaging Dataset Registry Definitions.

Loads DatasetMetadata instances from medical.yaml for the MedMNIST v2
collection at 28x28, 64x64, 128x128, and 224x224 resolutions. All
datasets sourced from Zenodo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import yaml

from ...paths import DATASET_DIR
from ..base import DatasetMetadata

_YAML_PATH: Final[Path] = Path(__file__).parent / "medical.yaml"


def _load_registries() -> dict[int, dict[str, DatasetMetadata]]:
    """
    Build per-resolution registries from the YAML manifest.

    Parses shared dataset properties (classes, channels, normalization,
    flags) once per dataset, then creates resolution-specific
    DatasetMetadata instances with the appropriate md5/url/path.

    Returns:
        Mapping of resolution → {dataset_name → DatasetMetadata}.
    """
    data = yaml.safe_load(_YAML_PATH.read_text())
    registries: dict[int, dict[str, DatasetMetadata]] = {}

    for ds_name, ds_info in data["datasets"].items():
        shared = {
            "name": ds_name,
            "display_name": ds_info["display_name"],
            "classes": ds_info["classes"],
            "mean": tuple(ds_info["mean"]),
            "std": tuple(ds_info["std"]),
            "in_channels": ds_info["in_channels"],
            "is_anatomical": ds_info["is_anatomical"],
            "is_texture_based": ds_info["is_texture_based"],
        }

        for res, res_info in ds_info["resolutions"].items():
            res = int(res)
            registries.setdefault(res, {})[ds_name] = DatasetMetadata(
                **shared,
                md5_checksum=res_info["md5"],
                url=res_info["url"],
                path=DATASET_DIR / f"{ds_name}_{res}.npz",
                native_resolution=res,
            )

    return registries


_ALL_REGISTRIES: Final[dict[int, dict[str, DatasetMetadata]]] = _load_registries()

REGISTRY_28: Final[dict[str, DatasetMetadata]] = _ALL_REGISTRIES[28]
REGISTRY_64: Final[dict[str, DatasetMetadata]] = _ALL_REGISTRIES[64]
REGISTRY_128: Final[dict[str, DatasetMetadata]] = _ALL_REGISTRIES[128]
REGISTRY_224: Final[dict[str, DatasetMetadata]] = _ALL_REGISTRIES[224]
