"""
Pydantic Wrapper for Multi-Domain Dataset Registries.

type-safe, validated access to multiple dataset domains (medical, space)
and resolutions (28x28, 64x64, 224x224). Merges domain registries based on
selected resolution while avoiding global metadata overwrites.
"""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..paths import SUPPORTED_RESOLUTIONS
from .base import DatasetMetadata
from .domains import BENCHMARK_32, MEDICAL_28, MEDICAL_64, MEDICAL_224, SPACE_224

# Resolution → registry merge map (add new resolutions here)
_RESOLUTION_REGISTRIES: dict[int, tuple[dict[str, DatasetMetadata], ...]] = {
    28: (MEDICAL_28,),
    32: (BENCHMARK_32,),
    64: (MEDICAL_64,),
    224: (MEDICAL_224, SPACE_224),
}


# WRAPPER DEFINITION
class DatasetRegistryWrapper(BaseModel):
    """
    Pydantic wrapper for multi-domain dataset registries.

    Merges domain-specific registries (medical, space) based on the
    selected resolution and provides validated, deep-copied access to
    dataset metadata entries.

    Attributes:
        resolution: Target dataset resolution (28, 64, or 224).
        registry: Deep-copied metadata registry for the selected resolution.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    resolution: int = Field(default=28, description="Target resolution (28, 32, 64, or 224)")

    registry: dict[str, DatasetMetadata] = Field(
        default_factory=dict, description="Dataset registry for selected resolution"
    )

    @model_validator(mode="before")
    @classmethod
    def _load_registry(cls, values: dict[str, Any]) -> dict[str, Any]:
        """
        Loads and merges domain registries based on resolution.

        Validates resolution and creates deep copy to prevent mutation.
        """
        res = values.get("resolution", 28)

        if res not in SUPPORTED_RESOLUTIONS:
            raise ValueError(
                f"Unsupported resolution {res}. Supported: {sorted(SUPPORTED_RESOLUTIONS)}"
            )

        # Merge domain registries via dispatch table
        registries = _RESOLUTION_REGISTRIES[res]
        merged: dict[str, DatasetMetadata] = {}
        for registry in registries:
            merged.update(registry)

        if not merged:
            raise ValueError(f"Dataset registry for resolution {res} is empty")

        values["resolution"] = res
        values["registry"] = copy.deepcopy(merged)

        return values

    def get_dataset(self, name: str) -> DatasetMetadata:
        """
        Retrieves specific DatasetMetadata by name.

        Args:
            name: Dataset identifier

        Returns:
            Deep copy of DatasetMetadata

        Raises:
            KeyError: If dataset not found in registry
        """
        if name not in self.registry:
            available = list(self.registry.keys())
            raise KeyError(f"Dataset '{name}' not found. Available: {available}")

        return copy.deepcopy(self.registry[name])
