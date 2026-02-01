"""
Pydantic Wrapper for Multi-Domain Dataset Registries.

Type-safe, validated access to multiple dataset domains (medical, space)
and resolutions (28x28, 224x224). Merges domain registries based on selected
resolution while avoiding global metadata overwrites.
"""

import copy
from typing import Dict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .base import DatasetMetadata
from .domains import MEDICAL_28, MEDICAL_224, SPACE_224


# WRAPPER DEFINITION
class DatasetRegistryWrapper(BaseModel):
    """
    Pydantic wrapper for dynamic dataset registries.

    Attributes:
        resolution: Target dataset resolution (28 or 224)
        registry: Deep copy of metadata registry for selected resolution
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    resolution: int = Field(default=28, description="Target resolution (28 or 224)")

    registry: Dict[str, DatasetMetadata] = Field(
        default_factory=dict, description="Dataset registry for selected resolution"
    )

    @model_validator(mode="before")
    @classmethod
    def _load_registry(cls, values):
        """
        Loads and merges domain registries based on resolution.
        Validates resolution and creates deep copy to prevent mutation.
        """
        res = values.get("resolution", 28)

        if res not in (28, 224):
            raise ValueError(f"Unsupported resolution {res}. Supported: [28, 224]")

        # Merge domain registries based on resolution
        if res == 28:
            merged = {**MEDICAL_28}
        else:  # res == 224
            merged = {**MEDICAL_224, **SPACE_224}

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


# Default wrapper for backward compatibility (28x28 resolution)
DEFAULT_WRAPPER = DatasetRegistryWrapper(resolution=28)
