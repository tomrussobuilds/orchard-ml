"""
Pydantic Wrapper for Dynamic MedMNIST Dataset Registries.

Type-safe, validated access to multiple dataset resolutions (28x28 or 224x224)
while avoiding global metadata overwrites. Integrates with YAML configs and
supports runtime resolution selection.
"""

# Standard Imports
import copy
from typing import Dict

# Third-Party Imports
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Internal Imports
from .base import DatasetMetadata
from .medmnist_v2_28x28 import DATASET_REGISTRY as REG_28
from .medmnist_v2_224x224 import DATASET_REGISTRY as REG_224


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
        Loads appropriate registry based on resolution.
        Validates resolution and creates deep copy to prevent mutation.
        """
        res = values.get("resolution", 28)  # Default to 28 if not specified

        if res not in (28, 224):
            raise ValueError(f"Unsupported resolution {res}. Supported: [28, 224]")

        source = REG_28 if res == 28 else REG_224

        if not source:
            raise ValueError(f"Dataset registry for resolution {res} is empty")

        values["resolution"] = res
        values["registry"] = copy.deepcopy(source)

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
