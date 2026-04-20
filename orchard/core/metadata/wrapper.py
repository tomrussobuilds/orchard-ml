"""
Pydantic Wrappers for Multi-Domain Dataset Registries.

Type-safe, validated access to dataset domains organized by task type
(classification, detection) and resolution. Each wrapper subclass
merges its own domain registries while avoiding global metadata overwrites.

Use ``get_registry(resolution, task_type)`` to obtain the correct wrapper.
"""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..paths import SUPPORTED_RESOLUTIONS
from .base import DatasetMetadata
from .domains import (
    BENCHMARK_32,
    MEDICAL_28,
    MEDICAL_64,
    MEDICAL_128,
    MEDICAL_224,
    SPACE_224,
)
from .domains.detection import PENNFUDAN_224

# Classification: resolution → domain registries
_CLASSIFICATION_REGISTRIES: dict[int, tuple[dict[str, DatasetMetadata], ...]] = {
    28: (MEDICAL_28,),
    32: (BENCHMARK_32,),
    64: (MEDICAL_64,),
    128: (MEDICAL_128,),
    224: (MEDICAL_224, SPACE_224),
}

# Detection: resolution → domain registries
_DETECTION_REGISTRIES: dict[int, tuple[dict[str, DatasetMetadata], ...]] = {
    224: (PENNFUDAN_224,),
}


class DatasetRegistryWrapper(BaseModel):
    """
    Base wrapper for dataset registries.

    Provides resolution validation, deep-copied access, and the
    ``get_dataset`` lookup method. Subclasses define which domain
    registries are available.

    Attributes:
        resolution: Target dataset resolution.
        registry: Deep-copied metadata registry for the selected resolution.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    resolution: int = Field(default=28, description="Target resolution (28, 32, 64, 128, or 224)")

    registry: dict[str, DatasetMetadata] = Field(
        default_factory=dict, description="Dataset registry for selected resolution"
    )

    @classmethod
    def _get_dispatch_table(cls) -> dict[int, tuple[dict[str, DatasetMetadata], ...]]:
        """Return the resolution → registries dispatch table for this wrapper."""
        return _CLASSIFICATION_REGISTRIES  # pragma: no mutate

    @model_validator(mode="before")
    @classmethod
    def _load_registry(cls, values: dict[str, Any]) -> dict[str, Any]:
        """
        Load and merge domain registries based on resolution.

        Validates resolution and creates deep copy to prevent mutation.
        """
        res = values.get("resolution", 28)

        if res not in SUPPORTED_RESOLUTIONS:
            raise ValueError(
                f"Unsupported resolution {res}. Supported: {sorted(SUPPORTED_RESOLUTIONS)}"
            )

        dispatch = cls._get_dispatch_table()
        registries = dispatch.get(res, ())
        merged: dict[str, DatasetMetadata] = {}
        for registry in registries:
            merged.update(registry)

        if not merged:
            raise ValueError(f"No datasets available at resolution {res} for this task type")

        values["resolution"] = res
        values["registry"] = copy.deepcopy(merged)

        return values

    def get_dataset(self, name: str) -> DatasetMetadata:
        """
        Retrieve a DatasetMetadata entry by name.

        Args:
            name: Dataset identifier.

        Returns:
            Deep copy of the matching DatasetMetadata.

        Raises:
            KeyError: If dataset not found in registry.
        """
        if name not in self.registry:
            available = list(self.registry.keys())
            raise KeyError(f"Dataset '{name}' not found. Available: {available}")

        return copy.deepcopy(self.registry[name])


class ClassificationRegistryWrapper(DatasetRegistryWrapper):
    """Registry wrapper for classification datasets (medical, space, benchmark)."""

    @classmethod
    def _get_dispatch_table(cls) -> dict[int, tuple[dict[str, DatasetMetadata], ...]]:
        """Return classification domain registries."""
        return _CLASSIFICATION_REGISTRIES  # pragma: no mutate


class DetectionRegistryWrapper(DatasetRegistryWrapper):
    """Registry wrapper for detection datasets."""

    @classmethod
    def _get_dispatch_table(cls) -> dict[int, tuple[dict[str, DatasetMetadata], ...]]:
        """Return detection domain registries."""
        return _DETECTION_REGISTRIES  # pragma: no mutate


def get_registry(
    resolution: int,
    task_type: str = "classification",
) -> DatasetRegistryWrapper:
    """
    Factory function to obtain the correct registry wrapper for a task.

    Args:
        resolution: Target image resolution.
        task_type: ``"classification"`` or ``"detection"``.

    Returns:
        Registry wrapper with datasets available for the given task and resolution.

    Raises:
        ValueError: If ``task_type`` is not ``"classification"`` or ``"detection"``.
    """
    if task_type == "detection":
        return DetectionRegistryWrapper(resolution=resolution)
    if task_type != "classification":
        raise ValueError(
            f"Unknown task_type: {task_type!r}. Expected 'classification' or 'detection'."
        )
    return ClassificationRegistryWrapper(resolution=resolution)
