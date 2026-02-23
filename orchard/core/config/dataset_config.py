"""
Dataset Registry Orchestration & Metadata Resolution.

Bridges static dataset metadata with runtime execution requirements. Normalizes
datasets regardless of native format (Grayscale/RGB) to meet model architecture
input specifications. Supports multi-resolution (28x28, 224x224) with proper
YAML override while maintaining frozen immutability.

Key Responsibilities:
    * Adaptive normalization: Adjusts mean/std based on channel logic
    * Feature promotion: Automates grayscale-to-RGB for ImageNet weights
    * Resource budgeting: Enforces sampling limits and class balancing
    * Multi-resolution support: Resolves metadata by selected resolution
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..metadata import DatasetMetadata, DatasetRegistryWrapper
from ..paths import DATASET_DIR, SUPPORTED_RESOLUTIONS
from .types import ImageSize, PositiveInt, Probability, ValidatedPath


# DATASET CONFIGURATION
class DatasetConfig(BaseModel):
    """
    Validated manifest for dataset execution context.

    Bridges static registry metadata with runtime preferences. Resolves
    channel promotion and sampling policies with multi-resolution support.
    Auto-syncs img_size with resolution when not explicitly set.

    Attributes:
        name: Dataset identifier from registry (e.g., 'bloodmnist', 'organcmnist').
        metadata: DatasetMetadata object (excluded from serialization).
        data_root: Root directory containing dataset files.
        use_weighted_sampler: Enable class-balanced sampling for imbalanced datasets.
        max_samples: Maximum samples to load (None=all).
        val_ratio: Fraction of max_samples for val/test splits (default 0.10).
        img_size: Target square resolution for model input (auto-synced).
        force_rgb: Convert grayscale to RGB for pretrained ImageNet weights.
        resolution: Target resolution variant (28 or 224).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    name: str = Field(
        default="bloodmnist", description="Dataset identifier (e.g., 'bloodmnist', 'organcmnist')"
    )
    metadata: DatasetMetadata | None = Field(default=None, exclude=True)

    # Runtime parameters
    data_root: ValidatedPath = DATASET_DIR
    use_weighted_sampler: bool = True
    max_samples: PositiveInt | None = Field(default=None)
    val_ratio: Probability = Field(
        default=0.10,
        description="Fraction of max_samples used for val/test splits (0.0-1.0)",
    )

    img_size: ImageSize | None = Field(
        description="Target square resolution for model input",
        default=None,
    )
    force_rgb: bool = Field(
        default=True, description="Convert grayscale to RGB for ImageNet weights"
    )
    resolution: int = Field(
        default=28,
        description=f"Target dataset resolution {sorted(SUPPORTED_RESOLUTIONS)}",
    )

    @field_validator("max_samples")
    @classmethod
    def validate_min_samples(cls, v: int | None) -> int | None:
        """Enforce minimum sample count for meaningful train/val/test splits."""
        if v is not None and v < 20:
            raise ValueError(
                f"max_samples={v} is too small for meaningful train/val/test splits. "
                f"Use max_samples >= 20 or None to load all samples."
            )
        return v

    @model_validator(mode="before")
    @classmethod
    def sync_img_size_with_resolution(cls, values: dict[str, Any]) -> dict[str, Any]:
        """
        Auto-sync img_size with resolution if not explicitly set.

        This runs BEFORE frozen instantiation, allowing us to modify values.

        Logic:
        1. If img_size is explicitly set in YAML/args → keep it
        2. If img_size is None/missing → use resolution
        3. If metadata exists → use metadata.native_resolution

        Args:
            values (dict[str, Any]): Raw input dict before Pydantic validation

        Returns:
            dict[str, Any]: Modified values dict with synced img_size
        """
        img_size = values.get("img_size")
        resolution = values.get("resolution", 28)
        metadata = values.get("metadata")

        if img_size is None:
            if metadata is not None:
                # Use metadata's native resolution
                img_size = metadata.native_resolution
            else:
                # Use resolution parameter
                img_size = resolution

            values["img_size"] = img_size

        return values

    # --- Properties ---

    @property
    def _ensure_metadata(self) -> DatasetMetadata:
        """
        Return metadata, lazily loading from registry if None.

        **Safety net only**: the primary path (``Config.from_recipe``)
        injects metadata before frozen instantiation.  This lazy
        fallback exists for direct ``DatasetConfig()`` construction
        in tests or standalone usage.  Uses ``object.__setattr__``
        to bypass frozen restriction for one-time initialization.

        Returns:
            DatasetMetadata object for this dataset.
        """
        if self.metadata is None:
            wrapper = DatasetRegistryWrapper(resolution=self.resolution)
            ds_name = self.name
            metadata = wrapper.get_dataset(ds_name)
            object.__setattr__(self, "metadata", metadata)
        return self.metadata  # type: ignore[return-value]

    @property
    def dataset_name(self) -> str:
        """
        Get dataset identifier from metadata.

        Returns:
            Dataset name string (e.g., 'bloodmnist').
        """
        return self._ensure_metadata.name

    @property
    def num_classes(self) -> int:
        """
        Get number of target classes.

        Returns:
            Integer count of classification classes.
        """
        return self._ensure_metadata.num_classes

    @property
    def in_channels(self) -> int:
        """
        Get native input channels from metadata.

        Returns:
            1 for grayscale datasets, 3 for RGB.
        """
        return self._ensure_metadata.in_channels

    @property
    def effective_in_channels(self) -> int:
        """
        Get actual channels the model receives after promotion.

        Returns:
            3 if force_rgb enabled, otherwise native in_channels.
        """
        return 3 if self.force_rgb else self.in_channels

    @property
    def mean(self) -> tuple[float, ...]:
        """
        Get channel-wise normalization mean.

        Expands single-channel mean to 3 channels if force_rgb is enabled
        on a grayscale dataset.

        Returns:
            tuple of mean values per channel.
        """
        m = self._ensure_metadata.mean
        return (m[0],) * 3 if self.force_rgb and self.in_channels == 1 else m

    @property
    def std(self) -> tuple[float, ...]:
        """
        Get channel-wise normalization standard deviation.

        Expands single-channel std to 3 channels if force_rgb is enabled
        on a grayscale dataset.

        Returns:
            tuple of std values per channel.
        """
        s = self._ensure_metadata.std
        return (s[0],) * 3 if self.force_rgb and self.in_channels == 1 else s

    @property
    def processing_mode(self) -> str:
        """
        Get description of channel processing mode.

        Returns:
            'NATIVE-RGB' for RGB datasets, 'RGB-PROMOTED' for grayscale
            with force_rgb, or 'NATIVE-GRAY' for grayscale without promotion.
        """
        if self.in_channels == 3:
            return "NATIVE-RGB"
        return "RGB-PROMOTED" if self.effective_in_channels == 3 else "NATIVE-GRAY"
