"""
Dataset Registry Resolver & Metadata Schema.

This module acts as the interface between the MedMNIST Metadata Registry 
and the training pipeline. It handles the dynamic resolution of dataset-specific 
constraints, such as channel counts, normalization constants (mean/std), 
and automated 3-channel expansion (RGB) for grayscale datasets when 
using pretrained models.

Key Architectural Features:
    * Metadata Binding: Resolves dataset names to official MedMNIST properties.
    * Channel Orchestration: Manages 'force_rgb' logic to ensure compatibility 
      between grayscale medical images and ImageNet-pretrained architectures.
    * Class Balance Logic: Configures Weighted Sampler settings based on 
      registry-provided class distributions.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
import argparse
from typing import Optional

# =========================================================================== #
#                                Third-Party Imports                          #
# =========================================================================== #
from pydantic import BaseModel, Field, ConfigDict

# =========================================================================== #
#                               Internal Imports                              #
# =========================================================================== #
from .types import (
    ImageSize, Channels, ValidatedPath, PositiveInt
)
from ..metadata import DATASET_REGISTRY
from ..paths import DATASET_DIR

# =========================================================================== #
#                             DATASET CONFIGURATION                           #
# =========================================================================== #

class DatasetConfig(BaseModel):
    """
    Resolves dataset-specific constraints and normalization metadata.
    
    Acts as the interface between the MedMNIST Metadata Registry and the 
    training pipeline, handling automated 3-channel expansion for 
    pretrained models and class-balance sampling logic.
    """
    model_config = ConfigDict(
        frozen=True,
        extra="forbid"
    )
    
    data_root: ValidatedPath = DATASET_DIR
    dataset_name: str = "BloodMNIST"
    use_weighted_sampler: bool = True
    in_channels: Channels = 3
    num_classes: PositiveInt = Field(
        default=8,
        description="Number of target classes in the dataset"
    )
    max_samples: Optional[PositiveInt] = Field(default=20000)
    img_size: ImageSize = Field(
        default=28,
        description="Target square resolution for the model input"
    )
    force_rgb: bool = Field(
        default=True,
        description="Convert grayscale to 3-channel to enable ImageNet weights"
    )
    mean: tuple[float, ...] = Field(
        default=(0.5, 0.5, 0.5),
        description="Channel-wise mean for normalization",
        min_length=1,
        max_length=3
    )
    std: tuple[float, ...] = Field(
        default=(0.5, 0.5, 0.5),
        description="Channel-wise std for normalization",
        min_length=1,
        max_length=3
    )
    normalization_info: str = "N/A"
    is_anatomical: bool = True
    is_texture_based: bool = True
    
    @property
    def is_grayscale(self) -> bool:
        """Indicates if the original dataset images are single-channel."""
        return self.in_channels == 1

    @property
    def effective_in_channels(self) -> int:
        """Returns the actual number of channels the model will see"""
        return 3 if self.force_rgb else self.in_channels
    
    @staticmethod
    def _resolve_dataset_metadata(dataset_raw: str):
        """Retrieve static metadata for the dataset from the central registry."""
        key = dataset_raw.lower()
        if key not in DATASET_REGISTRY:
            raise ValueError(f"Dataset '{dataset_raw}' not supported in DATASET_REGISTRY.")
        return DATASET_REGISTRY[key]
    
    @property
    def processing_mode(self) -> str:
        """
        Determina la modalità di elaborazione canali in base al dataset e alla config.
        """
        if self.in_channels == 3:
            return "NATIVE-RGB"
        if self.effective_in_channels == 3:
            return "RGB-PROMOTED"
        return "NATIVE-GRAY"

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "DatasetConfig":
        """Map dataset-specific metadata and resolve conditional RGB/sampling logic."""
        
        ds_name = getattr(args, 'dataset', "BloodMNIST")
        ds_meta = cls._resolve_dataset_metadata(ds_name)
            
        # Determine RGB logic: User override or automatic for pretrained grayscale
        force_rgb_arg = getattr(args, 'force_rgb', None)
        should_force_rgb = force_rgb_arg if force_rgb_arg is not None else \
                            (ds_meta.in_channels == 1 and getattr(args, 'pretrained', False))
            
        # Determine final max_samples value
        final_max_samples = args.max_samples if (getattr(args, 'max_samples', 0) > 0) else None

        return cls(
            dataset_name=ds_meta.name,
            max_samples=final_max_samples,
            use_weighted_sampler=getattr(args, 'use_weighted_sampler', True),
            in_channels=ds_meta.in_channels,
            num_classes=len(ds_meta.classes),
            mean=ds_meta.mean,
            std=ds_meta.std,
            normalization_info=f"Mean={ds_meta.mean}, Std={ds_meta.std}",
            is_anatomical=ds_meta.is_anatomical,
            is_texture_based=ds_meta.is_texture_based,
            force_rgb=should_force_rgb,
            img_size=getattr(args, 'img_size', 28)
        )